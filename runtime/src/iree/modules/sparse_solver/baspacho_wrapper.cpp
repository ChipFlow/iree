// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/sparse_solver/baspacho_wrapper.h"

#include <cstring>
#include <memory>
#include <vector>

// BaSpaCho C++ API
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"

// GPU buffer registries for zero-copy integration
#ifdef BASPACHO_USE_METAL
#include "baspacho/baspacho/MetalDefs.h"
#endif

#ifdef BASPACHO_USE_OPENCL
#include "baspacho/baspacho/OpenCLDefs.h"
#endif

//===----------------------------------------------------------------------===//
// Context Structure
//===----------------------------------------------------------------------===//

struct baspacho_context_s {
  // Backend configuration
  BaSpaCho::BackendType backend;

  // Symbolic analysis result (solver skeleton)
  BaSpaCho::SolverPtr solver;

  // Sparse structure info
  int64_t n;    // Matrix dimension
  int64_t nnz;  // Number of non-zeros

  // Original CSR structure (needed for loadFromCsr)
  std::vector<int64_t> csr_row_ptr;
  std::vector<int64_t> csr_col_idx;

  // Permutation for reordering
  std::vector<int64_t> permutation;
  std::vector<int64_t> inverse_permutation;

  // Block sizes (assuming scalar blocks for CSR input)
  std::vector<int64_t> block_sizes;

  // Factor data storage (for numeric factorization)
  // For CPU backend, use std::vector
  std::vector<float> factor_data_f32;
  std::vector<double> factor_data_f64;

#ifdef BASPACHO_USE_METAL
  // For Metal backend, use MetalMirror which auto-registers with registry
  std::unique_ptr<BaSpaCho::MetalMirror<float>> metal_factor_data_f32;
  // Temporary buffer for permuted RHS/solution during solve
  std::unique_ptr<BaSpaCho::MetalMirror<float>> metal_permuted_f32;
#endif

  // GPU device handle (if applicable)
  void* device_handle;
  void* command_queue;
};

//===----------------------------------------------------------------------===//
// Backend Conversion
//===----------------------------------------------------------------------===//

static BaSpaCho::BackendType convert_backend(baspacho_backend_t backend) {
  switch (backend) {
    case BASPACHO_BACKEND_CPU:
      return BaSpaCho::BackendFast;
    case BASPACHO_BACKEND_CUDA:
      return BaSpaCho::BackendCuda;
    case BASPACHO_BACKEND_METAL:
      return BaSpaCho::BackendMetal;
    case BASPACHO_BACKEND_OPENCL:
      return BaSpaCho::BackendOpenCL;
    case BASPACHO_BACKEND_AUTO:
    default:
#if defined(__APPLE__) && defined(__aarch64__) && defined(BASPACHO_USE_METAL)
      // Use Metal backend on Apple Silicon for GPU-accelerated factorization.
      return BaSpaCho::BackendMetal;
#else
      return BaSpaCho::detectBestBackend();
#endif
  }
}

//===----------------------------------------------------------------------===//
// Context Management
//===----------------------------------------------------------------------===//

baspacho_handle_t baspacho_create(baspacho_backend_t backend) {
  auto* ctx = new baspacho_context_s();
  ctx->backend = convert_backend(backend);
  ctx->n = 0;
  ctx->nnz = 0;
  ctx->device_handle = nullptr;
  ctx->command_queue = nullptr;
  return ctx;
}

baspacho_handle_t baspacho_create_with_device(baspacho_backend_t backend,
                                               void* device_handle) {
  auto* ctx = baspacho_create(backend);
  ctx->device_handle = device_handle;
  return ctx;
}

void baspacho_destroy(baspacho_handle_t h) {
  if (h) {
    delete h;
  }
}

baspacho_backend_t baspacho_get_backend(baspacho_handle_t h) {
  if (!h) return BASPACHO_BACKEND_CPU;

  switch (h->backend) {
    case BaSpaCho::BackendFast:
      return BASPACHO_BACKEND_CPU;
    case BaSpaCho::BackendCuda:
      return BASPACHO_BACKEND_CUDA;
    case BaSpaCho::BackendMetal:
      return BASPACHO_BACKEND_METAL;
    case BaSpaCho::BackendOpenCL:
      return BASPACHO_BACKEND_OPENCL;
    default:
      return BASPACHO_BACKEND_CPU;
  }
}

//===----------------------------------------------------------------------===//
// Symbolic Analysis
//===----------------------------------------------------------------------===//

int baspacho_analyze(baspacho_handle_t h, int64_t n, int64_t nnz,
                     const int64_t* row_ptr, const int64_t* col_idx) {
  if (!h || !row_ptr || !col_idx || n <= 0 || nnz <= 0) {
    return -1;
  }

  try {
    h->n = n;
    h->nnz = nnz;

    // Store original CSR structure for later use in loadFromCsr
    h->csr_row_ptr.assign(row_ptr, row_ptr + n + 1);
    h->csr_col_idx.assign(col_idx, col_idx + nnz);

    // For scalar CSR, each block is size 1
    h->block_sizes.assign(n, 1);

    // Create sparse structure from CSR
    BaSpaCho::SparseStructure ss;
    ss.ptrs.assign(row_ptr, row_ptr + n + 1);
    ss.inds.assign(col_idx, col_idx + nnz);

    // Create solver settings
    BaSpaCho::Settings settings;
    settings.backend = h->backend;
    settings.numThreads = 8;  // Reasonable default
    settings.addFillPolicy = BaSpaCho::AddFillComplete;
    // Disable sparse elimination for now - there's a bug with large matrices
    // that causes incorrect results when sparse elimination is enabled.
    settings.findSparseEliminationRanges = false;

    // Create solver (performs symbolic analysis)
    h->solver = BaSpaCho::createSolver(settings, h->block_sizes, ss);

    if (!h->solver) {
      return -2;  // Symbolic analysis failed
    }

    // Store permutation
    h->permutation = h->solver->paramToSpan();
    h->inverse_permutation.resize(n);
    for (int64_t i = 0; i < n; ++i) {
      h->inverse_permutation[h->permutation[i]] = i;
    }

    // Allocate factor storage
    int64_t data_size = h->solver->dataSize();

#ifdef BASPACHO_USE_METAL
    if (h->backend == BaSpaCho::BackendMetal) {
      // For Metal backend, use MetalMirror which auto-registers with registry.
      // This is required because BaSpaCho's Metal backend looks up buffers
      // in MetalBufferRegistry when performing GPU operations.
      h->metal_factor_data_f32 = std::make_unique<BaSpaCho::MetalMirror<float>>();
      h->metal_factor_data_f32->resizeToAtLeast(data_size);
      // Also allocate permuted buffer for solve operations
      h->metal_permuted_f32 = std::make_unique<BaSpaCho::MetalMirror<float>>();
      h->metal_permuted_f32->resizeToAtLeast(n);
    } else {
      h->factor_data_f32.resize(data_size);
    }
#else
    h->factor_data_f32.resize(data_size);
#endif
    h->factor_data_f64.resize(data_size);

    return 0;  // Success
  } catch (const std::exception& e) {
    fprintf(stderr, "BaSpaCho analyze exception: %s\n", e.what());
    return -3;  // Exception during analysis
  }
}

int64_t baspacho_get_factor_nnz(baspacho_handle_t h) {
  if (!h || !h->solver) return 0;
  return h->solver->dataSize();
}

int64_t baspacho_get_num_supernodes(baspacho_handle_t h) {
  if (!h || !h->solver) return 0;
  return h->solver->skel().numSpans();
}

//===----------------------------------------------------------------------===//
// Numeric Factorization
//===----------------------------------------------------------------------===//

int baspacho_factor_f32(baspacho_handle_t h, const float* values) {
  if (!h || !h->solver || !values) return -1;

  try {
    float* data = nullptr;
    size_t data_size = 0;

#ifdef BASPACHO_USE_METAL
    if (h->backend == BaSpaCho::BackendMetal && h->metal_factor_data_f32) {
      // Use MetalMirror buffer for GPU factorization.
      // MetalMirror auto-registers with MetalBufferRegistry.
      data = h->metal_factor_data_f32->ptr();
      data_size = h->metal_factor_data_f32->allocSize();
    } else {
      data = h->factor_data_f32.data();
      data_size = h->factor_data_f32.size();
    }
#else
    data = h->factor_data_f32.data();
    data_size = h->factor_data_f32.size();
#endif

    // Zero out factor storage
    std::memset(data, 0, data_size * sizeof(float));

    // Load CSR values into BaSpaCho's internal format using original CSR structure
    // This handles the permutation and format conversion automatically
    h->solver->loadFromCsr(h->csr_row_ptr.data(), h->csr_col_idx.data(),
                           h->block_sizes.data(), values, data);

    // Perform numeric factorization (GPU or CPU depending on backend)
    h->solver->factor(data);

#ifdef BASPACHO_USE_METAL
    // Metal operations are batched - synchronize to ensure GPU work is complete
    if (h->backend == BaSpaCho::BackendMetal) {
      BaSpaCho::MetalContext::instance().synchronize();
    }
#endif

    return 0;  // Success
  } catch (const std::exception& e) {
    // Log error for debugging
    fprintf(stderr, "BaSpaCho factor_f32 exception: %s\n", e.what());
    return -2;  // Factorization failed
  }
}

int baspacho_factor_f64(baspacho_handle_t h, const double* values) {
  if (!h || !h->solver || !values) return -1;

  try {
    double* data = h->factor_data_f64.data();
    std::memset(data, 0, h->factor_data_f64.size() * sizeof(double));

    // Load CSR values into BaSpaCho's internal format using original CSR structure
    h->solver->loadFromCsr(h->csr_row_ptr.data(), h->csr_col_idx.data(),
                           h->block_sizes.data(), values, data);

    h->solver->factor(data);
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "BaSpaCho factor_f64 exception: %s\n", e.what());
    return -2;
  }
}

int baspacho_factor_f32_device(baspacho_handle_t h, void* device_ptr) {
  // GPU factorization - data already on device
  if (!h || !h->solver || !device_ptr) return -1;

  try {
    h->solver->factor(static_cast<float*>(device_ptr));
    return 0;
  } catch (const std::exception&) {
    return -2;
  }
}

int baspacho_factor_f64_device(baspacho_handle_t h, void* device_ptr) {
  if (!h || !h->solver || !device_ptr) return -1;

  try {
    h->solver->factor(static_cast<double*>(device_ptr));
    return 0;
  } catch (const std::exception&) {
    return -2;
  }
}

//===----------------------------------------------------------------------===//
// LU Factorization
//===----------------------------------------------------------------------===//

int baspacho_factor_lu_f32(baspacho_handle_t h, const float* values,
                            int64_t* pivots) {
  if (!h || !h->solver || !values || !pivots) return -1;

  try {
    float* data = h->factor_data_f32.data();
    std::memset(data, 0, h->factor_data_f32.size() * sizeof(float));

    // Copy values to factor storage (with format conversion if needed)
    std::memcpy(data, values, std::min((size_t)h->nnz,
                h->factor_data_f32.size()) * sizeof(float));

    // Perform LU factorization with partial pivoting
    h->solver->factorLU(data, pivots);
    return 0;
  } catch (const std::exception&) {
    return -2;
  }
}

int baspacho_factor_lu_f64(baspacho_handle_t h, const double* values,
                            int64_t* pivots) {
  if (!h || !h->solver || !values || !pivots) return -1;

  try {
    double* data = h->factor_data_f64.data();
    std::memset(data, 0, h->factor_data_f64.size() * sizeof(double));

    std::memcpy(data, values, std::min((size_t)h->nnz,
                h->factor_data_f64.size()) * sizeof(double));

    h->solver->factorLU(data, pivots);
    return 0;
  } catch (const std::exception&) {
    return -2;
  }
}

int baspacho_factor_lu_f32_device(baspacho_handle_t h, void* device_ptr,
                                   int64_t* pivots) {
  if (!h || !h->solver || !device_ptr || !pivots) return -1;

  try {
    h->solver->factorLU(static_cast<float*>(device_ptr), pivots);
    return 0;
  } catch (const std::exception&) {
    return -2;
  }
}

int baspacho_factor_lu_f64_device(baspacho_handle_t h, void* device_ptr,
                                   int64_t* pivots) {
  if (!h || !h->solver || !device_ptr || !pivots) return -1;

  try {
    h->solver->factorLU(static_cast<double*>(device_ptr), pivots);
    return 0;
  } catch (const std::exception&) {
    return -2;
  }
}

//===----------------------------------------------------------------------===//
// Solve Operations
//===----------------------------------------------------------------------===//

void baspacho_solve_f32(baspacho_handle_t h, const float* rhs, float* solution) {
  if (!h || !h->solver || !rhs || !solution) return;

  try {
    float* factor_data = nullptr;
    float* permuted = nullptr;

    // Debug: force CPU solve to check if Metal solve is the issue
    static bool force_cpu = (getenv("BASPACHO_FORCE_CPU_SOLVE") != nullptr);

#ifdef BASPACHO_USE_METAL
    if (force_cpu && h->backend == BaSpaCho::BackendMetal &&
        h->metal_factor_data_f32) {
      // Copy Metal factor data to CPU for CPU solve test
      // This is slow but useful for debugging
      size_t data_size = h->metal_factor_data_f32->allocSize();
      h->factor_data_f32.resize(data_size);
      BaSpaCho::MetalContext::instance().synchronize();
      std::memcpy(h->factor_data_f32.data(), h->metal_factor_data_f32->ptr(),
                  data_size * sizeof(float));
      fprintf(stderr, "Copied Metal factor data to CPU (%zu elements)\n", data_size);
    }

    if (!force_cpu && h->backend == BaSpaCho::BackendMetal &&
        h->metal_factor_data_f32 && h->metal_permuted_f32) {
      // Use Metal buffers - both are registered with MetalBufferRegistry
      factor_data = h->metal_factor_data_f32->ptr();
      permuted = h->metal_permuted_f32->ptr();

      // Apply permutation to RHS: permuted[p[i]] = rhs[i] (scatter)
      // BaSpaCho's solve does NOT apply permutation internally for Cholesky
      for (int64_t i = 0; i < h->n; ++i) {
        permuted[h->permutation[i]] = rhs[i];
      }

      // Solve in-place using GPU (Metal backend finds buffers in registry)
      h->solver->solve(factor_data, permuted, h->n, 1);

      // Metal operations are batched - synchronize to ensure GPU work is complete
      BaSpaCho::MetalContext::instance().synchronize();

      // Apply inverse permutation to solution: solution[i] = permuted[p[i]] (gather)
      for (int64_t i = 0; i < h->n; ++i) {
        solution[i] = permuted[h->permutation[i]];
      }
    } else {
      // CPU backend path
      factor_data = h->factor_data_f32.data();
      std::vector<float> permuted_vec(h->n);

      // Apply permutation to RHS: permuted[p[i]] = rhs[i] (scatter)
      for (int64_t i = 0; i < h->n; ++i) {
        permuted_vec[h->permutation[i]] = rhs[i];
      }

      // Solve in-place
      h->solver->solve(factor_data, permuted_vec.data(), h->n, 1);

      // Apply inverse permutation to solution: solution[i] = permuted[p[i]] (gather)
      for (int64_t i = 0; i < h->n; ++i) {
        solution[i] = permuted_vec[h->permutation[i]];
      }
    }
#else
    // CPU-only build
    factor_data = h->factor_data_f32.data();
    std::vector<float> permuted_vec(h->n);

    // Apply permutation to RHS: permuted[p[i]] = rhs[i] (scatter)
    for (int64_t i = 0; i < h->n; ++i) {
      permuted_vec[h->permutation[i]] = rhs[i];
    }

    // Solve in-place
    h->solver->solve(factor_data, permuted_vec.data(), h->n, 1);

    // Apply inverse permutation to solution: solution[i] = permuted[p[i]] (gather)
    for (int64_t i = 0; i < h->n; ++i) {
      solution[i] = permuted_vec[h->permutation[i]];
    }
#endif
  } catch (const std::exception&) {
    // Silently fail - could log error
  }
}

void baspacho_solve_f64(baspacho_handle_t h, const double* rhs, double* solution) {
  if (!h || !h->solver || !rhs || !solution) return;

  try {
    std::vector<double> permuted(h->n);

    // Apply permutation to RHS: permuted[p[i]] = rhs[i] (scatter)
    for (int64_t i = 0; i < h->n; ++i) {
      permuted[h->permutation[i]] = rhs[i];
    }

    // Solve in-place
    h->solver->solve(h->factor_data_f64.data(), permuted.data(), h->n, 1);

    // Apply inverse permutation to solution: solution[i] = permuted[p[i]] (gather)
    for (int64_t i = 0; i < h->n; ++i) {
      solution[i] = permuted[h->permutation[i]];
    }
  } catch (const std::exception&) {
    // Silently fail
  }
}

void baspacho_solve_f32_device(baspacho_handle_t h, void* rhs_device,
                                void* solution_device) {
  if (!h || !h->solver || !rhs_device || !solution_device) return;

  try {
    // For GPU backends, data is already on device and properly formatted.
    // The solver will use the appropriate buffer registry to find GPU buffers.
    float* rhs = static_cast<float*>(rhs_device);
    float* sol = static_cast<float*>(solution_device);

    // Copy RHS to solution if needed (in-place solve)
    if (rhs != sol) {
#ifdef BASPACHO_USE_METAL
      if (h->backend == BaSpaCho::BackendMetal) {
        // Metal: Use buffer registry to find MTLBuffers.
        // On unified memory systems, the CPU pointers work directly.
        // This is a placeholder for future Metal blit encoder support.
      }
#endif
#ifdef BASPACHO_USE_OPENCL
      if (h->backend == BaSpaCho::BackendOpenCL) {
        // OpenCL: Use buffer registry to find cl_mem objects
        auto& registry = BaSpaCho::OpenCLBufferRegistry::instance();
        auto [rhsBuf, rhsOff] = registry.findBuffer(rhs);
        auto [solBuf, solOff] = registry.findBuffer(sol);
        if (rhsBuf && solBuf) {
          auto& ctx = BaSpaCho::OpenCLContext::instance();
          clEnqueueCopyBuffer(ctx.queue(), rhsBuf, solBuf,
                              rhsOff, solOff, h->n * sizeof(float),
                              0, nullptr, nullptr);
        }
      }
#endif
    }

    // Solve in-place - solver uses buffer registry internally
    h->solver->solve(h->factor_data_f32.data(), sol, h->n, 1);
  } catch (const std::exception&) {
    // Silently fail
  }
}

void baspacho_solve_f64_device(baspacho_handle_t h, void* rhs_device,
                                void* solution_device) {
  if (!h || !h->solver || !rhs_device || !solution_device) return;

  try {
    double* rhs = static_cast<double*>(rhs_device);
    double* sol = static_cast<double*>(solution_device);

    if (rhs != sol) {
#ifdef BASPACHO_USE_OPENCL
      if (h->backend == BaSpaCho::BackendOpenCL) {
        auto& registry = BaSpaCho::OpenCLBufferRegistry::instance();
        auto [rhsBuf, rhsOff] = registry.findBuffer(rhs);
        auto [solBuf, solOff] = registry.findBuffer(sol);
        if (rhsBuf && solBuf) {
          auto& ctx = BaSpaCho::OpenCLContext::instance();
          clEnqueueCopyBuffer(ctx.queue(), rhsBuf, solBuf,
                              rhsOff, solOff, h->n * sizeof(double),
                              0, nullptr, nullptr);
        }
      }
#endif
      // Note: Metal backend doesn't support double precision
    }

    h->solver->solve(h->factor_data_f64.data(), sol, h->n, 1);
  } catch (const std::exception&) {
    // Silently fail
  }
}

//===----------------------------------------------------------------------===//
// LU Solve Operations
//===----------------------------------------------------------------------===//

void baspacho_solve_lu_f32(baspacho_handle_t h, const int64_t* pivots,
                            const float* rhs, float* solution) {
  if (!h || !h->solver || !pivots || !rhs || !solution) return;

  try {
    // Copy RHS to solution (solve is in-place)
    std::memcpy(solution, rhs, h->n * sizeof(float));

    // Apply permutation to solution vector
    std::vector<float> permuted(h->n);
    for (int64_t i = 0; i < h->n; ++i) {
      permuted[h->permutation[i]] = solution[i];
    }

    // Solve in-place using LU factors
    h->solver->solveLU(h->factor_data_f32.data(), pivots, permuted.data(), h->n, 1);

    // Apply inverse permutation
    for (int64_t i = 0; i < h->n; ++i) {
      solution[i] = permuted[h->permutation[i]];
    }
  } catch (const std::exception&) {
    // Silently fail
  }
}

void baspacho_solve_lu_f64(baspacho_handle_t h, const int64_t* pivots,
                            const double* rhs, double* solution) {
  if (!h || !h->solver || !pivots || !rhs || !solution) return;

  try {
    std::memcpy(solution, rhs, h->n * sizeof(double));

    std::vector<double> permuted(h->n);
    for (int64_t i = 0; i < h->n; ++i) {
      permuted[h->permutation[i]] = solution[i];
    }

    h->solver->solveLU(h->factor_data_f64.data(), pivots, permuted.data(), h->n, 1);

    for (int64_t i = 0; i < h->n; ++i) {
      solution[i] = permuted[h->permutation[i]];
    }
  } catch (const std::exception&) {
    // Silently fail
  }
}

void baspacho_solve_lu_f32_device(baspacho_handle_t h, const int64_t* pivots,
                                   void* rhs_device, void* solution_device) {
  if (!h || !h->solver || !pivots || !rhs_device || !solution_device) return;

  try {
    float* rhs = static_cast<float*>(rhs_device);
    float* sol = static_cast<float*>(solution_device);

    if (rhs != sol) {
      // Copy RHS to solution for in-place solve
      // For GPU backends, would need appropriate copy mechanism
    }

    h->solver->solveLU(h->factor_data_f32.data(), pivots, sol, h->n, 1);
  } catch (const std::exception&) {
    // Silently fail
  }
}

void baspacho_solve_lu_f64_device(baspacho_handle_t h, const int64_t* pivots,
                                   void* rhs_device, void* solution_device) {
  if (!h || !h->solver || !pivots || !rhs_device || !solution_device) return;

  try {
    double* rhs = static_cast<double*>(rhs_device);
    double* sol = static_cast<double*>(solution_device);

    if (rhs != sol) {
      // Copy RHS to solution for in-place solve
    }

    h->solver->solveLU(h->factor_data_f64.data(), pivots, sol, h->n, 1);
  } catch (const std::exception&) {
    // Silently fail
  }
}

//===----------------------------------------------------------------------===//
// Batched Solve Operations
//===----------------------------------------------------------------------===//

void baspacho_solve_batched_f32(baspacho_handle_t h, const float* rhs,
                                 float* solution, int64_t num_rhs) {
  if (!h || !h->solver || !rhs || !solution || num_rhs <= 0) return;

  try {
    int64_t n = h->n;
    float* factor_data = nullptr;

#ifdef BASPACHO_USE_METAL
    if (h->backend == BaSpaCho::BackendMetal && h->metal_factor_data_f32) {
      // Use Metal buffer for GPU solve
      factor_data = h->metal_factor_data_f32->ptr();

      // Resize permuted buffer if needed for batched solve
      size_t needed_size = n * num_rhs;
      if (!h->metal_permuted_f32 || h->metal_permuted_f32->allocSize() < needed_size) {
        h->metal_permuted_f32.reset(new BaSpaCho::MetalMirror<float>());
        h->metal_permuted_f32->resizeToAtLeast(needed_size);
      }
      float* permuted = h->metal_permuted_f32->ptr();

      // Apply permutation to each RHS vector: permuted[p[i]] = rhs[i] (scatter)
      for (int64_t k = 0; k < num_rhs; ++k) {
        for (int64_t i = 0; i < n; ++i) {
          permuted[h->permutation[i] + k * n] = rhs[i + k * n];
        }
      }

      // Batched solve in-place using GPU
      h->solver->solve(factor_data, permuted, n, static_cast<int>(num_rhs));

      // Metal operations are batched - synchronize to ensure GPU work is complete
      BaSpaCho::MetalContext::instance().synchronize();

      // Apply inverse permutation to all solution vectors: solution[i] = permuted[p[i]] (gather)
      for (int64_t k = 0; k < num_rhs; ++k) {
        for (int64_t i = 0; i < n; ++i) {
          solution[i + k * n] = permuted[h->permutation[i] + k * n];
        }
      }
    } else
#endif
    {
      // CPU backend path
      factor_data = h->factor_data_f32.data();
      std::vector<float> permuted(n * num_rhs);

      // Apply permutation to each RHS vector: permuted[p[i]] = rhs[i] (scatter)
      for (int64_t k = 0; k < num_rhs; ++k) {
        for (int64_t i = 0; i < n; ++i) {
          permuted[h->permutation[i] + k * n] = rhs[i + k * n];
        }
      }

      // Batched solve in-place
      h->solver->solve(factor_data, permuted.data(), n, static_cast<int>(num_rhs));

      // Apply inverse permutation to all solution vectors: solution[i] = permuted[p[i]] (gather)
      for (int64_t k = 0; k < num_rhs; ++k) {
        for (int64_t i = 0; i < n; ++i) {
          solution[i + k * n] = permuted[h->permutation[i] + k * n];
        }
      }
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "baspacho_solve_batched_f32 exception: %s\n", e.what());
  }
}

void baspacho_solve_batched_f64(baspacho_handle_t h, const double* rhs,
                                 double* solution, int64_t num_rhs) {
  if (!h || !h->solver || !rhs || !solution || num_rhs <= 0) return;

  try {
    int64_t n = h->n;
    std::vector<double> permuted(n * num_rhs);

    // Apply permutation to each RHS vector: permuted[p[i]] = rhs[i] (scatter)
    for (int64_t k = 0; k < num_rhs; ++k) {
      for (int64_t i = 0; i < n; ++i) {
        permuted[h->permutation[i] + k * n] = rhs[i + k * n];
      }
    }

    // Batched solve in-place
    h->solver->solve(h->factor_data_f64.data(), permuted.data(), n,
                     static_cast<int>(num_rhs));

    // Apply inverse permutation to all solution vectors: solution[i] = permuted[p[i]] (gather)
    for (int64_t k = 0; k < num_rhs; ++k) {
      for (int64_t i = 0; i < n; ++i) {
        solution[i + k * n] = permuted[h->permutation[i] + k * n];
      }
    }
  } catch (const std::exception&) {
    // Silently fail
  }
}

void baspacho_solve_batched_f32_device(baspacho_handle_t h, void* rhs_device,
                                        void* solution_device, int64_t num_rhs) {
  // GPU batched solve - data permutation happens on GPU
  if (!h || !h->solver || !rhs_device || !solution_device) return;
  // Implementation depends on GPU backend
}

void baspacho_solve_batched_f64_device(baspacho_handle_t h, void* rhs_device,
                                        void* solution_device, int64_t num_rhs) {
  if (!h || !h->solver || !rhs_device || !solution_device) return;
  // Implementation depends on GPU backend
}

//===----------------------------------------------------------------------===//
// Async Operations
//===----------------------------------------------------------------------===//

void baspacho_set_command_queue(baspacho_handle_t h, void* queue) {
  if (!h) return;
  h->command_queue = queue;
}

int baspacho_factor_f32_async(baspacho_handle_t h, void* device_ptr) {
  // Async factorization - encodes work to command buffer/stream
  // The caller must synchronize after calling this
  return baspacho_factor_f32_device(h, device_ptr);
}

int baspacho_factor_f64_async(baspacho_handle_t h, void* device_ptr) {
  return baspacho_factor_f64_device(h, device_ptr);
}

void baspacho_solve_f32_async(baspacho_handle_t h, void* rhs_device,
                               void* solution_device) {
  baspacho_solve_f32_device(h, rhs_device, solution_device);
}

void baspacho_solve_f64_async(baspacho_handle_t h, void* rhs_device,
                               void* solution_device) {
  baspacho_solve_f64_device(h, rhs_device, solution_device);
}

//===----------------------------------------------------------------------===//
// Buffer Registration
//===----------------------------------------------------------------------===//

void baspacho_register_metal_buffer(void* host_ptr, void* mtl_buffer,
                                     size_t size) {
#ifdef BASPACHO_USE_METAL
  if (!host_ptr || !mtl_buffer || size == 0) return;

  try {
    auto& registry = BaSpaCho::MetalBufferRegistry::instance();
    // mtl_buffer is passed as void* from C; cast to appropriate type
    // BaSpaCho's registerBuffer expects the buffer handle
    registry.registerBuffer(host_ptr, mtl_buffer, size);
  } catch (const std::exception&) {
    // Silently fail - buffer won't be found for zero-copy operations
  }
#else
  (void)host_ptr;
  (void)mtl_buffer;
  (void)size;
#endif
}

void baspacho_unregister_metal_buffer(void* host_ptr) {
#ifdef BASPACHO_USE_METAL
  if (!host_ptr) return;

  try {
    auto& registry = BaSpaCho::MetalBufferRegistry::instance();
    registry.unregisterBuffer(host_ptr);
  } catch (const std::exception&) {
    // Silently fail
  }
#else
  (void)host_ptr;
#endif
}
