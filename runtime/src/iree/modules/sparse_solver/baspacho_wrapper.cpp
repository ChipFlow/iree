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
  int64_t nnz;  // Number of non-zeros (in lower triangular part)
  int64_t original_nnz;  // Number of non-zeros (in original input)

  // Lower triangular CSR structure (for loadFromCsr)
  // BaSpaCho expects lower triangular input for Cholesky factorization
  std::vector<int64_t> csr_row_ptr;
  std::vector<int64_t> csr_col_idx;

  // Mapping from lower triangular positions to original CSR positions
  // Used to extract lower triangular values from full matrix input
  std::vector<int64_t> lower_to_original_idx;

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
  ctx->original_nnz = 0;
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
    h->original_nnz = nnz;

    // For scalar CSR, each block is size 1
    h->block_sizes.assign(n, 1);

    // BaSpaCho expects LOWER TRIANGULAR CSR for Cholesky factorization.
    // The input may be a full symmetric matrix, so we extract only the
    // lower triangular part (entries where col <= row).
    //
    // We also build a mapping from lower triangular positions to original
    // positions so we can extract the correct values during factorization.
    h->csr_row_ptr.resize(n + 1);
    h->csr_row_ptr[0] = 0;

    // First pass: count lower triangular entries per row
    for (int64_t row = 0; row < n; ++row) {
      int64_t count = 0;
      for (int64_t ptr = row_ptr[row]; ptr < row_ptr[row + 1]; ++ptr) {
        int64_t col = col_idx[ptr];
        if (col <= row) {  // Lower triangular (including diagonal)
          ++count;
        }
      }
      h->csr_row_ptr[row + 1] = h->csr_row_ptr[row] + count;
    }

    int64_t lower_nnz = h->csr_row_ptr[n];
    h->nnz = lower_nnz;
    h->csr_col_idx.resize(lower_nnz);
    h->lower_to_original_idx.resize(lower_nnz);

    // Debug: print extraction stats
    fprintf(stderr, "[BaSpaCho] Lower triangular extraction: n=%lld, original_nnz=%lld, lower_nnz=%lld\n",
            (long long)n, (long long)nnz, (long long)lower_nnz);

    // Second pass: extract lower triangular entries and build mapping
    int64_t lower_idx = 0;
    for (int64_t row = 0; row < n; ++row) {
      for (int64_t ptr = row_ptr[row]; ptr < row_ptr[row + 1]; ++ptr) {
        int64_t col = col_idx[ptr];
        if (col <= row) {  // Lower triangular (including diagonal)
          h->csr_col_idx[lower_idx] = col;
          h->lower_to_original_idx[lower_idx] = ptr;  // Map to original position
          ++lower_idx;
        }
      }
    }

    // Create sparse structure from lower triangular CSR
    BaSpaCho::SparseStructure ss;
    ss.ptrs = h->csr_row_ptr;
    ss.inds = h->csr_col_idx;

    // Create solver settings
    BaSpaCho::Settings settings;
    settings.backend = h->backend;  // Use configured backend (Metal on Apple Silicon)
    settings.numThreads = 8;  // Reasonable default
    settings.addFillPolicy = BaSpaCho::AddFillComplete;
    // Enable sparse elimination for both CPU and Metal backends
    settings.findSparseEliminationRanges = true;
    fprintf(stderr, "[BaSpaCho] Using backend: %d (0=Ref, 1=Fast, 2=CUDA, 3=Metal)\n",
            (int)settings.backend);

    // Create solver (performs symbolic analysis)
    h->solver = BaSpaCho::createSolver(settings, h->block_sizes, ss);

    if (!h->solver) {
      return -2;  // Symbolic analysis failed
    }

    // Check if sparse elimination ranges were created
    const auto& ranges = h->solver->sparseEliminationRanges();
    fprintf(stderr, "[BaSpaCho] Sparse elim ranges: %zu elements: [", ranges.size());
    for (size_t i = 0; i < std::min(ranges.size(), size_t(10)); ++i) {
      fprintf(stderr, "%lld%s", (long long)ranges[i], i + 1 < ranges.size() ? ", " : "");
    }
    if (ranges.size() > 10) fprintf(stderr, ", ...");
    fprintf(stderr, "]\n");

    // Store permutation
    h->permutation = h->solver->paramToSpan();
    h->inverse_permutation.resize(n);
    for (int64_t i = 0; i < n; ++i) {
      h->inverse_permutation[h->permutation[i]] = i;
    }

    // Note: Factor storage will be allocated during factor_f32/factor_f64 calls.
    // For Metal backend, MetalMirror is created from std::vector to match
    // BaSpaCho test pattern (ensures proper buffer initialization).
    // Pre-allocate permuted buffer for solve operations.
    int64_t data_size = h->solver->dataSize();

#ifdef BASPACHO_USE_METAL
    if (h->backend == BaSpaCho::BackendMetal) {
      // Allocate permuted buffer for solve operations (factor buffer created in factor_f32)
      h->metal_permuted_f32 = std::make_unique<BaSpaCho::MetalMirror<float>>();
      h->metal_permuted_f32->resizeToAtLeast(n);
    }
#endif
    // CPU factor storage (used for CPU backend or fallback)
    h->factor_data_f32.resize(data_size);
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
    int64_t data_size = h->solver->dataSize();

    // Extract lower triangular values from the input using the mapping.
    // The input 'values' array is in original (possibly full symmetric) CSR order,
    // but BaSpaCho expects lower triangular only.
    std::vector<float> lower_values(h->nnz);
    for (int64_t i = 0; i < h->nnz; ++i) {
      lower_values[i] = values[h->lower_to_original_idx[i]];
    }

    // Debug: print first few values
    fprintf(stderr, "[BaSpaCho] factor_f32: data_size=%lld, nnz=%lld\n",
            (long long)data_size, (long long)h->nnz);
    fprintf(stderr, "[BaSpaCho] First 5 lower_values: ");
    for (int i = 0; i < std::min(5, (int)h->nnz); ++i) {
      fprintf(stderr, "%.4f ", lower_values[i]);
    }
    fprintf(stderr, "\n");

    // Following BaSpaCho test pattern exactly:
    // 1. Load CSR values into a CPU std::vector first
    // 2. Then copy to MetalMirror (which triggers proper initialization)
    // This ensures correct data flow for Metal GPU operations.
    std::vector<float> factorData(data_size, 0.0f);

    // Load CSR values into BaSpaCho's internal format using lower triangular structure
    // This handles the permutation and format conversion automatically
    h->solver->loadFromCsr(h->csr_row_ptr.data(), h->csr_col_idx.data(),
                           h->block_sizes.data(), lower_values.data(), factorData.data());

    // Debug: print first few factor data values after loadFromCsr
    fprintf(stderr, "[BaSpaCho] First 10 factor data after loadFromCsr: ");
    for (int i = 0; i < std::min(10, (int)data_size); ++i) {
      fprintf(stderr, "%.4f ", factorData[i]);
    }
    fprintf(stderr, "\n");

    float* data = nullptr;

#ifdef BASPACHO_USE_METAL
    if (h->backend == BaSpaCho::BackendMetal) {
      // Copy factor data to MetalMirror using load() method
      // This matches the BaSpaCho test pattern: MetalMirror<T> dataGpu(factorData);
      h->metal_factor_data_f32 = std::make_unique<BaSpaCho::MetalMirror<float>>(factorData);
      data = h->metal_factor_data_f32->ptr();
      fprintf(stderr, "[BaSpaCho] Copied %lld floats to MetalMirror, ptr=%p\n",
              (long long)data_size, (void*)data);
    } else {
      h->factor_data_f32 = factorData;
      data = h->factor_data_f32.data();
    }
#else
    h->factor_data_f32 = factorData;
    data = h->factor_data_f32.data();
#endif

    // Perform numeric factorization (GPU or CPU depending on backend)
    h->solver->factor(data);

    // Debug: print first few factor data values after factor
    fprintf(stderr, "[BaSpaCho] First 10 factor data after factor: ");
#ifdef BASPACHO_USE_METAL
    if (h->backend == BaSpaCho::BackendMetal) {
      BaSpaCho::MetalContext::instance().synchronize();
    }
#endif
    for (int i = 0; i < std::min(10, (int)data_size); ++i) {
      fprintf(stderr, "%.4f ", data[i]);
    }
    fprintf(stderr, "\n");

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

    // Extract lower triangular values from the input using the mapping.
    std::vector<double> lower_values(h->nnz);
    for (int64_t i = 0; i < h->nnz; ++i) {
      lower_values[i] = values[h->lower_to_original_idx[i]];
    }

    // Load CSR values into BaSpaCho's internal format using lower triangular structure
    h->solver->loadFromCsr(h->csr_row_ptr.data(), h->csr_col_idx.data(),
                           h->block_sizes.data(), lower_values.data(), data);

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

    // Extract lower triangular values from the input using the mapping.
    std::vector<float> lower_values(h->nnz);
    for (int64_t i = 0; i < h->nnz; ++i) {
      lower_values[i] = values[h->lower_to_original_idx[i]];
    }

    // Load CSR values into BaSpaCho's internal format using lower triangular structure
    h->solver->loadFromCsr(h->csr_row_ptr.data(), h->csr_col_idx.data(),
                           h->block_sizes.data(), lower_values.data(), data);

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

    // Extract lower triangular values from the input using the mapping.
    std::vector<double> lower_values(h->nnz);
    for (int64_t i = 0; i < h->nnz; ++i) {
      lower_values[i] = values[h->lower_to_original_idx[i]];
    }

    // Load CSR values into BaSpaCho's internal format using lower triangular structure
    h->solver->loadFromCsr(h->csr_row_ptr.data(), h->csr_col_idx.data(),
                           h->block_sizes.data(), lower_values.data(), data);

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

    // Check environment variable to force CPU solve for debugging
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

      // Debug: print pointer addresses and factor data
      fprintf(stderr, "[BaSpaCho] solve: factor_data ptr=%p, permuted ptr=%p\n",
              (void*)factor_data, (void*)permuted);
      fprintf(stderr, "[BaSpaCho] solve: n=%lld, permutation.size()=%zu\n",
              (long long)h->n, h->permutation.size());
      fprintf(stderr, "[BaSpaCho] solve: First 10 permutation: ");
      for (int i = 0; i < std::min(10, (int)h->permutation.size()); ++i) {
        fprintf(stderr, "%lld ", (long long)h->permutation[i]);
      }
      fprintf(stderr, "\n");
      fprintf(stderr, "[BaSpaCho] solve: First 5 factor data in solve: ");
      for (int i = 0; i < 5; ++i) {
        fprintf(stderr, "%.4f ", factor_data[i]);
      }
      fprintf(stderr, "\n");

      // Debug: print first few RHS values
      fprintf(stderr, "[BaSpaCho] solve: First 5 RHS values: ");
      for (int i = 0; i < std::min(5, (int)h->n); ++i) {
        fprintf(stderr, "%.4f ", rhs[i]);
      }
      fprintf(stderr, "\n");

      // Apply permutation to RHS: permuted[p[i]] = rhs[i] (scatter)
      // BaSpaCho's solve does NOT apply permutation internally for Cholesky
      for (int64_t i = 0; i < h->n; ++i) {
        permuted[h->permutation[i]] = rhs[i];
      }

      // IMPORTANT: Synchronize to ensure CPU writes are visible to GPU
      // On unified memory, this flushes any CPU caches to ensure coherency
      BaSpaCho::MetalContext::instance().synchronize();

      // Debug: print first few permuted values
      fprintf(stderr, "[BaSpaCho] solve: First 5 permuted values: ");
      for (int i = 0; i < std::min(5, (int)h->n); ++i) {
        fprintf(stderr, "%.4f ", permuted[i]);
      }
      fprintf(stderr, "\n");

      // Solve in-place using GPU (Metal backend finds buffers in registry)
      fprintf(stderr, "[Wrapper] About to call h->solver->solve() with factor_data=%p, permuted=%p\n",
              (void*)factor_data, (void*)permuted);
      h->solver->solve(factor_data, permuted, h->n, 1);
      fprintf(stderr, "[Wrapper] h->solver->solve() returned\n");

      // Metal operations are batched - synchronize to ensure GPU work is complete
      BaSpaCho::MetalContext::instance().synchronize();

      // Debug: print solution before inverse permutation
      fprintf(stderr, "[BaSpaCho] solve: First 5 permuted after solve: ");
      for (int i = 0; i < std::min(5, (int)h->n); ++i) {
        fprintf(stderr, "%.4f ", permuted[i]);
      }
      fprintf(stderr, "\n");

      // Apply inverse permutation to solution: solution[i] = permuted[p[i]] (gather)
      for (int64_t i = 0; i < h->n; ++i) {
        solution[i] = permuted[h->permutation[i]];
      }

      // Debug: print final solution
      fprintf(stderr, "[BaSpaCho] solve: First 5 solution values: ");
      for (int i = 0; i < std::min(5, (int)h->n); ++i) {
        fprintf(stderr, "%.4f ", solution[i]);
      }
      fprintf(stderr, "\n");
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
  } catch (const std::exception& e) {
    // Log error - this is critical for debugging
    fprintf(stderr, "[BaSpaCho] solve_f32 EXCEPTION: %s\n", e.what());
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
