// Sparse Solver runtime module imports.
//
// This is embedded in the compiler binary and inserted into any module
// containing sparse solver operations that need to call into the BaSpaCho
// sparse direct solver backend.
//
// BaSpaCho supports multiple backends:
// - Metal (Apple GPUs)
// - CUDA (NVIDIA GPUs)
// - OpenCL (generic fallback)
// - CPU (reference implementation)
//
// Workflow for sparse LU solve:
// 1. analyze() - Symbolic analysis of sparsity pattern (done once per pattern)
// 2. factor.lu() - Numeric LU factorization (done when matrix values change)
// 3. solve.lu() - Forward/backward substitution (can be called multiple times)
// 4. release() - Free the context handle
//
// NOTE: each method added here requires a corresponding method in
// `iree/modules/sparse_solver/module.c`.
//
// The calling conventions use IREE VM ABI encoding:
//   r = ref (buffer_view or opaque handle)
//   I = i64
//   i = i32
//   v = void (no return)
//   _ separates inputs from outputs
//
vm.module @sparse_solver {

//===----------------------------------------------------------------------===//
// Context Management
//===----------------------------------------------------------------------===//

// Analyze sparsity pattern and create solver context.
// This performs symbolic analysis which is reused across multiple
// factorizations with the same sparsity pattern.
//
// Args:
//   device: HAL device reference (for backend selection)
//   n: Matrix dimension (n x n)
//   nnz: Number of non-zeros
//   row_ptr: CSR row pointers buffer (n+1 elements, int64)
//   col_idx: CSR column indices buffer (nnz elements, int64)
//
// Returns:
//   handle: Opaque solver context handle
vm.import private @analyze(
  %device : !vm.ref<!hal.device>,
  %n : i64,
  %nnz : i64,
  %row_ptr : !vm.ref<!hal.buffer_view>,
  %col_idx : !vm.ref<!hal.buffer_view>
) -> !vm.ref<?>

// Release solver context and free associated memory.
vm.import private @release(
  %handle : !vm.ref<?>
)

//===----------------------------------------------------------------------===//
// Cholesky Factorization (Symmetric Positive Definite)
//===----------------------------------------------------------------------===//

// Numeric Cholesky factorization (f32).
// For symmetric positive definite matrices.
vm.import private @factor(
  %handle : !vm.ref<?>,
  %values : !vm.ref<!hal.buffer_view>
) -> i32

// Numeric Cholesky factorization (f64).
vm.import private @factor.f64(
  %handle : !vm.ref<?>,
  %values : !vm.ref<!hal.buffer_view>
) -> i32

// Solve using Cholesky factors (f32).
// Solves Ax = b where A = L * L^T has been factored.
vm.import private @solve(
  %handle : !vm.ref<?>,
  %rhs : !vm.ref<!hal.buffer_view>,
  %solution : !vm.ref<!hal.buffer_view>
)

// Solve using Cholesky factors (f64).
vm.import private @solve.f64(
  %handle : !vm.ref<?>,
  %rhs : !vm.ref<!hal.buffer_view>,
  %solution : !vm.ref<!hal.buffer_view>
)

// Batched solve using Cholesky factors (f32).
vm.import private @solve.batched(
  %handle : !vm.ref<?>,
  %rhs : !vm.ref<!hal.buffer_view>,
  %solution : !vm.ref<!hal.buffer_view>,
  %batch_size : i64
)

// Batched solve using Cholesky factors (f64).
vm.import private @solve.batched.f64(
  %handle : !vm.ref<?>,
  %rhs : !vm.ref<!hal.buffer_view>,
  %solution : !vm.ref<!hal.buffer_view>,
  %batch_size : i64
)

//===----------------------------------------------------------------------===//
// LU Factorization (General Matrices)
//===----------------------------------------------------------------------===//

// Numeric LU factorization with partial pivoting (f32).
// For general (non-symmetric) matrices.
//
// Args:
//   handle: Solver context from analyze()
//   values: CSR values buffer (nnz elements, f32)
//   pivots: Output pivot indices buffer (n elements, int64)
//
// Returns:
//   result: 0 on success, negative on error
vm.import private @factor.lu(
  %handle : !vm.ref<?>,
  %values : !vm.ref<!hal.buffer_view>,
  %pivots : !vm.ref<!hal.buffer_view>
) -> i32

// Numeric LU factorization with partial pivoting (f64).
vm.import private @factor.lu.f64(
  %handle : !vm.ref<?>,
  %values : !vm.ref<!hal.buffer_view>,
  %pivots : !vm.ref<!hal.buffer_view>
) -> i32

// Solve using LU factors (f32).
// Solves Ax = b where A = P * L * U has been factored.
//
// Args:
//   handle: Solver context
//   pivots: Pivot indices from factor.lu()
//   rhs: Right-hand side vector (n elements, f32)
//   solution: Output solution vector (n elements, f32)
vm.import private @solve.lu(
  %handle : !vm.ref<?>,
  %pivots : !vm.ref<!hal.buffer_view>,
  %rhs : !vm.ref<!hal.buffer_view>,
  %solution : !vm.ref<!hal.buffer_view>
)

// Solve using LU factors (f64).
vm.import private @solve.lu.f64(
  %handle : !vm.ref<?>,
  %pivots : !vm.ref<!hal.buffer_view>,
  %rhs : !vm.ref<!hal.buffer_view>,
  %solution : !vm.ref<!hal.buffer_view>
)

//===----------------------------------------------------------------------===//
// Single-Shot Sparse Solve (Convenience Functions)
//===----------------------------------------------------------------------===//

// Complete sparse LU solve in a single call (f32).
// Combines analyze + factor.lu + solve.lu + release into one operation.
// This is convenient for one-shot solves where the matrix won't be reused.
//
// Args:
//   n: Matrix dimension (n x n)
//   nnz: Number of non-zeros
//   row_ptr: CSR row pointers buffer (n+1 elements, int64)
//   col_idx: CSR column indices buffer (nnz elements, int64)
//   values: CSR values buffer (nnz elements, f32)
//   rhs: Right-hand side vector (n elements, f32)
//   solution: Output solution vector (n elements, f32) - pre-allocated
vm.import private @spsolve_complete(
  %n : i64,
  %nnz : i64,
  %row_ptr : !vm.ref<!hal.buffer_view>,
  %col_idx : !vm.ref<!hal.buffer_view>,
  %values : !vm.ref<!hal.buffer_view>,
  %rhs : !vm.ref<!hal.buffer_view>,
  %solution : !vm.ref<!hal.buffer_view>
)

// Complete sparse LU solve in a single call (f64).
vm.import private @spsolve_complete.f64(
  %n : i64,
  %nnz : i64,
  %row_ptr : !vm.ref<!hal.buffer_view>,
  %col_idx : !vm.ref<!hal.buffer_view>,
  %values : !vm.ref<!hal.buffer_view>,
  %rhs : !vm.ref<!hal.buffer_view>,
  %solution : !vm.ref<!hal.buffer_view>
)

//===----------------------------------------------------------------------===//
// Diagnostic Functions
//===----------------------------------------------------------------------===//

// Get number of non-zeros in the factored matrix.
vm.import private @get_factor_nnz(
  %handle : !vm.ref<?>
) -> i64

// Get number of supernodes (for supernodal factorization).
vm.import private @get_num_supernodes(
  %handle : !vm.ref<?>
) -> i64

} // vm.module @sparse_solver
