################################################################################
# Sparse Solver Dialect
#
# Provides VM imports for the sparse_solver runtime module (BaSpaCho backend).
#
# Conversion patterns are in SparseSolverDialect.cpp using the interface pattern.
################################################################################

add_subdirectory(IR)
# Note: Conversion patterns are integrated via HALConversionDialectInterface
# and VMConversionDialectInterface in SparseSolverDialect.cpp

iree_c_embed_data(
  NAME
    sparse_solver_imports
  SRCS
    "sparse_solver.imports.mlir"
  C_FILE_OUTPUT
    "sparse_solver.imports.c"
  H_FILE_OUTPUT
    "sparse_solver.imports.h"
  IDENTIFIER
    "iree_sparse_solver_imports"
  FLATTEN
  PUBLIC
)
