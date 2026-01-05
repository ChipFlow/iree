################################################################################
# SparseSolver IR - Dialect and operations
################################################################################

iree_tablegen_library(
  NAME
    SparseSolverDialectGen
  TD_FILE
    "SparseSolverDialect.td"
  OUTS
    -gen-dialect-decls SparseSolverDialect.h.inc
    -gen-dialect-defs SparseSolverDialect.cpp.inc
)

iree_tablegen_library(
  NAME
    SparseSolverOpsGen
  TD_FILE
    "SparseSolverOps.td"
  OUTS
    -gen-op-decls SparseSolverOps.h.inc
    -gen-op-defs SparseSolverOps.cpp.inc
)

iree_cc_library(
  NAME
    IR
  HDRS
    "SparseSolverDialect.h"
    "SparseSolverOps.h"
  SRCS
    "SparseSolverDialect.cpp"
  DEPS
    ::SparseSolverDialectGen
    ::SparseSolverOpsGen
    iree::compiler::Dialect::HAL::Conversion
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::SparseSolver::sparse_solver_imports
    iree::compiler::Dialect::VM::Conversion
    iree::compiler::Dialect::VM::IR
    MLIRFuncDialect
    MLIRIR
    MLIRParser
    MLIRSupport
  PUBLIC
)
