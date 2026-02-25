// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-stream-fuse-loop-iteration-execution))" %s | FileCheck %s

// Tests basic scan loop fusion: a simple loop with one dispatch per iteration,
// constant bounds, and a single resource carry. Should be fused into a single
// stream.async.execute with the dispatch unrolled N times.

stream.executable private @scan_body {
  stream.executable.export public @entry
  builtin.module {
    func.func @entry(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) {
      return
    }
  }
}

// CHECK-LABEL: @simpleScanFusion
util.func public @simpleScanFusion(%init: !stream.resource<*>, %xs: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index

  // The scf.for should be replaced by a single stream.async.execute.
  // CHECK-NOT: scf.for
  // CHECK: %[[FUSED_RESULT:.+]], %[[FUSED_TP:.+]] = stream.async.execute
  // CHECK-SAME: with(%{{.+}} as %[[CARRY:.+]]: !stream.resource<*>{%c4},
  // CHECK-SAME:       %{{.+}} as %[[BUF:.+]]: !stream.resource<*>{%c16})
  // CHECK-SAME: -> !stream.resource<*>{%c4}

  // Iteration 0
  // CHECK: stream.async.dispatch @scan_body::@entry[%c1](%[[CARRY]]
  // CHECK-SAME: %[[BUF]]

  // Iteration 1 - uses previous dispatch result as carry
  // CHECK: stream.async.dispatch @scan_body::@entry[%c1](
  // CHECK-SAME: %[[BUF]]

  // Iteration 2
  // CHECK: stream.async.dispatch @scan_body::@entry[%c1](
  // CHECK-SAME: %[[BUF]]

  // Iteration 3
  // CHECK: stream.async.dispatch @scan_body::@entry[%c1](
  // CHECK-SAME: %[[BUF]]

  // Single yield and single await
  // CHECK: stream.yield
  // CHECK: stream.timepoint.await %[[FUSED_TP]]

  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%carry = %init) -> !stream.resource<*> {
    %exec_result, %tp = stream.async.execute
        with(%carry as %arg0: !stream.resource<*>{%c4},
             %xs as %arg1: !stream.resource<*>{%c16})
        -> !stream.resource<*>{%c4} {
      %dispatch = stream.async.dispatch @scan_body::@entry[%c1](
          %arg0[%c0 to %c4 for %c4],
          %arg1[%c0 to %c16 for %c16]) : (!stream.resource<*>{%c4}, !stream.resource<*>{%c16}) -> !stream.resource<*>{%c4}
      stream.yield %dispatch : !stream.resource<*>{%c4}
    } => !stream.timepoint
    %awaited = stream.timepoint.await %tp => %exec_result : !stream.resource<*>{%c4}
    scf.yield %awaited : !stream.resource<*>
  }

  util.return %result : !stream.resource<*>
}

// -----

// Tests that loops with non-constant bounds are NOT fused.

stream.executable private @dispatch {
  stream.executable.export public @entry
  builtin.module {
    func.func @entry(%arg0: !stream.binding, %arg1: !stream.binding) {
      return
    }
  }
}

// CHECK-LABEL: @nonConstantBoundsNotFused
util.func public @nonConstantBoundsNotFused(%init: !stream.resource<*>, %ub: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  // CHECK: scf.for
  %result = scf.for %i = %c0 to %ub step %c1 iter_args(%carry = %init) -> !stream.resource<*> {
    %exec_result, %tp = stream.async.execute
        with(%carry as %arg0: !stream.resource<*>{%c4})
        -> !stream.resource<*>{%c4} {
      %d = stream.async.dispatch @dispatch::@entry[%c1](
          %arg0[%c0 to %c4 for %c4]) : (!stream.resource<*>{%c4}) -> !stream.resource<*>{%c4}
      stream.yield %d : !stream.resource<*>{%c4}
    } => !stream.timepoint
    %awaited = stream.timepoint.await %tp => %exec_result : !stream.resource<*>{%c4}
    scf.yield %awaited : !stream.resource<*>
  }

  util.return %result : !stream.resource<*>
}

// -----

// Tests that single-iteration loops are NOT fused (nothing to fuse).

// CHECK-LABEL: @singleIterationNotFused
util.func public @singleIterationNotFused(%init: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  // CHECK: scf.for
  %result = scf.for %i = %c0 to %c1 step %c1 iter_args(%carry = %init) -> !stream.resource<*> {
    %exec_result, %tp = stream.async.execute
        with(%carry as %arg0: !stream.resource<*>{%c4})
        -> !stream.resource<*>{%c4} {
      %d = stream.async.dispatch @dispatch::@entry[%c1](
          %arg0[%c0 to %c4 for %c4]) : (!stream.resource<*>{%c4}) -> !stream.resource<*>{%c4}
      stream.yield %d : !stream.resource<*>{%c4}
    } => !stream.timepoint
    %awaited = stream.timepoint.await %tp => %exec_result : !stream.resource<*>{%c4}
    scf.yield %awaited : !stream.resource<*>
  }

  util.return %result : !stream.resource<*>
}

// -----

// Tests fusion with iteration-dependent values (offsets computed from
// induction variable). The arith ops should be cloned for each iteration.

// CHECK-LABEL: @iterationDependentOffsets
util.func public @iterationDependentOffsets(%init: !stream.resource<*>, %xs: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index

  // CHECK-NOT: scf.for
  // CHECK: stream.async.execute
  // Three dispatches (trip count = 3)
  // CHECK: stream.async.dispatch @scan_body::@entry
  // CHECK: stream.async.dispatch @scan_body::@entry
  // CHECK: stream.async.dispatch @scan_body::@entry
  // CHECK: stream.yield
  // CHECK: stream.timepoint.await

  %result = scf.for %i = %c0 to %c3 step %c1 iter_args(%carry = %init) -> !stream.resource<*> {
    %offset = arith.muli %i, %c4 : index
    %end = arith.addi %offset, %c4 : index
    %exec_result, %tp = stream.async.execute
        with(%carry as %arg0: !stream.resource<*>{%c4},
             %xs as %arg1: !stream.resource<*>{%c12})
        -> !stream.resource<*>{%c4} {
      %slice = stream.async.slice %arg1[%offset to %end] : !stream.resource<*>{%c12} -> !stream.resource<*>{%c4}
      %dispatch = stream.async.dispatch @scan_body::@entry[%c1](
          %arg0[%c0 to %c4 for %c4],
          %slice[%c0 to %c4 for %c4]) : (!stream.resource<*>{%c4}, !stream.resource<*>{%c4}) -> !stream.resource<*>{%c4}
      stream.yield %dispatch : !stream.resource<*>{%c4}
    } => !stream.timepoint
    %awaited = stream.timepoint.await %tp => %exec_result : !stream.resource<*>{%c4}
    scf.yield %awaited : !stream.resource<*>
  }

  util.return %result : !stream.resource<*>
}

// -----

// Tests that loops with non-resource iter_args are NOT fused.

// CHECK-LABEL: @nonResourceIterArgNotFused
util.func public @nonResourceIterArgNotFused(%init: !stream.resource<*>) -> (i64, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64

  // CHECK: scf.for
  %count, %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%cnt = %c0_i64, %carry = %init) -> (i64, !stream.resource<*>) {
    %exec_result, %tp = stream.async.execute
        with(%carry as %arg0: !stream.resource<*>{%c4})
        -> !stream.resource<*>{%c4} {
      %d = stream.async.dispatch @dispatch::@entry[%c1](
          %arg0[%c0 to %c4 for %c4]) : (!stream.resource<*>{%c4}) -> !stream.resource<*>{%c4}
      stream.yield %d : !stream.resource<*>{%c4}
    } => !stream.timepoint
    %awaited = stream.timepoint.await %tp => %exec_result : !stream.resource<*>{%c4}
    %new_cnt = arith.addi %cnt, %c1_i64 : i64
    scf.yield %new_cnt, %awaited : i64, !stream.resource<*>
  }

  util.return %count, %result : i64, !stream.resource<*>
}

// -----

// Tests fusion with external await timepoint (execute awaits a pre-loop
// timepoint). The fused execute should also await this timepoint.

// CHECK-LABEL: @fusionWithExternalTimepoint
util.func public @fusionWithExternalTimepoint(%init: !stream.resource<*>, %xs: !stream.resource<*>, %pre_tp: !stream.timepoint) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // CHECK-NOT: scf.for
  // CHECK: stream.async.execute await({{.+}}) => with(
  // Two dispatches
  // CHECK: stream.async.dispatch @scan_body::@entry
  // CHECK: stream.async.dispatch @scan_body::@entry
  // CHECK: stream.yield
  // CHECK: stream.timepoint.await

  %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%carry = %init) -> !stream.resource<*> {
    %exec_result, %tp = stream.async.execute await(%pre_tp) =>
        with(%carry as %arg0: !stream.resource<*>{%c4},
             %xs as %arg1: !stream.resource<*>{%c8})
        -> !stream.resource<*>{%c4} {
      %dispatch = stream.async.dispatch @scan_body::@entry[%c1](
          %arg0[%c0 to %c4 for %c4],
          %arg1[%c0 to %c8 for %c8]) : (!stream.resource<*>{%c4}, !stream.resource<*>{%c8}) -> !stream.resource<*>{%c4}
      stream.yield %dispatch : !stream.resource<*>{%c4}
    } => !stream.timepoint
    %awaited = stream.timepoint.await %tp => %exec_result : !stream.resource<*>{%c4}
    scf.yield %awaited : !stream.resource<*>
  }

  util.return %result : !stream.resource<*>
}

// -----

// Tests that loops with trip count exceeding the max unroll threshold (128)
// are NOT fused.

// CHECK-LABEL: @largeTripCountNotFused
util.func public @largeTripCountNotFused(%init: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c200 = arith.constant 200 : index

  // CHECK: scf.for
  %result = scf.for %i = %c0 to %c200 step %c1 iter_args(%carry = %init) -> !stream.resource<*> {
    %exec_result, %tp = stream.async.execute
        with(%carry as %arg0: !stream.resource<*>{%c4})
        -> !stream.resource<*>{%c4} {
      %d = stream.async.dispatch @dispatch::@entry[%c1](
          %arg0[%c0 to %c4 for %c4]) : (!stream.resource<*>{%c4}) -> !stream.resource<*>{%c4}
      stream.yield %d : !stream.resource<*>{%c4}
    } => !stream.timepoint
    %awaited = stream.timepoint.await %tp => %exec_result : !stream.resource<*>{%c4}
    scf.yield %awaited : !stream.resource<*>
  }

  util.return %result : !stream.resource<*>
}

// -----

// Tests fusion with non-unit step (step=2). The loop should be fused
// with trip count = ceil((6-0)/2) = 3.

stream.executable private @step_kernel {
  stream.executable.export public @entry
  builtin.module {
    func.func @entry(%arg0: !stream.binding, %arg1: !stream.binding) {
      return
    }
  }
}

// CHECK-LABEL: @nonUnitStepFusion
util.func public @nonUnitStepFusion(%init: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index

  // CHECK-NOT: scf.for
  // CHECK: stream.async.execute
  // Three dispatches (trip count = ceil((6-0)/2) = 3)
  // CHECK: stream.async.dispatch @step_kernel::@entry
  // CHECK: stream.async.dispatch @step_kernel::@entry
  // CHECK: stream.async.dispatch @step_kernel::@entry
  // CHECK-NOT: stream.async.dispatch
  // CHECK: stream.yield
  // CHECK: stream.timepoint.await

  %result = scf.for %i = %c0 to %c6 step %c2 iter_args(%carry = %init) -> !stream.resource<*> {
    %exec_result, %tp = stream.async.execute
        with(%carry as %arg0: !stream.resource<*>{%c4})
        -> !stream.resource<*>{%c4} {
      %d = stream.async.dispatch @step_kernel::@entry[%c1](
          %arg0[%c0 to %c4 for %c4]) : (!stream.resource<*>{%c4}) -> !stream.resource<*>{%c4}
      stream.yield %d : !stream.resource<*>{%c4}
    } => !stream.timepoint
    %awaited = stream.timepoint.await %tp => %exec_result : !stream.resource<*>{%c4}
    scf.yield %awaited : !stream.resource<*>
  }

  util.return %result : !stream.resource<*>
}

// -----

// Tests fusion with multiple non-carry resource captures. The fused execute
// should capture all resources (both carries and read-only buffers).

stream.executable private @multi_capture_kernel {
  stream.executable.export public @entry
  builtin.module {
    func.func @entry(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding) {
      return
    }
  }
}

// CHECK-LABEL: @multipleResourceCaptures
util.func public @multipleResourceCaptures(%init: !stream.resource<*>, %buf_a: !stream.resource<*>, %buf_b: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index

  // CHECK-NOT: scf.for
  // CHECK: stream.async.execute
  // CHECK-SAME: with(
  // Three dispatches
  // CHECK: stream.async.dispatch @multi_capture_kernel::@entry
  // CHECK: stream.async.dispatch @multi_capture_kernel::@entry
  // CHECK: stream.async.dispatch @multi_capture_kernel::@entry
  // CHECK: stream.yield
  // CHECK: stream.timepoint.await

  %result = scf.for %i = %c0 to %c3 step %c1 iter_args(%carry = %init) -> !stream.resource<*> {
    %exec_result, %tp = stream.async.execute
        with(%carry as %arg0: !stream.resource<*>{%c4},
             %buf_a as %arg1: !stream.resource<*>{%c8},
             %buf_b as %arg2: !stream.resource<*>{%c16})
        -> !stream.resource<*>{%c4} {
      %d = stream.async.dispatch @multi_capture_kernel::@entry[%c1](
          %arg0[%c0 to %c4 for %c4],
          %arg1[%c0 to %c8 for %c8],
          %arg2[%c0 to %c16 for %c16]) : (!stream.resource<*>{%c4}, !stream.resource<*>{%c8}, !stream.resource<*>{%c16}) -> !stream.resource<*>{%c4}
      stream.yield %d : !stream.resource<*>{%c4}
    } => !stream.timepoint
    %awaited = stream.timepoint.await %tp => %exec_result : !stream.resource<*>{%c4}
    scf.yield %awaited : !stream.resource<*>
  }

  util.return %result : !stream.resource<*>
}

// -----

// Tests fusion with multiple resource iter_args (two carries).
// Both carries should be threaded through all unrolled iterations.

stream.executable private @dual_carry_kernel {
  stream.executable.export public @entry
  builtin.module {
    func.func @entry(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding) {
      return
    }
  }
}

// CHECK-LABEL: @multipleCarries
util.func public @multipleCarries(%init_a: !stream.resource<*>, %init_b: !stream.resource<*>) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // CHECK-NOT: scf.for
  // CHECK: stream.async.execute
  // Two dispatches (trip count = 2), each producing two results
  // CHECK: stream.async.dispatch @dual_carry_kernel::@entry
  // CHECK: stream.async.dispatch @dual_carry_kernel::@entry
  // CHECK: stream.yield
  // CHECK: stream.timepoint.await

  %result_a, %result_b = scf.for %i = %c0 to %c2 step %c1
      iter_args(%carry_a = %init_a, %carry_b = %init_b) -> (!stream.resource<*>, !stream.resource<*>) {
    %exec_a, %exec_b, %tp = stream.async.execute
        with(%carry_a as %arg0: !stream.resource<*>{%c4},
             %carry_b as %arg1: !stream.resource<*>{%c8})
        -> (!stream.resource<*>{%c4}, !stream.resource<*>{%c8}) {
      %d_a, %d_b = stream.async.dispatch @dual_carry_kernel::@entry[%c1](
          %arg0[%c0 to %c4 for %c4],
          %arg1[%c0 to %c8 for %c8]) : (!stream.resource<*>{%c4}, !stream.resource<*>{%c8}) -> (!stream.resource<*>{%c4}, !stream.resource<*>{%c8})
      stream.yield %d_a, %d_b : !stream.resource<*>{%c4}, !stream.resource<*>{%c8}
    } => !stream.timepoint
    %awaited_a, %awaited_b = stream.timepoint.await %tp => %exec_a, %exec_b : !stream.resource<*>{%c4}, !stream.resource<*>{%c8}
    scf.yield %awaited_a, %awaited_b : !stream.resource<*>, !stream.resource<*>
  }

  util.return %result_a, %result_b : !stream.resource<*>, !stream.resource<*>
}
