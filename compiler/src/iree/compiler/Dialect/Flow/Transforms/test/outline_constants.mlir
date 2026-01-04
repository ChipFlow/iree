// RUN: iree-opt --split-input-file --iree-flow-outline-constants %s | FileCheck %s

// Tests that we don't outline splats (as we want them to be transients).

// CHECK-LABEL: @splatConstant
util.func @splatConstant() {
  // CHECK-DAG: = arith.constant dense<1> : tensor<512x128xi32>
  %arith_cst = arith.constant dense<1> : tensor<512x128xi32>
  // CHECK-DAG: = flow.tensor.constant dense<1> : tensor<512x128xi32>
  %flow_cst = flow.tensor.constant dense<1> : tensor<512x128xi32>
  util.return
}

// -----

// Tests that constant parameters are outlined.

// CHECK: util.global private @__parameter_scope_key_tensor_4x2xi32 {inlining_policy = #util.inline.never} = #flow.parameter.named<"scope"::"key"> : tensor<4x2xi32>
// CHECK-LABEL: @parameterConstant
util.func @parameterConstant() {
  // CHECK: = util.global.load immutable @__parameter_scope_key_tensor_4x2xi32 : tensor<4x2xi32>
  %cst = flow.tensor.constant #flow.parameter.named<"scope"::"key"> : tensor<4x2xi32>
  util.return
}

// -----

// Tests that multiple constants will be hoisted and named uniquely.

//      CHECK: util.global private @__constant_tensor_2xf32 {inlining_policy = #util.inline.never} = dense<[0.0287729427, 0.0297581609]> : tensor<2xf32>
// CHECK-NEXT: util.global private @__constant_tensor_2xf32_0 {inlining_policy = #util.inline.never} = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>
// CHECK-NEXT: util.func private @denseConstants
util.func private @denseConstants() {
  // CHECK-NEXT: = util.global.load immutable @__constant_tensor_2xf32 : tensor<2xf32>
  %cst_0 = arith.constant dense<[0.0287729427, 0.0297581609]> : tensor<2xf32>
  // CHECK-NEXT: = util.global.load immutable @__constant_tensor_2xf32_0 : tensor<2xf32>
  %cst_1 = flow.tensor.constant dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>
  util.return
}

// -----

// Tests that constants are outlined to the module scope above their use to
// preserve ordering of existing functions/globals.

// CHECK: util.func private @external_func
util.func private @external_func()
// CHECK-NEXT: util.global private @__constant_tensor_2xi32
// CHECK-NEXT: util.func private @func_0()
util.func private @func_0() {
  // CHECK-NEXT: = util.global.load immutable @__constant_tensor_2xi32
  %cst_0 = arith.constant dense<[0, 1]> : tensor<2xi32>
  util.return
}

// CHECK: util.global private @existing_global
util.global private @existing_global : tensor<4xf32>
// CHECK-NEXT: util.global private @__constant_tensor_3xi32
// CHECK-NEXT: util.func private @func_1()
util.func private @func_1() {
  // CHECK-NEXT: = util.global.load immutable @__constant_tensor_3xi32
  %cst_1 = arith.constant dense<[2, 3, 4]> : tensor<3xi32>
  util.return
}

// -----

// Tests that any hoistable attrs are propagated to the outlined globals.

util.global private @device : !hal.device

//      CHECK: util.global private @__constant_tensor_2xi32
// CHECK-SAME:   stream.affinity = #hal.device.affinity<@device, [0]>
// CHECK-NEXT: util.func private @set_affinity
util.func private @set_affinity() attributes {
  stream.affinity = #hal.device.affinity<@device, [0]>
} {
  // CHECK-NEXT: = util.global.load immutable @__constant_tensor_2xi32
  %cst = arith.constant dense<[0, 1]> : tensor<2xi32>
  util.return
}

// -----

// Tests that constants inside scf.while loop bodies are NOT hoisted.
// These constants cannot be replaced with global.load because the loop body
// region cannot access module-level globals.

// CHECK-LABEL: @constantInsideWhileLoop
// CHECK-NOT: util.global
util.func @constantInsideWhileLoop(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %c0 = arith.constant 0 : index
  %init = arith.constant dense<0.0> : tensor<2xf32>
  // CHECK: scf.while
  %result:2 = scf.while (%iter = %init, %acc = %arg0) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
    // CHECK: arith.constant dense<[1.000000e+00, 2.000000e+00]>
    %threshold = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
    %cmp = arith.cmpf olt, %iter, %threshold : tensor<2xf32>
    %any_less = tensor.extract %cmp[%c0] : tensor<2xi1>
    scf.condition(%any_less) %iter, %acc : tensor<2xf32>, tensor<2xf32>
  } do {
  ^bb0(%iter: tensor<2xf32>, %acc: tensor<2xf32>):
    // CHECK: arith.constant dense<[0.100000{{.*}}, 0.200000{{.*}}]>
    %step = arith.constant dense<[0.1, 0.2]> : tensor<2xf32>
    %next = arith.addf %iter, %step : tensor<2xf32>
    scf.yield %next, %acc : tensor<2xf32>, tensor<2xf32>
  }
  util.return %result#1 : tensor<2xf32>
}

// -----

// Tests that constants inside scf.for loop bodies are NOT hoisted.

// CHECK-LABEL: @constantInsideForLoop
// CHECK-NOT: util.global
util.func @constantInsideForLoop(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  // CHECK: scf.for
  %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %arg0) -> tensor<2xf32> {
    // CHECK: arith.constant dense<[0.100000{{.*}}, 0.200000{{.*}}]>
    %step = arith.constant dense<[0.1, 0.2]> : tensor<2xf32>
    %next = arith.addf %acc, %step : tensor<2xf32>
    scf.yield %next : tensor<2xf32>
  }
  util.return %result : tensor<2xf32>
}
