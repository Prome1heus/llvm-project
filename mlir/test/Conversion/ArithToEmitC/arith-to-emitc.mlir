// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-arith-to-emitc))" %s -split-input-file | FileCheck %s

// CHECK-LABEL: @testGeneric
func.func @testGeneric(f32, f32, i32, i32, ui32, ui32) -> (f32, i32) {
^bb0(%arg0: f32, %arg1: f32, %arg2: i32, %arg3: i32, %arg4: ui32, %arg5: ui32):
// CHECK:  = emitc.generic "@0 - @1" %arg0, %arg1 : f32, f32 -> f32
  %0 = arith.subf %arg0, %arg1: f32
// CHECK: = emitc.generic "@0 - @1" %arg2, %arg3 : i32, i32 -> i32
  %1 = arith.subi %arg2, %arg3: i32
// CHECK: = emitc.generic "@0 & @1" %arg2, %arg3 : i32, i32 -> i32
  %2 = arith.andi %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 / @1 + ((@0 % @1)>0)" %arg2, %arg3 : i32, i32 -> i32
  %3 = arith.ceildivsi %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 / @1 + ((@0 % @1)>0)" %arg2, %arg3 : i32, i32 -> i32
  %4 = arith.ceildivui %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 / @1" %arg0, %arg1 : f32, f32 -> f32
  %5 = arith.divf %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 / @1" %arg2, %arg3 : i32, i32 -> i32
  %6 = arith.divsi %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 / @1" %arg2, %arg3 : i32, i32 -> i32
  %7 = arith.divui %arg2, %arg3 : i32

  func.return %arg0, %arg2: f32, i32
}


// CHECK-LABEL: @testCast
func.func @testCast(f32, f32, i32, i32, ui32, ui32) -> (f32, i32) {
^bb0(%arg0: f32, %arg1: f32, %arg2: i32, %arg3: i32, %arg4: ui32, %arg5: ui32):
// CHECK: = emitc.cast %arg0 : f32 to f64
  %0 = arith.extf %arg0 : f32 to f64
// CHECK: = emitc.cast %arg2 : i32 to i64
  %1 = arith.extsi %arg2 : i32 to i64
// CHECK: = emitc.cast %arg2 : i32 to i64
  %2 = arith.extui %arg2 : i32 to i64
// CHECK: = emitc.cast %arg0 : f32 to i32
  %3 = arith.fptosi %arg0 : f32 to i32
// CHECK: = emitc.cast %arg0 : f32 to i32
  %4 = arith.fptoui %arg0 : f32 to i32

  func.return %arg0, %arg2: f32, i32
}

// CHECK-LABEL: @testCompare
func.func @testCompare(f32, f32, i32, i32, ui32, ui32) -> (f32, i32) {
^bb0(%arg0: f32, %arg1: f32, %arg2: i32, %arg3: i32, %arg4: ui32, %arg5: ui32):
// CHECK: = emitc.generic "@0 == @1" %arg0, %arg1 : f32, f32 -> i1
  %0 = arith.cmpf oeq, %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 < @1" %arg0, %arg1 : f32, f32 -> i1
  %1 = arith.cmpf olt, %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 <= @1" %arg0, %arg1 : f32, f32 -> i1
  %2 = arith.cmpf ole, %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 > @1" %arg0, %arg1 : f32, f32 -> i1
  %3 = arith.cmpf ogt, %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 >= @1" %arg0, %arg1 : f32, f32 -> i1
  %4 = arith.cmpf oge, %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 != @1" %arg0, %arg1 : f32, f32 -> i1
  %5 = arith.cmpf one, %arg0, %arg1 : f32

// CHECK: = emitc.generic "@0 == @1" %arg2, %arg3 : i32, i32 -> i1
  %6 = arith.cmpi eq, %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 < @1" %arg2, %arg3 : i32, i32 -> i1
  %7 = arith.cmpi slt, %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 <= @1" %arg2, %arg3 : i32, i32 -> i1
  %8 = arith.cmpi ule, %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 > @1" %arg2, %arg3 : i32, i32 -> i1
  %9 = arith.cmpi ugt, %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 >= @1" %arg2, %arg3 : i32, i32 -> i1
  %10 = arith.cmpi sge, %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 != @1" %arg2, %arg3 : i32, i32 -> i1
  %11 = arith.cmpi ne, %arg2, %arg3 : i32

  func.return %arg0, %arg2: f32, i32
}