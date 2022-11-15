//===- ArithToEmitC.cpp - Arithmetic to EmitC dialect conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/ArithToEmitC/ConversionTarget.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
#define GEN_PASS_DEF_ARITHTOEMITCCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//
namespace {
struct ArithToEmitCConversionPass
    : public impl::ArithToEmitCConversionPassBase<ArithToEmitCConversionPass> {
  using Base::Base;

  void runOnOperation() override {
    EmitCConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    mlir::arith::populateArithToEmitCConversionPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Straightforward Op Lowerings
//===----------------------------------------------------------------------===//
LogicalResult AddIOpLowering(arith::AddIOp op, PatternRewriter &rewriter){
  rewriter.replaceOpWithNewOp<emitc::GenericOp>(
      op, op->getResult(0).getType(), "@0 + @1", op->getOperands()
      );
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//
void mlir::arith::populateArithToEmitCConversionPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add(AddIOpLowering);
}
