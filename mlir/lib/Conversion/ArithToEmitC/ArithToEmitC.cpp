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

namespace {
  enum COperation {
    ADD, SUBTRACT, MULTIPLY, DIVIDE, AND, OR, MAX, MIN, NEGATE, SHIFT_LEFT,
    SHIFT_RIGHT, XOR, TERNARY_OP, COMPARE_EQUALS
  };

  std::unordered_map<COperation, std::string> opToFormatString{
      {COperation::ADD, "@0 + @1"},
      {COperation::SUBTRACT, "@0 - @1"},
      {COperation::MULTIPLY, "@0 * @1"},
      {COperation::DIVIDE, "@0 / @1"},
      {COperation::AND, "@0 & @1"},
      {COperation::OR, "@0 | @1"},
      {COperation::MAX, "std::max(@0, @1)"},
      {COperation::MIN, "std::min(@0, @1)"},
      {COperation::NEGATE, "-@0"},
      {COperation::SHIFT_LEFT, "@0 << @1"},
      {COperation::SHIFT_RIGHT, "@0 >> @1"},
      {COperation::XOR, "@0 ^ @1"},
      {COperation::TERNARY_OP, "@0 ? @1 : @2"},
      {COperation::COMPARE_EQUALS, "@0 == @1"}
  };

  template <typename ArithmeticOp, COperation cOp>
  LogicalResult GenericOpLowering(ArithmeticOp op, PatternRewriter &rewriter){
    rewriter.replaceOpWithNewOp<emitc::GenericOp>(
        op, op->getResult(0).getType(), opToFormatString[cOp], op->getOperands());
    return success();
  }

  template <typename CastOp>
  LogicalResult CastOpLowering(CastOp op, PatternRewriter &rewriter){
    rewriter.replaceOpWithNewOp<emitc::CastOp>(
        op, op->getResult(0).getType(), op->getOperands());
    return success();
  }
}



//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//
void mlir::arith::populateArithToEmitCConversionPatterns(mlir::RewritePatternSet &patterns) {
    patterns.add(GenericOpLowering<arith::AddFOp, COperation::ADD>);
    patterns.add(GenericOpLowering<arith::AddIOp, COperation::ADD>);
    // TODO: arith.addui_carry (::mlir::arith::AddUICarryOp)
    patterns.add(GenericOpLowering<arith::AndIOp, COperation::AND>);
    // TODO: arith.bitcast (::mlir::arith::BitcastOp)
    // TODO: arith.ceildivsi (::mlir::arith::CeilDivSIOp)
    // TODO: arith.ceildivui (::mlir::arith::CeilDivUIOp)
    patterns.add(GenericOpLowering<arith::CmpFOp, COperation::COMPARE_EQUALS>);
    patterns.add(GenericOpLowering<arith::CmpIOp, COperation::COMPARE_EQUALS>);
    // TODO: arith.constant (::mlir::arith::ConstantOp)
    patterns.add(GenericOpLowering<arith::DivFOp, COperation::DIVIDE>);
    // TODO: Attention treates leading bit as sign bit
    patterns.add(GenericOpLowering<arith::DivSIOp, COperation::DIVIDE>);
    patterns.add(GenericOpLowering<arith::DivUIOp, COperation::DIVIDE>);
    patterns.add(CastOpLowering<arith::ExtFOp>);
    patterns.add(CastOpLowering<arith::ExtSIOp>);
    patterns.add(CastOpLowering<arith::ExtUIOp>);
    patterns.add(CastOpLowering<arith::FPToSIOp>);
    patterns.add(CastOpLowering<arith::FPToUIOp>);
    // TODO: arith.floordivsi (::mlir::arith::FloorDivSIOp)
    patterns.add(CastOpLowering<arith::IndexCastOp>);
    patterns.add(CastOpLowering<arith::IndexCastUIOp>);
    patterns.add(GenericOpLowering<arith::MaxFOp, COperation::MAX>);
    patterns.add(GenericOpLowering<arith::MaxSIOp, COperation::MAX>);
    patterns.add(GenericOpLowering<arith::MaxUIOp, COperation::MAX>);
    patterns.add(GenericOpLowering<arith::MaxFOp, COperation::MIN>);
    patterns.add(GenericOpLowering<arith::MaxSIOp, COperation::MIN>);
    patterns.add(GenericOpLowering<arith::MaxUIOp, COperation::MIN>);
    patterns.add(GenericOpLowering<arith::MulFOp, COperation::MULTIPLY>);
    patterns.add(GenericOpLowering<arith::MulIOp, COperation::MULTIPLY>);
    patterns.add(GenericOpLowering<arith::NegFOp, COperation::NEGATE>);
    patterns.add(GenericOpLowering<arith::OrIOp, COperation::NEGATE>);
    // TODO: arith.remf (::mlir::arith::RemFOp)
    // TODO: arith.remsi (::mlir::arith::RemSIOp)
    // TODO: arith.remui (::mlir::arith::RemUIOp)
    // TODO: arith.sitofp (::mlir::arith::SIToFPOp)
    patterns.add(GenericOpLowering<arith::ShLIOp, COperation::SHIFT_LEFT>);
    // TODO check difference for negative numbers
    patterns.add(GenericOpLowering<arith::ShRSIOp, COperation::SHIFT_RIGHT>);
    patterns.add(GenericOpLowering<arith::ShRUIOp, COperation::SHIFT_RIGHT>);
    patterns.add(GenericOpLowering<arith::SubFOp, COperation::SUBTRACT>);
    patterns.add(GenericOpLowering<arith::SubIOp, COperation::SUBTRACT>);
    patterns.add(CastOpLowering<arith::TruncFOp>);
    // TODO: Probably wrong, top bits are discarded
    patterns.add(CastOpLowering<arith::TruncIOp>);
    patterns.add(CastOpLowering<arith::UIToFPOp>);
    patterns.add(GenericOpLowering<arith::XOrIOp, COperation::XOR>);
    patterns.add(GenericOpLowering<arith::SelectOp, COperation::TERNARY_OP>);
}
