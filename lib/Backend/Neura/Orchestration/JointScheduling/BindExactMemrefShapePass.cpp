//===- BindExactMemrefShapePass.cpp - Caller memref shape contract -------===//
//
// Attaches a checked caller descriptor contract to one function argument.
// The contract is consumed by TaskEdgeContract when a dynamic memref result
// is used as a communication payload.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/NeuraBackendPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

using namespace mlir;

namespace {

constexpr StringLiteral kLogicalTransferShapeAttr =
    "amoeba.logical_transfer_shape";

FailureOr<SmallVector<int64_t>> parseDimensions(StringRef specification) {
  SmallVector<int64_t> dimensions;
  if (specification.trim().empty())
    return failure();

  SmallVector<StringRef> fields;
  specification.split(fields, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  for (StringRef field : fields) {
    field = field.trim();
    int64_t extent = 0;
    if (field.empty() || field.getAsInteger(10, extent) || extent <= 0)
      return failure();
    dimensions.push_back(extent);
  }
  return dimensions;
}

struct BindExactMemrefShapePass
    : public PassWrapper<BindExactMemrefShapePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BindExactMemrefShapePass)

  BindExactMemrefShapePass() = default;
  BindExactMemrefShapePass(const BindExactMemrefShapePass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "bind-exact-memref-shape";
  }
  StringRef getDescription() const override {
    return "Bind an exact caller memref shape to one function argument";
  }

  Option<std::string> functionName{
      *this, "function",
      llvm::cl::desc("Exact func.func symbol name to annotate."),
      llvm::cl::init("")};
  Option<int64_t> argumentNumber{
      *this, "argument",
      llvm::cl::desc("Zero-based function argument index to annotate."),
      llvm::cl::init(-1)};
  Option<std::string> dimensions{
      *this, "dimensions",
      llvm::cl::desc("Comma-separated positive caller extents, e.g. 8,100."),
      llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (functionName.getValue().empty()) {
      module.emitError()
          << "bind-exact-memref-shape requires a non-empty function name";
      return signalPassFailure();
    }
    if (argumentNumber.getValue() < 0) {
      module.emitError()
          << "bind-exact-memref-shape requires a non-negative argument index";
      return signalPassFailure();
    }

    FailureOr<SmallVector<int64_t>> parsedDimensions =
        parseDimensions(dimensions.getValue());
    if (failed(parsedDimensions)) {
      module.emitError() << "bind-exact-memref-shape dimensions must be a "
                            "comma-separated list of positive integers";
      return signalPassFailure();
    }

    func::FuncOp function = module.lookupSymbol<func::FuncOp>(functionName);
    if (!function) {
      module.emitError() << "bind-exact-memref-shape function '"
                         << functionName << "' was not found";
      return signalPassFailure();
    }

    uint64_t argument = static_cast<uint64_t>(argumentNumber.getValue());
    if (argument >= function.getNumArguments()) {
      module.emitError() << "bind-exact-memref-shape argument index "
                         << argumentNumber << " is out of range for function '"
                         << functionName << "'";
      return signalPassFailure();
    }

    unsigned argumentIndex = static_cast<unsigned>(argument);
    Type argumentType = function.getArgumentTypes()[argumentIndex];
    auto memrefType = dyn_cast<MemRefType>(argumentType);
    if (!memrefType) {
      module.emitError() << "bind-exact-memref-shape argument " << argument
                         << " of function '" << functionName
                         << "' must be a ranked memref";
      return signalPassFailure();
    }

    ArrayRef<int64_t> shape = *parsedDimensions;
    if (shape.size() != static_cast<size_t>(memrefType.getRank())) {
      module.emitError() << "bind-exact-memref-shape dimensions rank "
                         << shape.size() << " does not match memref argument "
                         << "rank " << memrefType.getRank();
      return signalPassFailure();
    }
    for (auto [dimension, extent] : llvm::enumerate(shape)) {
      int64_t staticExtent = memrefType.getDimSize(dimension);
      if (!ShapedType::isDynamic(staticExtent) && staticExtent != extent) {
        module.emitError()
            << "bind-exact-memref-shape dimension " << dimension
            << " disagrees with static memref extent " << staticExtent
            << " (contract says " << extent << ")";
        return signalPassFailure();
      }
    }

    // All parsing and type checks complete before mutating the argument.  A
    // rejected invocation therefore cannot leave a partial shape contract.
    function.setArgAttr(argumentIndex, kLogicalTransferShapeAttr,
                        DenseI64ArrayAttr::get(module.getContext(), shape));
  }
};

} // namespace

namespace mlir {
namespace amoeba {
namespace neura {

std::unique_ptr<Pass> createBindExactMemrefShapePass() {
  return std::make_unique<BindExactMemrefShapePass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir
