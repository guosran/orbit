//===- ImportInput0CallerNoAliasPass.cpp ----------------------*- C++ -*-===//
//
// Import an explicit no-alias contract from the input-0 caller harness.
//
// The pass is deliberately scoped to the generated input-0 harness.  It does
// not infer aliasing from a C/C++ signature, a model, or an existing function
// attribute.  In ordinary mode every call to the selected target must come
// from the selected in-module entry point.  Prepared mode instead proves
// every call in an exact lower-generated caller module while leaving the
// prepared Taskflow/Neura target body in place.  Every memref operand must
// trace through only statically identity-preserving casts/full subviews to a
// distinct static or source-proven constant-sized allocation in that caller.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/NeuraBackendPasses.h"
#include "AnalyticalTaskCandidateCommon.h"

#include "NeuraDialect/NeuraDialect.h"
#include "TaskflowDialect/TaskflowDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

using namespace mlir;

namespace {

constexpr StringLiteral kNoAliasAttr = "amoeba.noalias";
constexpr StringLiteral kLogicalTransferShapeAttr =
    "amoeba.logical_transfer_shape";
constexpr StringLiteral kProvenAttr = "amoeba.input0_caller_noalias_proven";
constexpr StringLiteral kEvidencePathAttr =
    "amoeba.input0_caller_evidence_path";
constexpr StringLiteral kCallerAttr = "amoeba.input0_caller_function";
constexpr StringLiteral kStaticBoundAttr =
    "amoeba.input0_caller_static_bound";
constexpr StringLiteral kPreparedInputPathAttr =
    "amoeba.input0_caller_prepared_input_path";
constexpr StringLiteral kProofModeAttr =
    "amoeba.input0_caller_noalias_proof_mode";

using Shape = SmallVector<int64_t, 4>;

struct RootProof {
  Value root;
  Shape shape;
};

struct CallProof {
  func::CallOp call;
  Operation *callOperation = nullptr;
  SmallVector<std::pair<unsigned, RootProof>, 8> memrefs;
};

static std::string printValue(Value value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  value.print(stream);
  return stream.str();
}

static std::string printOperation(Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  operation->print(stream);
  return stream.str();
}

static std::string printType(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return stream.str();
}

// Read the exact bytes that are subsequently parsed.  Using the same buffer
// for parsing and evidence prevents a path-only/mtime-only caller proof from
// authorizing a different file after a concurrent rewrite.
static FailureOr<std::string> readExactText(StringRef path,
                                            std::string &reason) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    reason = (llvm::Twine("could not read evidence path '") + path + "'")
                 .str();
    return failure();
  }
  return buffer.get()->getBuffer().str();
}

static OwningOpRef<ModuleOp> parseExactModule(StringRef text, StringRef path,
                                               MLIRContext *context) {
  return parseSourceString<ModuleOp>(text, ParserConfig(context), path);
}

static std::string printShape(ArrayRef<int64_t> shape) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  for (auto [index, extent] : llvm::enumerate(shape)) {
    if (index)
      stream << "x";
    stream << extent;
  }
  return stream.str();
}

static FailureOr<SmallVector<Shape>>
parseShapeContract(StringRef specification) {
  if (specification.trim().empty())
    return failure();

  SmallVector<StringRef> shapeFields;
  specification.split(shapeFields, ';', /*MaxSplit=*/-1,
                      /*KeepEmpty=*/true);
  SmallVector<Shape> shapes;
  shapes.reserve(shapeFields.size());
  for (StringRef shapeField : shapeFields) {
    shapeField = shapeField.trim();
    if (shapeField.empty())
      return failure();
    SmallVector<StringRef> dimensionFields;
    shapeField.split(dimensionFields, 'x', /*MaxSplit=*/-1,
                     /*KeepEmpty=*/true);
    Shape shape;
    shape.reserve(dimensionFields.size());
    for (StringRef dimensionField : dimensionFields) {
      int64_t extent = 0;
      if (dimensionField.trim().empty() ||
          dimensionField.trim().getAsInteger(10, extent) || extent <= 0)
        return failure();
      shape.push_back(extent);
    }
    if (shape.empty())
      return failure();
    shapes.push_back(std::move(shape));
  }
  return shapes;
}

static bool isIdentityMemRef(MemRefType type) {
  return type && type.getLayout().isIdentity();
}

static std::optional<int64_t> constantInteger(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getValue().getSExtValue();
  return std::nullopt;
}

static FailureOr<Value> traceAllocationRoot(Value value, func::FuncOp caller,
                                             ArrayRef<int64_t> expectedShape,
                                             std::string &reason) {
  SmallPtrSet<Operation *, 16> visited;
  while (true) {
    Operation *definition = value.getDefiningOp();
    if (!definition) {
      reason = "memref operand has no defining operation (function/external "
               "root is not a caller allocation)";
      return failure();
    }
    if (!visited.insert(definition).second) {
      reason = "memref root tracing encountered a definition cycle";
      return failure();
    }
    if (isa<memref::AllocOp, memref::AllocaOp>(definition)) {
      if (definition->getParentOfType<func::FuncOp>() != caller) {
        reason = "allocation root is outside the selected caller function";
        return failure();
      }
      auto type = dyn_cast<MemRefType>(value.getType());
      if (!type || !isIdentityMemRef(type) ||
          type.getRank() != static_cast<int64_t>(expectedShape.size())) {
        reason = "allocation root must be a ranked identity memref with the "
                 "source logical rank";
        return failure();
      }
      SmallVector<Value> dynamicSizes;
      if (auto alloc = dyn_cast<memref::AllocOp>(definition))
        dynamicSizes.append(alloc.getDynamicSizes().begin(),
                            alloc.getDynamicSizes().end());
      else if (auto alloca = dyn_cast<memref::AllocaOp>(definition))
        dynamicSizes.append(alloca.getDynamicSizes().begin(),
                            alloca.getDynamicSizes().end());
      unsigned dynamicOrdinal = 0;
      for (auto [index, extent] : llvm::enumerate(type.getShape())) {
        if (!ShapedType::isDynamic(extent)) {
          if (extent != expectedShape[index]) {
            reason = "static allocation extent disagrees with the source "
                     "logical shape";
            return failure();
          }
          continue;
        }
        if (dynamicOrdinal >= dynamicSizes.size()) {
          reason = "dynamic allocation extent has no defining size operand";
          return failure();
        }
        auto size = constantInteger(dynamicSizes[dynamicOrdinal++]);
        if (!size || *size != expectedShape[index]) {
          reason = "dynamic allocation extent is not the source-proven "
                   "constant logical extent";
          return failure();
        }
      }
      if (dynamicOrdinal != dynamicSizes.size()) {
        reason = "allocation has an extra dynamic size operand outside the "
                 "source logical shape";
        return failure();
      }
      return value;
    }
    if (auto cast = dyn_cast<memref::CastOp>(definition)) {
      auto sourceType = dyn_cast<MemRefType>(cast.getSource().getType());
      auto resultType = dyn_cast<MemRefType>(cast.getResult().getType());
      if (!sourceType || !resultType || sourceType.getRank() != resultType.getRank() ||
          !isIdentityMemRef(sourceType) || !isIdentityMemRef(resultType)) {
        reason = "memref.cast is not a ranked identity-layout cast";
        return failure();
      }
      value = cast.getSource();
      continue;
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(definition)) {
      auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
      auto resultType = dyn_cast<MemRefType>(subview.getResult().getType());
      if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
          !resultType.hasStaticShape() ||
          sourceType.getRank() != resultType.getRank() ||
          !isIdentityMemRef(sourceType) || !isIdentityMemRef(resultType)) {
        reason = "memref.subview is not a static identity-layout view";
        return failure();
      }
      auto offsets = subview.getStaticOffsets();
      auto sizes = subview.getStaticSizes();
      auto strides = subview.getStaticStrides();
      if (offsets.size() != static_cast<size_t>(sourceType.getRank()) ||
          sizes.size() != static_cast<size_t>(sourceType.getRank()) ||
          strides.size() != static_cast<size_t>(sourceType.getRank())) {
        reason = "memref.subview rank metadata is incomplete";
        return failure();
      }
      for (auto [index, sourceExtent] : llvm::enumerate(sourceType.getShape())) {
        if (offsets[index] != 0 || strides[index] != 1 ||
            sizes[index] != sourceExtent ||
            resultType.getDimSize(index) != sourceExtent) {
          reason = "memref.subview is partial or has dynamic offset/size/stride";
          return failure();
        }
      }
      value = subview.getSource();
      continue;
    }
    reason = (llvm::Twine("memref operand root is unsupported operation '") +
              definition->getName().getStringRef() + "'")
                 .str();
    return failure();
  }
}

static FailureOr<Shape> argumentShape(func::FuncOp target, unsigned argument,
                                      ArrayRef<Shape> explicitShapes,
                                      unsigned memrefOrdinal,
                                      std::string &reason) {
  auto type = dyn_cast<MemRefType>(target.getArgument(argument).getType());
  if (!type) {
    reason = "selected target argument is not a ranked memref";
    return failure();
  }

  Shape shape;
  if (!explicitShapes.empty()) {
    if (memrefOrdinal >= explicitShapes.size()) {
      reason = "logical shape contract has too few memref entries";
      return failure();
    }
    shape = explicitShapes[memrefOrdinal];
  } else {
    Attribute attribute = target.getArgAttr(argument, kLogicalTransferShapeAttr);
    auto dense = dyn_cast_or_null<DenseI64ArrayAttr>(attribute);
    if (!dense) {
      reason = "dynamic target memref lacks a source-owned logical shape "
               "contract";
      return failure();
    }
    shape.assign(dense.asArrayRef().begin(), dense.asArrayRef().end());
  }
  if (shape.size() != static_cast<size_t>(type.getRank())) {
    reason = "logical shape rank disagrees with target memref rank";
    return failure();
  }
  for (auto [index, extent] : llvm::enumerate(shape)) {
    if (extent <= 0) {
      reason = "logical shape contains a non-positive extent";
      return failure();
    }
    int64_t staticExtent = type.getDimSize(index);
    if (!ShapedType::isDynamic(staticExtent) && staticExtent != extent) {
      reason = "logical shape disagrees with a static target memref extent";
      return failure();
    }
  }
  if (auto existing = dyn_cast_or_null<DenseI64ArrayAttr>(
          target.getArgAttr(argument, kLogicalTransferShapeAttr))) {
    if (static_cast<size_t>(existing.size()) != shape.size() ||
        !llvm::equal(existing.asArrayRef(), shape)) {
      reason = "logical shape disagrees with the target's existing source "
               "logical_transfer_shape contract";
      return failure();
    }
  }
  return shape;
}

// The path is an input to the proof, rather than a decorative provenance
// field.  Reparse it and compare the complete selected caller operation.  The
// target body is allowed to differ because the lowerer deliberately retains
// the prepared graph body when importing a harness, but its symbol and
// function type must match.  A caller file with the right name or a numeric
// shape fixture therefore cannot authorize a different caller or target.
static LogicalResult validateCallerEvidence(
    StringRef path, ModuleOp module, func::FuncOp target, func::FuncOp caller,
    ArrayRef<func::CallOp> calls, std::string &reason) {
  OwningOpRef<ModuleOp> evidence =
      parseSourceFile<ModuleOp>(path, ParserConfig(module.getContext()));
  if (!evidence) {
    reason = "caller-evidence is not a parseable MLIR module";
    return failure();
  }

  func::FuncOp evidenceCaller =
      evidence->lookupSymbol<func::FuncOp>(caller.getSymName());
  if (!evidenceCaller || evidenceCaller.isDeclaration()) {
    reason = "caller-evidence does not define the selected caller symbol";
    return failure();
  }
  func::FuncOp evidenceTarget =
      evidence->lookupSymbol<func::FuncOp>(target.getSymName());
  if (!evidenceTarget || evidenceTarget.isDeclaration()) {
    reason = "caller-evidence does not define the selected target symbol";
    return failure();
  }
  if (evidenceTarget.getFunctionType() != target.getFunctionType()) {
    reason = "caller-evidence target signature differs from the prepared "
             "target (no caller proof is imported)";
    return failure();
  }

  // Operation printing is canonical for a parsed module, so this binds the
  // proof to the exact caller text (including all casts, allocations, and
  // call operands) while avoiding a digest or an mtime-only check.
  if (printOperation(evidenceCaller.getOperation()) !=
      printOperation(caller.getOperation())) {
    reason = "caller-evidence selected caller text differs from the in-module "
             "caller";
    return failure();
  }

  SmallVector<func::CallOp, 8> evidenceCalls;
  evidence->walk([&](func::CallOp call) {
    if (call.getCallee() == target.getSymName())
      evidenceCalls.push_back(call);
  });
  if (evidenceCalls.size() != calls.size()) {
    reason = "caller-evidence target-call count differs from the in-module "
             "caller";
    return failure();
  }
  for (auto [index, evidenceCall] : llvm::enumerate(evidenceCalls)) {
    if (evidenceCall->getParentOfType<func::FuncOp>() != evidenceCaller ||
        evidenceCall.getNumOperands() != calls[index]->getNumOperands()) {
      reason = "caller-evidence target call is not in the selected caller or "
               "has a different operand count";
      return failure();
    }
    for (auto [operandIndex, operand] : llvm::enumerate(evidenceCall.getOperands()))
      if (operand.getType() !=
          calls[index]->getOperand(operandIndex).getType()) {
        reason = "caller-evidence target call operand types differ from the "
                 "in-module caller";
        return failure();
      }
  }
  return success();
}

// Prove the complete closed-world caller graph in one module.  This helper is
// used for both the current module and the separately parsed lower-generated
// evidence module; keeping the root, shape, scalar, and callsite checks shared
// prevents the prepared-module path from accepting a weaker external proof.
static LogicalResult proveCallerAllocations(
    ModuleOp module, func::FuncOp target, func::FuncOp caller,
    ArrayRef<unsigned> memrefArguments, unsigned scalarArgument,
    int64_t staticBound, ArrayRef<Shape> expectedShapes,
    SmallVectorImpl<CallProof> &proofs, std::string &reason) {
  SmallVector<func::CallOp, 8> calls;
  module.walk([&](func::CallOp call) {
    if (call.getCallee() == target.getSymName())
      calls.push_back(call);
  });
  if (calls.empty()) {
    reason = "selected target has no in-module callsite";
    return failure();
  }

  proofs.clear();
  proofs.reserve(calls.size());
  for (func::CallOp call : calls) {
    if (call->getParentOfType<func::FuncOp>() != caller) {
      reason = "every target callsite must be in the selected closed-world "
               "caller; an additional/unknown caller is not proven";
      return failure();
    }
    if (call.getNumOperands() != target.getNumArguments()) {
      reason = "caller operand count does not match target signature";
      return failure();
    }
    auto bound = constantInteger(call.getOperand(scalarArgument));
    if (!bound || *bound != staticBound) {
      reason = "caller scalar bound is not the supplied compile-time value";
      return failure();
    }

    CallProof proof;
    proof.call = call;
    proof.callOperation = call.getOperation();
    SmallVector<Value> roots;
    for (auto [ordinal, argument] : llvm::enumerate(memrefArguments)) {
      Value operand = call.getOperand(argument);
      if (operand.getType() != target.getArgument(argument).getType()) {
        reason = (llvm::Twine("call operand type disagrees with target "
                              "argument ") +
                  llvm::Twine(argument))
                     .str();
        return failure();
      }
      auto operandType = dyn_cast<MemRefType>(operand.getType());
      if (!operandType ||
          operandType.getRank() !=
              static_cast<int64_t>(expectedShapes[ordinal].size()) ||
          !isIdentityMemRef(operandType)) {
        reason = "caller memref operand must be a ranked identity memref";
        return failure();
      }
      std::string rootReason;
      FailureOr<Value> root = traceAllocationRoot(
          operand, caller, expectedShapes[ordinal], rootReason);
      if (failed(root)) {
        reason = (llvm::Twine("argument ") + llvm::Twine(argument) + ": " +
                  rootReason)
                     .str();
        return failure();
      }
      auto rootType = dyn_cast<MemRefType>((*root).getType());
      if (!rootType ||
          rootType.getElementType() != operandType.getElementType() ||
          rootType.getMemorySpace() != operandType.getMemorySpace() ||
          rootType.getRank() !=
              static_cast<int64_t>(expectedShapes[ordinal].size())) {
        reason =
            (llvm::Twine("caller allocation shape/type disagrees with the "
                         "source logical shape for argument ") +
             llvm::Twine(argument))
                .str();
        return failure();
      }
      for (Value priorRoot : roots) {
        if (priorRoot == *root) {
          reason = "two target arguments trace to the same caller allocation";
          return failure();
        }
      }
      roots.push_back(*root);
      proof.memrefs.push_back(
          {argument, RootProof{*root, expectedShapes[ordinal]}});
    }
    proofs.push_back(std::move(proof));
  }
  return success();
}

static LogicalResult rejectPreparedTargetUses(ModuleOp module,
                                              func::FuncOp target,
                                              std::string &reason) {
  // SymbolTable returns nullopt when an unknown operation may contain symbol
  // uses.  Treat that as an unknown caller rather than assuming the prepared
  // body has no invocation edge.
  auto uses = SymbolTable::getSymbolUses(target.getOperation(), module);
  if (!uses) {
    reason = "prepared module contains an operation with unknown symbol uses "
             "while proving the target has no in-module caller";
    return failure();
  }
  for (const SymbolTable::SymbolUse &use : *uses) {
    reason = (llvm::Twine("prepared module has an in-module use of target '") +
              target.getSymName() + "' in operation '" +
              use.getUser()->getName().getStringRef() + "'")
                 .str();
    return failure();
  }
  return success();
}

static LogicalResult writeEvidence(StringRef path, StringRef callerEvidence,
                                   StringRef targetName, StringRef callerName,
                                   StringRef callerText,
                                   StringRef targetSignature,
                                   int64_t staticBound,
                                   ArrayRef<Shape> shapes,
                                   ArrayRef<CallProof> proofs,
                                   bool preparedExternal,
                                   StringRef preparedInput,
                                   StringRef preparedFunctionText,
                                   StringRef callerEvidenceFileText,
                                   StringRef preparedInputFileText) {

  llvm::json::Array shapeArray;
  for (const Shape &shape : shapes)
    shapeArray.push_back(printShape(shape));

  llvm::json::Array callArray;
  for (const CallProof &proof : proofs) {
    llvm::json::Object callObject;
    callObject["caller"] = callerName.str();
    callObject["callee"] = targetName.str();
    callObject["call_text"] = printOperation(proof.callOperation);
    llvm::json::Array arguments;
    for (const auto &[argument, root] : proof.memrefs) {
      llvm::json::Object argumentObject;
      argumentObject["argument"] = static_cast<int64_t>(argument);
      argumentObject["root"] = printValue(root.root);
      argumentObject["root_shape"] = printShape(root.shape);
      arguments.push_back(std::move(argumentObject));
    }
    callObject["memref_arguments"] = std::move(arguments);
    callArray.push_back(std::move(callObject));
  }

  llvm::json::Object record;
  record["schema"] = preparedExternal
                          ? "amoeba.input0-caller-noalias-proof-v2"
                          : "amoeba.input0-caller-noalias-proof-v1";
  record["proof_mode"] = preparedExternal ? "prepared-external-caller"
                                           : "in-module-caller";
  record["caller_evidence_path"] = callerEvidence.str();
  record["caller_function"] = callerName.str();
  record["target_function"] = targetName.str();
  record["caller_function_text"] = callerText.str();
  record["target_signature"] = targetSignature.str();
  record["static_bound"] = staticBound;
  record["logical_shapes"] = std::move(shapeArray);
  record["calls"] = std::move(callArray);
  if (preparedExternal) {
    record["prepared_input_path"] = preparedInput.str();
    record["prepared_function_text"] = preparedFunctionText.str();
    // These exact text fields deliberately avoid a digest and bind the
    // machine record to the bytes that were parsed for both external files.
    record["caller_evidence_file_text"] = callerEvidenceFileText.str();
    record["prepared_input_file_text"] = preparedInputFileText.str();
  }
  std::string error;
  bool written = mlir::amoeba::neura::joint_scheduling::writeAtomically(
      path, [&](llvm::raw_ostream &output) {
        output << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(record)));
        return true;
      }, error);
  return success(written);
}

struct ImportInput0CallerNoAliasPass
    : PassWrapper<ImportInput0CallerNoAliasPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportInput0CallerNoAliasPass)

  ImportInput0CallerNoAliasPass() = default;
  ImportInput0CallerNoAliasPass(const ImportInput0CallerNoAliasPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "import-input0-caller-noalias";
  }
  StringRef getDescription() const override {
    return "Import noalias only from a verified input-0 caller allocation graph";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    func::FuncDialect, LLVM::LLVMDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    neura::NeuraDialect, taskflow::TaskflowDialect>();
  }

  Option<std::string> targetFunction{
      *this, "target", llvm::cl::desc("Target func.func symbol"),
      llvm::cl::init("")};
  Option<std::string> callerFunction{
      *this, "caller", llvm::cl::desc("Closed-world input-0 caller symbol"),
      llvm::cl::init("main")};
  Option<std::string> callerEvidence{
      *this, "caller-evidence",
      llvm::cl::desc("Exact input-0 caller evidence path (no digest)"),
      llvm::cl::init("")};
  Option<std::string> logicalShapes{
      *this, "logical-shapes",
      llvm::cl::desc("Semicolon-separated x-separated memref shapes"),
      llvm::cl::init("")};
  Option<std::string> evidenceOutput{
      *this, "evidence-output", llvm::cl::desc("Proof JSON output path"),
      llvm::cl::init("")};
  Option<int64_t> staticBound{
      *this, "static-bound",
      llvm::cl::desc("Source-proven scalar loop bound supplied by caller"),
      llvm::cl::init(-1)};
  Option<bool> input0Only{
      *this, "input0-only",
      llvm::cl::desc("Required explicit input-0 scope marker"),
      llvm::cl::init(false)};
  Option<bool> preparedExternalCaller{
      *this, "prepared-external-caller",
      llvm::cl::desc("Prove a prepared target from an external lower-generated caller"),
      llvm::cl::init(false)};
  Option<std::string> preparedInput{
      *this, "prepared-input",
      llvm::cl::desc("Exact prepared module path used for the current input"),
      llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto fail = [&](Twine message) {
      module.emitError() << "input-0 caller noalias proof: " << message;
      signalPassFailure();
    };

    if (!input0Only) {
      fail("requires --input0-only=true; the pass is not a general alias "
           "inference mechanism");
      return;
    }
    if (targetFunction.empty() || callerFunction.empty() ||
        callerEvidence.empty() || staticBound <= 0) {
      fail("requires target, caller, caller-evidence, and positive "
           "static-bound options");
      return;
    }

    func::FuncOp target = module.lookupSymbol<func::FuncOp>(targetFunction);
    if (!target || target.isDeclaration()) {
      fail("target must be one in-module defined function");
      return;
    }
    func::FuncOp caller;
    if (!preparedExternalCaller) {
      caller = module.lookupSymbol<func::FuncOp>(callerFunction);
      if (!caller || caller.isDeclaration()) {
        fail("caller must be one in-module defined function");
        return;
      }
      if (target == caller) {
        fail("target and caller must be distinct functions");
        return;
      }
    }
    if (!llvm::sys::path::is_absolute(callerEvidence)) {
      fail("caller-evidence must be an absolute exact path");
      return;
    }
    if (!llvm::sys::fs::exists(callerEvidence)) {
      fail("caller-evidence path does not exist");
      return;
    }
    if (preparedExternalCaller) {
      if (preparedInput.empty() ||
          !llvm::sys::path::is_absolute(preparedInput)) {
        fail("prepared-external-caller requires an absolute prepared-input "
             "path");
        return;
      }
      if (!llvm::sys::fs::exists(preparedInput)) {
        fail("prepared-input path does not exist");
        return;
      }
    } else if (!preparedInput.empty()) {
      fail("prepared-input is only valid with prepared-external-caller=true");
      return;
    }

    FailureOr<SmallVector<Shape>> explicitShapes =
        parseShapeContract(logicalShapes);
    if (!logicalShapes.empty() && failed(explicitShapes)) {
      fail("logical-shapes must use positive x-separated shapes separated by "
           "semicolons");
      return;
    }
    ArrayRef<Shape> shapeContract =
        logicalShapes.empty() ? ArrayRef<Shape>() : *explicitShapes;

    SmallVector<unsigned> memrefArguments;
    std::optional<unsigned> scalarArgument;
    for (unsigned index = 0; index < target.getNumArguments(); ++index) {
      Type type = target.getArgument(index).getType();
      if (isa<MemRefType>(type)) {
        memrefArguments.push_back(index);
        continue;
      }
      if (!isa<IntegerType, IndexType>(type) || scalarArgument) {
        fail("target must have exactly one integer/index static-bound argument");
        return;
      }
      scalarArgument = index;
    }
    if (!scalarArgument || memrefArguments.empty()) {
      fail("target must have one scalar bound and at least one memref argument");
      return;
    }
    std::string staticBoundName =
        (llvm::Twine("amoeba.static_bound.arg.") +
         llvm::Twine(*scalarArgument))
            .str();
    if (auto existingBound =
            target->getAttrOfType<IntegerAttr>(staticBoundName)) {
      if (existingBound.getInt() != staticBound) {
        fail("caller static bound disagrees with the target's existing "
             "source static_bound contract");
        return;
      }
    }
    if (!shapeContract.empty() && shapeContract.size() != memrefArguments.size()) {
      fail("logical-shapes count does not match target memref argument count");
      return;
    }

    SmallVector<Shape> expectedShapes;
    expectedShapes.reserve(memrefArguments.size());
    for (auto [ordinal, argument] : llvm::enumerate(memrefArguments)) {
      std::string reason;
      FailureOr<Shape> shape = argumentShape(
          target, argument, shapeContract, ordinal, reason);
      if (failed(shape)) {
        fail(llvm::Twine("argument ") + llvm::Twine(argument) + ": " + reason);
        return;
      }
      expectedShapes.push_back(std::move(*shape));
    }

    SmallVector<CallProof, 8> proofs;
    std::string callerText;
    std::string preparedFunctionText;
    std::string callerEvidenceFileText;
    std::string preparedInputFileText;
    std::string proofReason;
    OwningOpRef<ModuleOp> preparedInputModule;
    OwningOpRef<ModuleOp> evidenceModule;

    if (!preparedExternalCaller) {
      SmallVector<func::CallOp> calls;
      module.walk([&](func::CallOp call) {
        if (call.getCallee() == target.getSymName())
          calls.push_back(call);
      });
      if (calls.empty()) {
        fail("selected target has no in-module callsite");
        return;
      }

      if (failed(validateCallerEvidence(callerEvidence, module, target, caller,
                                        calls, proofReason))) {
        fail(proofReason);
        return;
      }
      if (failed(proveCallerAllocations(
              module, target, caller, memrefArguments, *scalarArgument,
              staticBound, expectedShapes, proofs, proofReason))) {
        fail(proofReason);
        return;
      }
      callerText = printOperation(caller.getOperation());
    } else {
      // The prepared target is intentionally not lowered or cloned here.  A
      // separate lower-generated module supplies only its closed-world ABI
      // invocation proof; this preserves the Taskflow/Neura body for the
      // later replica/tiling passes.
      auto existingBound =
          target->getAttrOfType<IntegerAttr>(staticBoundName);
      if (!existingBound) {
        fail("prepared target must carry its source-owned static_bound "
             "contract");
        return;
      }
      if (failed(rejectPreparedTargetUses(module, target, proofReason))) {
        fail(proofReason);
        return;
      }

      FailureOr<std::string> preparedText =
          readExactText(preparedInput, proofReason);
      if (failed(preparedText)) {
        fail(proofReason);
        return;
      }
      preparedInputFileText = std::move(*preparedText);
      preparedInputModule =
          parseExactModule(preparedInputFileText, preparedInput,
                           module.getContext());
      if (!preparedInputModule) {
        fail("prepared-input is not a parseable MLIR module");
        return;
      }
      func::FuncOp inputTarget =
          preparedInputModule->lookupSymbol<func::FuncOp>(targetFunction);
      if (!inputTarget || inputTarget.isDeclaration()) {
        fail("prepared-input does not define the selected target symbol");
        return;
      }
      if (inputTarget.getFunctionType() != target.getFunctionType() ||
          printOperation(inputTarget.getOperation()) !=
              printOperation(target.getOperation())) {
        fail("prepared-input target function text/signature differs from the "
             "current prepared target");
        return;
      }
      preparedFunctionText = printOperation(target.getOperation());

      FailureOr<std::string> evidenceText =
          readExactText(callerEvidence, proofReason);
      if (failed(evidenceText)) {
        fail(proofReason);
        return;
      }
      callerEvidenceFileText = std::move(*evidenceText);
      evidenceModule =
          parseExactModule(callerEvidenceFileText, callerEvidence,
                           module.getContext());
      if (!evidenceModule) {
        fail("caller-evidence is not a parseable MLIR module");
        return;
      }
      func::FuncOp evidenceCaller =
          evidenceModule->lookupSymbol<func::FuncOp>(callerFunction);
      func::FuncOp evidenceTarget =
          evidenceModule->lookupSymbol<func::FuncOp>(targetFunction);
      if (!evidenceCaller || evidenceCaller.isDeclaration()) {
        fail("caller-evidence does not define the selected caller symbol");
        return;
      }
      if (!evidenceTarget || evidenceTarget.isDeclaration()) {
        fail("caller-evidence does not define the selected target symbol");
        return;
      }
      if (evidenceTarget.getFunctionType() != target.getFunctionType()) {
        fail("caller-evidence target signature differs from the prepared "
             "target");
        return;
      }
      if (evidenceTarget == evidenceCaller) {
        fail("caller-evidence target and caller must be distinct functions");
        return;
      }
      auto evidenceUses =
          SymbolTable::getSymbolUses(evidenceTarget.getOperation(),
                                     *evidenceModule);
      if (!evidenceUses) {
        fail("caller-evidence contains unknown symbol uses for the target");
        return;
      }
      for (const SymbolTable::SymbolUse &use : *evidenceUses) {
        auto call = dyn_cast<func::CallOp>(use.getUser());
        if (!call || call->getParentOfType<func::FuncOp>() != evidenceCaller) {
          fail("caller-evidence target has a non-direct or non-selected "
               "caller use");
          return;
        }
      }

      // The prepared function's source contracts, rather than an external
      // shape string, are authoritative.  Require the generated caller's
      // target to carry and agree with the same contracts.
      if (auto evidenceBound =
              evidenceTarget->getAttrOfType<IntegerAttr>(staticBoundName)) {
        if (evidenceBound.getInt() != staticBound) {
          fail("caller-evidence target static_bound disagrees with the "
               "prepared contract");
          return;
        }
      } else {
        fail("caller-evidence target lacks its source-owned static_bound "
             "contract");
        return;
      }
      for (auto [ordinal, argument] : llvm::enumerate(memrefArguments)) {
        if (!evidenceTarget.getArgAttr(argument,
                                      kLogicalTransferShapeAttr)) {
          fail("caller-evidence target lacks a source-owned logical shape "
               "contract");
          return;
        }
        FailureOr<Shape> evidenceShape = argumentShape(
            evidenceTarget, argument, expectedShapes, ordinal, proofReason);
        if (failed(evidenceShape)) {
          fail(llvm::Twine("caller-evidence argument ") +
               llvm::Twine(argument) + ": " + proofReason);
          return;
        }
      }
      if (failed(proveCallerAllocations(
              *evidenceModule, evidenceTarget, evidenceCaller, memrefArguments,
              *scalarArgument, staticBound, expectedShapes, proofs,
              proofReason))) {
        fail(proofReason);
        return;
      }
      callerText = printOperation(evidenceCaller.getOperation());
    }

    if (!evidenceOutput.empty() &&
        failed(writeEvidence(
            evidenceOutput, callerEvidence, target.getSymName(),
            callerFunction, callerText, printType(target.getFunctionType()),
            staticBound, expectedShapes, proofs, preparedExternalCaller,
            preparedInput, preparedFunctionText, callerEvidenceFileText,
            preparedInputFileText))) {
      fail("could not write evidence output");
      return;
    }

    // All checks completed.  This is the only mutation point, so a rejected
    // callsite never leaves a partially annotated target function.  The
    // resulting attribute is an input-0 invocation no-alias fact for this
    // closed-world caller; it is not a general C++ ABI or numerical-semantic
    // assertion about the target body.
    for (unsigned argument : memrefArguments)
      target.setArgAttr(argument, kNoAliasAttr,
                        UnitAttr::get(module.getContext()));
    target->setAttr(kProvenAttr, UnitAttr::get(module.getContext()));
    target->setAttr(kEvidencePathAttr,
                    StringAttr::get(module.getContext(), callerEvidence));
    target->setAttr(kCallerAttr,
                    StringAttr::get(module.getContext(), callerFunction));
    target->setAttr(kStaticBoundAttr,
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                     staticBound));
    target->setAttr(kProofModeAttr,
                    StringAttr::get(module.getContext(),
                                    preparedExternalCaller
                                        ? "prepared-external-caller"
                                        : "in-module-caller"));
    if (preparedExternalCaller)
      target->setAttr(
          kPreparedInputPathAttr,
          StringAttr::get(module.getContext(), preparedInput));
  }
};

} // namespace

namespace mlir::amoeba::neura {

std::unique_ptr<Pass> createImportInput0CallerNoAliasPass() {
  return std::make_unique<ImportInput0CallerNoAliasPass>();
}

} // namespace mlir::amoeba::neura
