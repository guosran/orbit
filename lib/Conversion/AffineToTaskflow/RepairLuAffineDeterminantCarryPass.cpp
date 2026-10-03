#include "Conversion/RepairLuAffineDeterminantCarryPass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdint>
#include <memory>
#include <string>

using namespace mlir;

namespace {

constexpr StringLiteral kPassName = "repair-lu-affine-determinant-carry";
constexpr StringLiteral kPassVersion =
    "repair-lu-affine-determinant-carry-v1";

constexpr StringLiteral kSourceAttribute =
    "amoeba.lu_affine_repair_source_file";
constexpr StringLiteral kSourceVerifiedAttribute =
    "amoeba.lu_affine_repair_source_verified";
constexpr StringLiteral kInputStatusAttribute =
    "amoeba.lu_affine_repair_input_status";
constexpr StringLiteral kRepairStatusAttribute =
    "amoeba.lu_affine_repair_status";
constexpr StringLiteral kPassAttribute = "amoeba.lu_affine_repair_pass";
constexpr StringLiteral kReasonAttribute = "amoeba.lu_affine_repair_reason";

// These snippets deliberately describe the frozen LU source contract.  The
// pass is an artifact repair, rather than a general determinant rewrite, so a
// source file that merely happens to contain a similarly named function is
// rejected.
constexpr StringLiteral kFixedScaleDefinition = "#define FIXED_SCALE 1024";
constexpr StringLiteral kLuFunctionContract = R"SRC(void lu_func(int matrix_size, const int input[MAX_MATRIX][MAX_MATRIX],
             const int rhs[MAX_MATRIX],
             int working_matrix[MAX_MATRIX][MAX_MATRIX],
             int lower[MAX_MATRIX][MAX_MATRIX],
             int upper[MAX_MATRIX][MAX_MATRIX],
             int forward_solution[MAX_MATRIX], int solution[MAX_MATRIX],
             int inverse[MAX_MATRIX][MAX_MATRIX], int determinant_out[1]) {)SRC";
constexpr StringLiteral kDeterminantRecurrence = R"SRC(  // Task 8: Q10 determinant from the product of U's diagonal.
  determinant_out[0] = FIXED_SCALE;
  for (int idx = 0; idx < matrix_size; ++idx) {
    determinant_out[0] = determinant_out[0] * upper[idx][idx] / FIXED_SCALE;
  })SRC";

constexpr StringLiteral kFrozenLuSource = R"ORBITLU(// LU decomposition benchmark for AMOEBA evaluation.
//
// Reference:
// - Streaming-Bench, commit 333782f78d5475c8b33b11ff2b9ba9d75c93ca49:
//   lu/application/lu.cpp and lu/{init,decompose,solver0,solver1,invert,
//   determinant}.
//
// The original Streaming-Bench LU code uses LUP decomposition with pivoting.
// This affine-friendly proxy instead uses no-pivot Doolittle factorization
// with Q10 fixed-point arithmetic and a static matrix bound. Benchmark inputs
// must be small, strictly diagonally dominant matrices, which guarantees
// nonzero pivots without row permutations and keeps intermediate products in
// signed 32-bit range. Removing pivot search makes the loop bounds affine while
// preserving the defining recursive L/U data dependences.
//
// All matrix, vector, inverse, solution, and determinant values use Q10.
// Products are requantized to Q10 after multiplication.
//
// Expected task count after affine-to-taskflow: 9 top-level loop nests.

#define MAX_MATRIX 100
#define FIXED_SCALE 1024

void lu_func(int matrix_size, const int input[MAX_MATRIX][MAX_MATRIX],
             const int rhs[MAX_MATRIX],
             int working_matrix[MAX_MATRIX][MAX_MATRIX],
             int lower[MAX_MATRIX][MAX_MATRIX],
             int upper[MAX_MATRIX][MAX_MATRIX],
             int forward_solution[MAX_MATRIX], int solution[MAX_MATRIX],
             int inverse[MAX_MATRIX][MAX_MATRIX], int determinant_out[1]) {
  // Stage outputs are caller-provided scratch buffers, matching the GCN and
  // Harris benchmark style. This keeps the generated affine IR from carrying
  // large function-local memref.alloca objects.
  //
  // The caller zero-initializes all scratch buffers before invoking this
  // function; predicated fixed-bound loops may read inactive entries before
  // masking them.

  // Task 0: Copy the input matrix into the working matrix.
  for (int row = 0; row < matrix_size; ++row) {
    for (int col = 0; col < matrix_size; ++col) {
      working_matrix[row][col] = input[row][col];
    }
  }

  // Task 1: Initialize L/U storage.
  for (int row = 0; row < matrix_size; ++row) {
    for (int col = 0; col < matrix_size; ++col) {
      lower[row][col] = row == col ? FIXED_SCALE : 0;
      upper[row][col] = 0;
      inverse[row][col] = row == col ? FIXED_SCALE : 0;
    }
  }

  // Task 2: Compact no-pivot Doolittle factorization.
  //
  // `working_matrix` stores unit-diagonal L below the diagonal and U on/above
  // it. Row-major order makes every L/U value available before it is consumed.
  //
  // The benchmark input contract guarantees every diagonal pivot is nonzero.
  for (int row = 0; row < matrix_size; ++row) {
    for (int col = 0; col < matrix_size; ++col) {
      for (int red = 0; red < matrix_size; ++red) {
        int before_row = red < row ? 1 : 0;
        int before_col = red < col ? 1 : 0;
        int active = before_row * before_col;
        int value = working_matrix[row][col];
        int product = working_matrix[row][red] * working_matrix[red][col];
        working_matrix[row][col] = value - active * (product / FIXED_SCALE);
      }
      int below_diagonal = row > col ? 1 : 0;
      int pivot = working_matrix[col][col];
      int divisor =
          FIXED_SCALE + below_diagonal * (pivot - FIXED_SCALE);
      working_matrix[row][col] =
          working_matrix[row][col] * FIXED_SCALE / divisor;
    }
  }

  // Task 3: Materialize separate unit-diagonal L and upper-triangular U.
  for (int row = 0; row < matrix_size; ++row) {
    for (int col = 0; col < matrix_size; ++col) {
      int below_diagonal = row > col ? 1 : 0;
      int on_diagonal = row == col ? 1 : 0;
      int on_or_above_diagonal = 1 - below_diagonal;
      int compact_value = working_matrix[row][col];
      lower[row][col] = below_diagonal * compact_value +
                        on_diagonal * FIXED_SCALE;
      upper[row][col] = on_or_above_diagonal * compact_value;
    }
  }

  // Task 4: Forward substitution, L * y = rhs.
  for (int row = 0; row < matrix_size; ++row) {
    forward_solution[row] = rhs[row];
    for (int col = 0; col < matrix_size; ++col) {
      int active = col < row ? 1 : 0;
      int value = forward_solution[row];
      int product = lower[row][col] * forward_solution[col];
      forward_solution[row] = value - active * (product / FIXED_SCALE);
    }
  }

  // Task 5: Backward substitution, U * x = y.
  for (int rev = 0; rev < matrix_size; ++rev) {
    int row = matrix_size - 1 - rev;
    solution[row] = forward_solution[row];
    for (int col = 0; col < matrix_size; ++col) {
      int active = col > row ? 1 : 0;
      int value = solution[row];
      int product = upper[row][col] * solution[col];
      solution[row] = value - active * (product / FIXED_SCALE);
    }
    solution[row] = solution[row] * FIXED_SCALE / upper[row][row];
  }

  // Task 6: Invert L by solving each identity column.
  for (int col = 0; col < matrix_size; ++col) {
    for (int row = 0; row < matrix_size; ++row) {
      inverse[row][col] = row == col ? FIXED_SCALE : 0;
      for (int red = 0; red < matrix_size; ++red) {
        int active = red < row ? 1 : 0;
        int value = inverse[row][col];
        int product = lower[row][red] * inverse[red][col];
        inverse[row][col] = value - active * (product / FIXED_SCALE);
      }
    }
  }

  // Task 7: Apply the U inverse to finish A^-1.
  for (int col = 0; col < matrix_size; ++col) {
    for (int rev = 0; rev < matrix_size; ++rev) {
      int row = matrix_size - 1 - rev;
      for (int red = 0; red < matrix_size; ++red) {
        int active = red > row ? 1 : 0;
        int value = inverse[row][col];
        int product = upper[row][red] * inverse[red][col];
        inverse[row][col] = value - active * (product / FIXED_SCALE);
      }
      inverse[row][col] = inverse[row][col] * FIXED_SCALE / upper[row][row];
    }
  }

  // Task 8: Q10 determinant from the product of U's diagonal.
  determinant_out[0] = FIXED_SCALE;
  for (int idx = 0; idx < matrix_size; ++idx) {
    determinant_out[0] = determinant_out[0] * upper[idx][idx] / FIXED_SCALE;
  }
}
)ORBITLU";

struct SourceContractResult {
  bool verified = false;
  std::string reason;
};

static SourceContractResult verifySourceContract(StringRef source) {
  if (source != kFrozenLuSource)
    return {false, "source differs from the exact audited frozen LU program"};
  if (source.count(kFixedScaleDefinition) != 1) {
    return {false, "source must contain exactly one '#define FIXED_SCALE 1024'"};
  }
  if (source.count(kLuFunctionContract) != 1) {
    return {false, "frozen lu_func function contract was not found exactly once"};
  }
  if (source.count(kDeterminantRecurrence) != 1) {
    return {false, "frozen Task 8 determinant recurrence was not found exactly once"};
  }
  return {true, "source contract verified"};
}

static bool isI32(Type type) {
  auto integer = dyn_cast<IntegerType>(type);
  return integer && integer.getWidth() == 32;
}

static bool isMatrixMemref(Value value) {
  auto memref = dyn_cast<MemRefType>(value.getType());
  if (!memref || memref.getRank() != 2 || !isI32(memref.getElementType()))
    return false;
  return memref.getDimSize(0) == ShapedType::kDynamic &&
         memref.getDimSize(1) == 100;
}

static bool isVectorMemref(Value value) {
  auto memref = dyn_cast<MemRefType>(value.getType());
  if (!memref || memref.getRank() != 1 || !isI32(memref.getElementType()))
    return false;
  return memref.getDimSize(0) == ShapedType::kDynamic;
}

static bool isZeroAffineMap(AffineMap map) {
  if (map.getNumDims() != 0 || map.getNumSymbols() != 0 ||
      map.getNumResults() != 1)
    return false;
  auto constant = dyn_cast<AffineConstantExpr>(map.getResult(0));
  return constant && constant.getValue() == 0;
}

static bool isZeroLoad(affine::AffineLoadOp load, Value memref) {
  return load.getMemRef() == memref && load.getMapOperands().empty() &&
         isZeroAffineMap(load.getAffineMap());
}

static bool isZeroStore(affine::AffineStoreOp store, Value memref) {
  return store.getMemRef() == memref && store.getMapOperands().empty() &&
         isZeroAffineMap(store.getAffineMap());
}

static bool isDiagonalLoad(affine::AffineLoadOp load, Value upper, Value iv) {
  if (load.getMemRef() != upper)
    return false;
  AffineMap map = load.getAffineMap();
  if (map.getNumSymbols() != 0 || map.getNumResults() != 2)
    return false;
  auto first = dyn_cast<AffineDimExpr>(map.getResult(0));
  auto second = dyn_cast<AffineDimExpr>(map.getResult(1));
  if (!first || !second)
    return false;

  // MLIR canonicalizes the source spelling [iv, iv] to one dimension with
  // the map (d0) -> (d0, d0).  Accept that canonical form and the equivalent
  // two-dimension identity form, but reject every other index shape.
  if (map.getNumDims() == 1 && load.getMapOperands().size() == 1 &&
      load.getMapOperands().front() == iv)
    return first.getPosition() == 0 && second.getPosition() == 0;
  return map.getNumDims() == 2 && load.getMapOperands().size() == 2 &&
         load.getMapOperands()[0] == iv && load.getMapOperands()[1] == iv &&
         first.getPosition() == 0 && second.getPosition() == 1;
}

static bool isScaleConstant(arith::ConstantOp constant) {
  auto integer = dyn_cast<IntegerAttr>(constant.getValue());
  return integer && integer.getType().isInteger(32) && integer.getInt() == 1024;
}

static bool isExactFunctionContract(func::FuncOp function) {
  FunctionType type = function.getFunctionType();
  if (type.getNumInputs() != 10 || type.getNumResults() != 0 ||
      !isI32(type.getInput(0)))
    return false;

  for (unsigned index : {1u, 3u, 4u, 5u, 8u}) {
    if (!isMatrixMemref(function.getArgument(index)))
      return false;
  }
  for (unsigned index : {2u, 6u, 7u, 9u}) {
    if (!isVectorMemref(function.getArgument(index)))
      return false;
  }
  return true;
}

static bool isExactLoopBound(affine::AffineForOp loop, func::FuncOp function) {
  if (loop->getNumResults() != 0 || loop.getBody()->getNumArguments() != 1 ||
      loop.getStepAsInt() != 1 || !loop.hasConstantLowerBound() ||
      loop.getConstantLowerBound() != 0)
    return false;

  AffineMap map = loop.getUpperBoundMap();
  if (map.getNumDims() != 0 || map.getNumSymbols() != 1 ||
      map.getNumResults() != 1 || loop.getUpperBoundOperands().size() != 1)
    return false;
  auto symbol = dyn_cast<AffineSymbolExpr>(map.getResult(0));
  if (!symbol || symbol.getPosition() != 0)
    return false;

  Value bound = loop.getUpperBoundOperands().front();
  auto indexCast = bound.getDefiningOp<arith::IndexCastOp>();
  return indexCast && indexCast.getIn() == function.getArgument(0) &&
         bound.getType().isIndex();
}

static bool isDirectlyInLoopBody(Operation *operation,
                                 affine::AffineForOp loop) {
  return operation->getBlock() == loop.getBody();
}

enum class CarryBodyKind { Malformed, Correct, Invalid };

struct CarryBody {
  CarryBodyKind kind = CarryBodyKind::Invalid;
  affine::AffineLoadOp diagonalLoad;
  affine::AffineLoadOp carryLoad;
  arith::MulIOp multiply;
  arith::DivSIOp divide;
  affine::AffineStoreOp store;
};

static CarryBody inspectCarryBody(affine::AffineForOp loop, Value upper,
                                  Value determinant, Value scale) {
  CarryBody body;
  if (loop.getBody()->empty() ||
      !isa<affine::AffineYieldOp>(loop.getBody()->back()))
    return body;

  SmallVector<Operation *> operations;
  for (Operation &operation : loop.getBody()->getOperations()) {
    if (&operation != &loop.getBody()->back())
      operations.push_back(&operation);
  }

  if (operations.size() != 4 && operations.size() != 5)
    return body;

  body.diagonalLoad = dyn_cast<affine::AffineLoadOp>(operations[0]);
  if (!body.diagonalLoad ||
      !isDiagonalLoad(body.diagonalLoad, upper, loop.getInductionVar()))
    return CarryBody();

  unsigned cursor = 1;
  if (operations.size() == 5) {
    body.carryLoad = dyn_cast<affine::AffineLoadOp>(operations[cursor++]);
    if (!body.carryLoad || !isZeroLoad(body.carryLoad, determinant))
      return CarryBody();
  }

  body.multiply = dyn_cast<arith::MulIOp>(operations[cursor++]);
  body.divide = dyn_cast<arith::DivSIOp>(operations[cursor++]);
  body.store = dyn_cast<affine::AffineStoreOp>(operations[cursor++]);
  if (!body.multiply || !body.divide || !body.store ||
      !isZeroStore(body.store, determinant) || cursor != operations.size())
    return CarryBody();

  if (body.divide.getLhs() != body.multiply.getResult() ||
      body.divide.getRhs() != scale ||
      body.store.getValue() != body.divide.getResult())
    return CarryBody();

  if (operations.size() == 4) {
    if (body.multiply.getLhs() != body.diagonalLoad.getResult() ||
        body.multiply.getRhs() != scale)
      return CarryBody();
    body.kind = CarryBodyKind::Malformed;
    return body;
  }

  if (body.multiply.getLhs() != body.carryLoad.getResult() ||
      body.multiply.getRhs() != body.diagonalLoad.getResult())
    return CarryBody();
  body.kind = CarryBodyKind::Correct;
  return body;
}

static bool hasExactlyOneScaleConstant(func::FuncOp function,
                                       arith::ConstantOp &scale) {
  SmallVector<arith::ConstantOp> constants;
  function.walk([&](arith::ConstantOp constant) {
    if (isScaleConstant(constant))
      constants.push_back(constant);
  });
  if (constants.size() != 1)
    return false;
  scale = constants.front();
  return true;
}

static bool findUniqueLuFunction(ModuleOp module, func::FuncOp &function) {
  SmallVector<func::FuncOp> matches;
  module.walk([&](func::FuncOp candidate) {
    if (candidate.getName() == "lu_func" ||
        candidate.getName() == "_Z7lu_funciPA100_KiPS_PA100_iS4_S4_PiS5_S4_S5_")
      matches.push_back(candidate);
  });
  if (matches.size() != 1)
    return false;
  function = matches.front();
  return true;
}

static void setAuditAttrs(ModuleOp module, StringRef sourceFile,
                          bool sourceVerified, StringRef inputStatus,
                          StringRef repairStatus, StringRef reason) {
  MLIRContext *context = module.getContext();
  module->setAttr(kSourceAttribute, StringAttr::get(context, sourceFile));
  module->setAttr(kSourceVerifiedAttribute,
                  BoolAttr::get(context, sourceVerified));
  module->setAttr(kInputStatusAttribute,
                  StringAttr::get(context, inputStatus));
  module->setAttr(kRepairStatusAttribute,
                  StringAttr::get(context, repairStatus));
  module->setAttr(kPassAttribute, StringAttr::get(context, kPassVersion));
  module->setAttr(kReasonAttribute, StringAttr::get(context, reason));
}

class RepairLuAffineDeterminantCarryPass
    : public PassWrapper<RepairLuAffineDeterminantCarryPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      RepairLuAffineDeterminantCarryPass)

  RepairLuAffineDeterminantCarryPass() = default;
  explicit RepairLuAffineDeterminantCarryPass(StringRef sourcePath) {
    sourceFile = sourcePath.str();
  }
  RepairLuAffineDeterminantCarryPass(
      const RepairLuAffineDeterminantCarryPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final { return kPassName; }
  StringRef getDescription() const final {
    return "Repair the source-anchored LU affine determinant carry";
  }

  Option<std::string> sourceFile{
      *this, "source-file",
      llvm::cl::desc("Original frozen LU C++ source file (required)"),
      llvm::cl::init("")};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto fail = [&](StringRef inputStatus, StringRef reason) {
      setAuditAttrs(module, sourceFile.getValue(), /*sourceVerified=*/false,
                    inputStatus, "failed", reason);
      module.emitError() << reason;
      signalPassFailure();
    };

    if (sourceFile.getValue().empty()) {
      fail("rejected", "--source-file is required");
      return;
    }

    auto sourceBuffer = llvm::MemoryBuffer::getFile(sourceFile.getValue());
    if (!sourceBuffer) {
      fail("rejected", "cannot read --source-file");
      return;
    }
    SourceContractResult source =
        verifySourceContract((*sourceBuffer)->getBuffer());
    if (!source.verified) {
      fail("rejected", source.reason);
      return;
    }

    auto failVerified = [&](StringRef reason) {
      setAuditAttrs(module, sourceFile.getValue(), /*sourceVerified=*/true,
                    "rejected", "failed", reason);
      module.emitError() << reason;
      signalPassFailure();
    };

    func::FuncOp function;
    if (!findUniqueLuFunction(module, function)) {
      failVerified("expected exactly one function whose symbol contains lu_func");
      return;
    }
    if (!isExactFunctionContract(function)) {
      failVerified("LU function type/layout does not match the frozen contract");
      return;
    }

    arith::ConstantOp scale;
    if (!hasExactlyOneScaleConstant(function, scale)) {
      failVerified("expected exactly one i32 constant with value FIXED_SCALE=1024");
      return;
    }

    Value upper = function.getArgument(5);
    Value determinant = function.getArgument(9);
    SmallVector<affine::AffineForOp> candidateLoops;
    function.walk([&](affine::AffineForOp loop) {
      bool hasDeterminantStore = false;
      for (Operation &operation : loop.getBody()->getOperations()) {
        if (auto store = dyn_cast<affine::AffineStoreOp>(&operation)) {
          if (store.getMemRef() == determinant)
            hasDeterminantStore = true;
        }
      }
      if (hasDeterminantStore)
        candidateLoops.push_back(loop);
    });
    if (candidateLoops.size() != 1) {
      failVerified("expected exactly one determinant-carry affine loop");
      return;
    }
    affine::AffineForOp loop = candidateLoops.front();
    if (loop->getBlock() != &function.getBody().front() ||
        !isExactLoopBound(loop, function)) {
      failVerified("determinant loop must be a top-level 0-to-matrix_size loop");
      return;
    }

    // The init store is part of the recurrence contract and must precede the
    // loop in the function's top-level block.  No other direct determinant
    // stores are allowed.
    SmallVector<affine::AffineStoreOp> determinantStores;
    SmallVector<affine::AffineLoadOp> determinantLoads;
    for (OpOperand &use : determinant.getUses()) {
      Operation *owner = use.getOwner();
      if (auto store = dyn_cast<affine::AffineStoreOp>(owner)) {
        determinantStores.push_back(store);
      } else if (auto load = dyn_cast<affine::AffineLoadOp>(owner)) {
        determinantLoads.push_back(load);
      } else {
        failVerified("determinant buffer has an unsupported alias or effect");
        return;
      }
    }

    affine::AffineStoreOp initStore;
    for (affine::AffineStoreOp store : determinantStores) {
      if (store->getBlock() == &function.getBody().front() &&
          store->isBeforeInBlock(loop.getOperation())) {
        if (initStore) {
          failVerified("multiple determinant initialization stores are ambiguous");
          return;
        }
        initStore = store;
      }
    }
    if (!initStore || !isZeroStore(initStore, determinant) ||
        initStore.getValue() != scale) {
      failVerified("determinant initialization must be one FIXED_SCALE scalar store");
      return;
    }
    if (determinantStores.size() != 2) {
      failVerified("determinant must have exactly one init and one loop store");
      return;
    }

    for (affine::AffineLoadOp load : determinantLoads) {
      if (!isDirectlyInLoopBody(load, loop)) {
        failVerified("determinant load is outside the unique carry loop");
        return;
      }
    }
    for (affine::AffineStoreOp store : determinantStores) {
      if (store != initStore && !isDirectlyInLoopBody(store, loop)) {
        failVerified("determinant store is outside the init or carry loop");
        return;
      }
    }

    CarryBody body = inspectCarryBody(loop, upper, determinant, scale);
    if (body.kind == CarryBodyKind::Invalid) {
      failVerified("determinant loop body is not the exact malformed or repaired recurrence");
      return;
    }

    if (body.kind == CarryBodyKind::Correct) {
      setAuditAttrs(module, sourceFile.getValue(), /*sourceVerified=*/true,
                    "already-correct", "unchanged", "input recurrence already correct");
      return;
    }

    // Repair only the malformed mul.  Inserting the load immediately before
    // it preserves the complete surrounding task source and leaves the
    // existing divide/store SSA chain intact.
    OpBuilder builder(body.multiply.getOperation());
    AffineMap zeroMap = AffineMap::get(
        /*dimCount=*/0, /*symbolCount=*/0,
        {builder.getAffineConstantExpr(0)}, builder.getContext());
    affine::AffineLoadOp carryLoad = builder.create<affine::AffineLoadOp>(
        body.multiply.getLoc(), determinant, zeroMap, ValueRange{});
    body.multiply->setOperand(0, carryLoad.getResult());
    body.multiply->setOperand(1, body.diagonalLoad.getResult());

    CarryBody repaired = inspectCarryBody(loop, upper, determinant, scale);
    if (repaired.kind != CarryBodyKind::Correct) {
      setAuditAttrs(module, sourceFile.getValue(), /*sourceVerified=*/true,
                    "malformed-determinant-carry", "failed",
                    "repair did not produce the exact carried determinant recurrence");
      module.emitError()
          << "repair did not produce the exact carried determinant recurrence";
      signalPassFailure();
      return;
    }
    setAuditAttrs(module, sourceFile.getValue(), /*sourceVerified=*/true,
                  "malformed-determinant-carry", "repaired",
                  "inserted determinant carry load and rewired product");
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createRepairLuAffineDeterminantCarryPass() {
  return std::make_unique<RepairLuAffineDeterminantCarryPass>();
}

std::unique_ptr<Pass>
mlir::createRepairLuAffineDeterminantCarryPass(StringRef sourceFile) {
  return std::make_unique<RepairLuAffineDeterminantCarryPass>(sourceFile);
}
