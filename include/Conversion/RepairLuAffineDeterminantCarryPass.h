#ifndef AMOEBA_REPAIR_LU_AFFINE_DETERMINANT_CARRY_PASS_H
#define AMOEBA_REPAIR_LU_AFFINE_DETERMINANT_CARRY_PASS_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace mlir {

/// Creates the source-anchored repair pass for the frozen LU determinant
/// recurrence.  The pass is intentionally opt-in; its source-file option is
/// required at run time and the normal backend must register it explicitly.
std::unique_ptr<Pass> createRepairLuAffineDeterminantCarryPass();

/// Test/integration convenience overload.  The registered pass should use the
/// no-argument factory and set --source-file through the pass pipeline.
std::unique_ptr<Pass>
createRepairLuAffineDeterminantCarryPass(llvm::StringRef sourceFile);

} // namespace mlir

#endif // AMOEBA_REPAIR_LU_AFFINE_DETERMINANT_CARRY_PASS_H
