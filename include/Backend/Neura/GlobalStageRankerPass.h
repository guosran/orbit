//===- GlobalStageRankerPass.h -------------------------------------------===//

#ifndef AMOEBA_BACKEND_NEURA_GLOBAL_STAGE_RANKER_PASS_H
#define AMOEBA_BACKEND_NEURA_GLOBAL_STAGE_RANKER_PASS_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace amoeba {
namespace neura {

std::unique_ptr<Pass> createGlobalStageRankerPass();

} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_BACKEND_NEURA_GLOBAL_STAGE_RANKER_PASS_H
