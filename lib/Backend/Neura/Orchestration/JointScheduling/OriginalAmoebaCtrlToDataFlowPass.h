//===- OriginalAmoebaCtrlToDataFlowPass.h ----------------------*- C++ -*-===//

#ifndef AMOEBA_ORIGINAL_AMOEBA_CTRL_TO_DATA_FLOW_PASS_H
#define AMOEBA_ORIGINAL_AMOEBA_CTRL_TO_DATA_FLOW_PASS_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::amoeba::neura::joint_scheduling {

// Reproduces the pinned original TaskProfiler control-to-data lowering.
std::unique_ptr<mlir::Pass> createOriginalAmoebaCtrlToDataFlowPass();

} // namespace mlir::amoeba::neura::joint_scheduling

#endif // AMOEBA_ORIGINAL_AMOEBA_CTRL_TO_DATA_FLOW_PASS_H
