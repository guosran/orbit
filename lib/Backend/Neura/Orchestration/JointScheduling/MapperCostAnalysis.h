//===- MapperCostAnalysis.h ------------------------------------*- C++ -*-===//
// Shared analytical bounds and structural-startup helpers for mapper costs.
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_MAPPER_COST_ANALYSIS_H
#define AMOEBA_MAPPER_COST_ANALYSIS_H

#include "mlir/IR/Region.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <string>

namespace mlir::amoeba::neura::joint_scheduling {

// Keep these formulas shared by the standard task-cost predictor and the
// original-AMOEBA profile adapter so a corrected body uses the same lower
// bound and startup protocol in both paths.
bool computeMapperAnalyticalFacts(mlir::Region &region, int64_t rows,
                                  int64_t columns, double &recMII,
                                  double &resMII, double &lowerBound,
                                  std::string &error);

bool deriveStartupCyclesFromCppFeatureOutput(const llvm::json::Object &entry,
                                             double &startup,
                                             std::string &error);

} // namespace mlir::amoeba::neura::joint_scheduling

#endif // AMOEBA_MAPPER_COST_ANALYSIS_H
