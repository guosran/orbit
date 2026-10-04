//===- RetimingOriginalAmoebaFixedDecisionsPass.h ------------*- C++ -*-===//
#ifndef AMOEBA_RETIMING_ORIGINAL_AMOEBA_FIXED_DECISIONS_PASS_H
#define AMOEBA_RETIMING_ORIGINAL_AMOEBA_FIXED_DECISIONS_PASS_H

#include <memory>

namespace mlir {
class Pass;
namespace amoeba::neura {

// Retimes the captured decisions in an original throughput-guided AMOEBA
// scheduler trace. The pass does not search for a replacement schedule.
std::unique_ptr<Pass> createRetimingOriginalAmoebaFixedDecisionsPass();

} // namespace amoeba::neura
} // namespace mlir

#endif // AMOEBA_RETIMING_ORIGINAL_AMOEBA_FIXED_DECISIONS_PASS_H
