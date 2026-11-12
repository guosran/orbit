#ifndef ORBIT_TASKFLOW_AMOEBA_MAPPING_H
#define ORBIT_TASKFLOW_AMOEBA_MAPPING_H

#include "Taskflow/ProgramMapping.h"

namespace orbit::taskflow {
// Architecture facts, not a physical SRAM or host-I/O timing model. Fabric
// dimensions count CGRA blocks; PE dimensions describe each block's mapper.
struct AmoebaTargetContract {
  int64_t fabricRows = 0, fabricColumns = 0;
  int64_t peRows = 0, peColumns = 0;
  // Zero means unspecified; a positive value limits statically resident
  // contexts per CGRA in the existing backend scheduler, not memory bytes.
  int64_t contextSlots = 0;
  std::string architectureIdentity, networkIdentity;
};
struct AmoebaTaskShape {
  int64_t rows = 1, columns = 1;
};

llvm::json::Value serializeAmoebaTarget(const AmoebaTargetContract &);
// Initial finite adapter: one actual source task per instance, exact whole
// storage accesses, one abstract data-availability endpoint per task. Shapes
// are choices for the real backend mapper/scheduler, not placed coordinates.
bool buildAmoebaProgramGraph(GenericAnalysis &, const AmoebaTargetContract &,
                             const std::vector<AmoebaTaskShape> &,
                             ProgramGraph &, std::string &error);
bool checkAmoebaProgramGraph(const GenericAnalysis &,
                             const AmoebaTargetContract &,
                             const std::vector<AmoebaTaskShape> &,
                             const ProgramGraph &, std::string &error);
} // namespace orbit::taskflow
#endif
