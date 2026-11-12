#ifndef ORBIT_TASKFLOW_GENERIC_CONSTRAINT_SEARCH_H
#define ORBIT_TASKFLOW_GENERIC_CONSTRAINT_SEARCH_H
#include "Taskflow/GenericReuse.h"
#include <optional>
namespace orbit::taskflow {
struct GenericConstraintSearchResult {
  std::optional<GenericReuseProposal> proposal;
  llvm::json::Object report;
};
GenericConstraintSearchResult
searchGenericConstraints(const GenericAnalysis &, const llvm::json::Value &,
                         GenericReuseOptions, int64_t wallMs,
                         uint64_t maxCostProviderCalls);
} // namespace orbit::taskflow
#endif
