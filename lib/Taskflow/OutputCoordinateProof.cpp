#include "Taskflow/OutputCoordinateProof.h"
#include <atomic>
namespace mlir::taskflow {
static std::atomic<OutputCoordinateProofProvider> provider{nullptr};
void registerOutputCoordinateProofProvider(
    OutputCoordinateProofProvider value) {
  provider.store(value);
}
OutputCoordinateProof analyzeOutputCoordinates(TaskflowTaskOp task,
                                               int64_t axis, unsigned output) {
  if (auto callback = provider.load())
    return callback(task, axis, output);
  OutputCoordinateProof result;
  result.reason = "no output-coordinate proof provider registered; legacy "
                  "replica verification fails closed";
  return result;
}
} // namespace mlir::taskflow
