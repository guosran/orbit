#ifndef ORBIT_TASKFLOW_AIR_BACKEND_H
#define ORBIT_TASKFLOW_AIR_BACKEND_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>
namespace mlir {
class Pass;
}
namespace orbit::taskflow {
std::unique_ptr<mlir::Pass> createTaskflowToAIRPass();
void registerTaskflowToAIRPass();
struct GenericAnalysis;
struct GenericMappingPlan;
bool checkGenericAIRMapping(const GenericAnalysis &, const GenericMappingPlan &,
                            mlir::ModuleOp, std::string &diagnostic);
// Legalize the installed standard DMA-to-channel result using explicit checked
// physical-port assignments. The input must use generic operation syntax.
bool legalizeGenericAIRChannels(const GenericAnalysis &, const GenericMappingPlan &,
                               mlir::ModuleOp, std::string &diagnostic);
bool checkGenericAIRChannels(const GenericAnalysis &, const GenericMappingPlan &,
                            mlir::ModuleOp, std::string &diagnostic);
// Separate backend: constructs AIR operations without linking AIR into Core.
// The installed AIR verifier/compiler validates the serialized dialect
// boundary.
// Serial tiled schedules append temporary-storage host BO arguments in analysis
// storage order; original input/output argument positions are preserved. Checked
// stages have independent local pools and rectangular DMA; compact serial tiles
// reuse storage within each stage. The public retain-across-users choice is
// currently realized only by proved readonly panels inside one execution
// group; it does not preserve SRAM contents across hardware stages. Read-only union scopes and loop traversal
// consume the public mapping choices; actual DMA operands bind to checked
// version/copy events and allocations. Strict f32 lane-vector multiplication
// is explicitly unavailable on the validated installed Peano target.
bool lowerGenericToAIR(const GenericAnalysis &analysis,
                       const GenericMappingPlan &plan,
                       mlir::OwningOpRef<mlir::ModuleOp> &output,
                       std::string &diagnostic, bool approximateExp = false);
} // namespace orbit::taskflow
#endif
