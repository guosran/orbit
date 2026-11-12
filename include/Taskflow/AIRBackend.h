#ifndef ORBIT_TASKFLOW_AIR_BACKEND_H
#define ORBIT_TASKFLOW_AIR_BACKEND_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"
#include <memory>
#include <string>
namespace mlir {
class Pass;
class MLIRContext;
} // namespace mlir
namespace orbit::taskflow {
std::unique_ptr<mlir::Pass> createTaskflowToAIRPass();
void registerTaskflowToAIRPass();
// Call before parsing generic AIR containing symbol references. Loads a
// namespace-only facade when real AIR is not registered; installed AIR still
// owns AIR operation verification. It does not mutate source IR.
void prepareGenericAIRContext(mlir::MLIRContext &);
// Extract actual AIR or generic final-AIE MLIR structure. Binding attributes
// are separate witnesses, never substituted for operation/SSA facts.
llvm::json::Value extractGenericAIRImplementation(mlir::ModuleOp);
struct GenericAnalysis;
struct GenericMappingPlan;
bool checkGenericAIEImplementation(const GenericAnalysis &,
                                   const GenericMappingPlan &, mlir::ModuleOp,
                                   llvm::json::Value &receipt,
                                   std::string &diagnostic);
bool checkGenericAIRMapping(const GenericAnalysis &, const GenericMappingPlan &,
                            mlir::ModuleOp, std::string &diagnostic);
// Legalize the installed standard DMA-to-channel result using explicit checked
// physical-port assignments. The input must use generic operation syntax.
bool legalizeGenericAIRChannels(const GenericAnalysis &,
                                const GenericMappingPlan &, mlir::ModuleOp,
                                std::string &diagnostic);
bool checkGenericAIRChannels(const GenericAnalysis &,
                             const GenericMappingPlan &, mlir::ModuleOp,
                             std::string &diagnostic);
// Separate backend: consumes the checked common ProgramGraph nested in the
// native realization; source instances, allocation identities and copy events
// are authoritative. Native phases remain source-view/compression witnesses,
// not an independent transfer/allocation policy. The realization is explicitly
// bound to common execution groups and explicit copy/release dependencies.
// Source compute order is preserved; conservative timing reservations are
// separate from target runtime arbitration. This is not a general event
// scheduler. Constructs AIR operations without linking AIR into Core.
// The installed AIR verifier/compiler validates the serialized dialect
// boundary.
// Serial tiled schedules append only temporary storages having checked common
// system-memory allocations as host BO arguments in analysis storage order;
// original input/output argument positions are preserved. Checked execution
// groups have independent local pools and rectangular DMA. Their member events
// and spatial owner are authoritative; a multi-task block shares its checked
// reusable allocation without an intermediate BO. Its output DMA operations
// follow all block compute bodies, matching explicit release edges. Compact
// serial tiles reuse storage within each stage. The public retain-across-users
// choice is realized either by whole-program local copies in the existing
// single-herd resident path or by proved readonly panels inside one execution
// group. Explicit whole-program and tiled-phase-local choices cannot fall back
// to a different realization. Whole-program copies share one allocation across
// all consumers and never add a temporary host BO. Readonly panels do not
// preserve SRAM contents across hardware stages. Read-only union scopes and
// loop traversal consume the public mapping choices; actual DMA operands bind
// to checked version/copy events and allocations. Strict f32 lane-vector
// multiplication is explicitly unavailable on the validated installed Peano
// target.
bool lowerGenericToAIR(const GenericAnalysis &analysis,
                       const GenericMappingPlan &plan,
                       mlir::OwningOpRef<mlir::ModuleOp> &output,
                       std::string &diagnostic, bool approximateExp = false);
} // namespace orbit::taskflow
#endif
