#include "Taskflow/GenericMapping.h"
#include "Taskflow/NativeTransforms.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
using namespace orbit::taskflow;
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  GenericAnalysis a;
  GenericMappingPlan p;
  std::string error;
  llvm::json::Value config = llvm::json::Object{
      {"generic_air",
       llvm::json::Object{{"tile_sizes", llvm::json::Array{2, 3}}}}};
  if (!module || !analyzeGenericProgram(*module, a, error) ||
      !planGenericMapping(a, config, p, error)) {
    llvm::errs() << error << "\n";
    return 1;
  }
  if (p.strategy != "serial-tiled" || p.phases.empty() ||
      p.residentBytes > 24576)
    return 1;
  unsigned checks = 1;
  auto reject = [&](const char *name,
                    std::function<void(GenericMappingPlan &)> edit) {
    auto changed = p;
    changed.planId.clear();
    edit(changed);
    if (checkGenericMapping(a, changed, error)) {
      llvm::errs() << "accepted invalid tile plan: " << name << "\n";
      return false;
    }
    ++checks;
    return true;
  };
  bool ok = true;
  ok &= reject("missing phase", [](auto &x) { x.phases.pop_back(); });
  ok &= reject("duplicate phase",
               [](auto &x) { x.phases.push_back(x.phases.front()); });
  ok &= reject("missing transfer", [](auto &x) { x.transfers.pop_back(); });
  ok &= reject("transfer bytes", [](auto &x) { ++x.transfers.front().bytes; });
  ok &=
      reject("phase association", [](auto &x) { ++x.transfers.front().phase; });
  ok &= reject("local peak", [](auto &x) { --x.residentBytes; });
  ok &= reject("invalid phase task",
               [&](auto &x) { x.phases.front().task = a.tasks.size(); });
  ok &= reject("invalid transfer storage", [&](auto &x) {
    x.transfers.front().storage = a.storages.size();
  });
  ok &= reject("region offset", [](auto &x) {
    ++x.phases.front().regions.front().offsets.front();
  });
  ok &= reject("region extent", [](auto &x) {
    ++x.phases.front().regions.front().sizes.front();
  });
  ok &= reject("required read", [](auto &x) {
    x.phases.front().regions.front().read =
        !x.phases.front().regions.front().read;
  });
  ok &= reject("loop coverage",
               [](auto &x) { ++x.phases.front().loops.front().lower; });
  if (p.phases.size() > 1) {
    ok &= reject("missing completion dependency",
                 [](auto &x) { x.phases[1].dependencies.clear(); });
    ok &= reject("early reuse dependency",
                 [](auto &x) { x.phases[1].dependencies = {1}; });
  }
  ok &= reject("capacity bypass", [](auto &x) { x.capacityBytes = 24577; });
  ok &= reject("invalid tile size", [](auto &x) { x.tileSizes.front() = 0; });
  ok &= reject("missing hardware stage",
               [](auto &x) { x.hardwareStages.pop_back(); });
  ok &= reject("hardware stage source coverage",
               [](auto &x) { ++x.hardwareStages.front().beginPhase; });
  if (p.hardwareStages.size() > 1)
    ok &= reject("hardware stage mixes tail rate shapes", [](auto &x) {
      x.hardwareStages.front().endPhase = x.hardwareStages.back().endPhase;
      x.hardwareStages.resize(1);
    });
  auto reduction = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  GenericAnalysis reductionAnalysis;
  GenericMappingPlan reductionPlan;
  llvm::json::Value reductionConfig = llvm::json::Object{
      {"generic_air",
       llvm::json::Object{{"tile_sizes", llvm::json::Array{2, 2, 3}}}}};
  if (!reduction || !distributeNativeInitialization(*reduction, error) ||
      !analyzeGenericProgram(*reduction, reductionAnalysis, error) ||
      !planGenericMapping(reductionAnalysis, reductionConfig, reductionPlan,
                          error)) {
    llvm::errs() << "source reduction retention fixture failed: " << error
                 << "\n";
    ok = false;
  } else if (!reductionPlan.retainedRegions.empty()) {
    ++checks;
    if (reductionPlan.channelAllocations.empty() ||
        !llvm::any_of(reductionPlan.hardwareStages, [](const auto &stage) {
          return stage.legalizeChannels;
        })) {
      llvm::errs()
          << "K tail did not receive required explicit DMA channel plan\n";
      ok = false;
    } else
      ++checks;
    bool scopeStages =
        llvm::all_of(reductionPlan.hardwareStages, [&](const auto &stage) {
          return stage.legalizeChannels &&
                 llvm::any_of(reductionPlan.retainedRegions,
                              [&](const auto &scope) {
                                return scope.beginPhase == stage.beginPhase &&
                                       scope.endPhase == stage.endPhase;
                              });
        });
    if (!scopeStages || reductionPlan.hardwareStages.size() !=
                            reductionPlan.retainedRegions.size()) {
      llvm::errs() << "conflicting K templates still aggregate output scopes\n";
      ok = false;
    } else
      ++checks;
    auto original = reductionPlan;
    auto rejectRetention =
        [&](const char *name,
            std::function<void(GenericMappingPlan &)> mutate) {
          auto changed = original;
          changed.planId.clear();
          mutate(changed);
          if (checkGenericMapping(reductionAnalysis, changed, error)) {
            llvm::errs() << "accepted invalid retained plan: " << name << "\n";
            ok = false;
          } else
            ++checks;
        };
    rejectRetention("channel conflict aggregates output scopes", [](auto &x) {
      x.hardwareStages[0].endPhase = x.hardwareStages[1].endPhase;
      x.hardwareStages.erase(x.hardwareStages.begin() + 1);
      x.channelAllocations.erase(
          std::remove_if(x.channelAllocations.begin(),
                         x.channelAllocations.end(),
                         [](const auto &entry) { return entry.stage == 1; }),
          x.channelAllocations.end());
      for (auto &entry : x.channelAllocations)
        if (entry.stage > 1)
          --entry.stage;
    });
    rejectRetention("unplanned logical FIFO merge policy",
                    [](auto &x) { x.channelMergePolicy = "merge-everything"; });
    rejectRetention("wrong logical FIFO identity", [](auto &x) {
      x.channelAllocations.front().logicalFifo += "_different";
    });
    rejectRetention("wrong FIFO transfer count", [](auto &x) {
      ++x.channelAllocations.front().transferCount;
    });
    rejectRetention("wrong FIFO byte accounting",
                    [](auto &x) { ++x.channelAllocations.front().bytes; });
    rejectRetention("missing DMA channel allocation",
                    [](auto &x) { x.channelAllocations.pop_back(); });
    rejectRetention("duplicate DMA channel allocation", [](auto &x) {
      x.channelAllocations.push_back(x.channelAllocations.front());
    });
    rejectRetention("tile DMA channel outside hardware ports", [](auto &x) {
      x.channelAllocations.front().tileChannel = 2;
    });
    rejectRetention("wrong channel transfer direction", [](auto &x) {
      x.channelAllocations.front().direction = "l1_to_host";
    });
    rejectRetention("hidden required channel legalization", [](auto &x) {
      x.hardwareStages.front().legalizeChannels = false;
    });
    rejectRetention("channel storage identity",
                    [](auto &x) { ++x.channelAllocations.front().storage; });
    rejectRetention("missing invariant scope",
                    [](auto &x) { x.retainedRegions.clear(); });
    rejectRetention("scope beyond source", [](auto &x) {
      x.retainedRegions.front().endPhase = x.phases.size() + 1;
    });
    rejectRetention("scope changed output offsets",
                    [](auto &x) { ++x.retainedRegions.front().endPhase; });
    rejectRetention("overlapping lifetimes", [](auto &x) {
      x.retainedRegions.push_back(x.retainedRegions.front());
    });
    rejectRetention("unaccounted import staging",
                    [](auto &x) { x.residentBytes -= 4; });
    rejectRetention("missing final export",
                    [](auto &x) { x.transfers.pop_back(); });
    if (!original.fusedInitializerTasks.empty()) {
      rejectRetention("missing fused source coverage",
                      [](auto &x) { x.fusedInitializerTasks.clear(); });
      rejectRetention("wrong fused initializer source", [](auto &x) {
        x.retainedRegions.front().initializerTask = x.phases.back().task;
      });
      rejectRetention("hardware stage splits retained output", [](auto &x) {
        const auto scope = x.retainedRegions.front();
        auto stage = llvm::find_if(x.hardwareStages, [&](const auto &stage) {
          return stage.beginPhase <= scope.beginPhase &&
                 stage.endPhase >= scope.endPhase;
        });
        stage->endPhase = scope.beginPhase + 1;
      });
      rejectRetention("duplicate fused source", [](auto &x) {
        x.fusedInitializerTasks.push_back(x.fusedInitializerTasks.front());
      });
      rejectRetention(
          "missing initializer with fabricated BO import", [&](auto &x) {
            auto &scope = x.retainedRegions.front();
            scope.initializerTask = -1;
            const auto &phase = x.phases[scope.beginPhase];
            auto region = llvm::find_if(phase.regions, [&](const auto &r) {
              return r.storage == scope.storage;
            });
            auto type = mlir::cast<mlir::MemRefType>(
                reductionAnalysis.storages[scope.storage].root.getType());
            int64_t bytes =
                (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
            for (auto size : region->sizes)
              bytes *= size;
            x.transfers.push_back({scope.storage, "host_to_l1", bytes,
                                   scope.beginPhase, region->offsets,
                                   region->sizes});
            std::sort(x.transfers.begin(), x.transfers.end(),
                      [](const auto &a, const auto &b) {
                        return std::tie(a.phase, a.storage, a.direction) <
                               std::tie(b.phase, b.storage, b.direction);
                      });
          });
      rejectRetention("reinitialize in middle of accumulation", [](auto &x) {
        auto scope = x.retainedRegions.front();
        x.retainedRegions.front().endPhase = scope.beginPhase + 1;
        scope.beginPhase += 1;
        x.retainedRegions.insert(x.retainedRegions.begin() + 1, scope);
      });
    }
  }
  const char *largeSource = R"mlir(
module {
  func.func @large(%out: memref<129x129xf32> {orbit.role = "output"}) attributes {orbit.noalias} {
    %done = taskflow.task @fill will_writes(%out : memref<129x129xf32>) [original_write_memrefs(%out : memref<129x129xf32>)] : (memref<129x129xf32>) -> memref<129x129xf32> {
    ^bb0(%o: memref<129x129xf32>):
      %zero = arith.constant 0.0 : f32
      affine.for %i = 0 to 129 {
        affine.for %j = 0 to 129 {
          affine.store %zero, %o[%i, %j] : memref<129x129xf32>
        }
      }
      taskflow.yield done_writes(%o : memref<129x129xf32>)
    }
    return
  }
}
)mlir";
  auto large = mlir::parseSourceString<mlir::ModuleOp>(largeSource, &context);
  GenericAnalysis largeAnalysis;
  GenericMappingPlan largePlan;
  llvm::json::Value largeConfig = llvm::json::Object{
      {"generic_air",
       llvm::json::Object{{"tile_sizes", llvm::json::Array{1, 1}}}}};
  if (!large || !analyzeGenericProgram(*large, largeAnalysis, error) ||
      !planGenericMapping(largeAnalysis, largeConfig, largePlan, error)) {
    llvm::errs() << "large phase plan failed: " << error << "\n";
    ok = false;
  } else {
    auto report = serializeGenericMapping(largeAnalysis, largePlan);
    auto *object = report.getAsObject();
    if (largePlan.phases.size() != 16641 || largePlan.residentBytes != 4 ||
        object->getBoolean("expanded_phase_details_omitted") != true ||
        !object->getArray("phases")->empty() ||
        object->getArray("phase_domains")->size() != 1 ||
        object->getInteger("system_memory_peak_bytes") != 129 * 129 * 4 ||
        object->getArray("host_buffers")->size() != 1 ||
        object->getInteger("backend_tensor_padding_bytes") != 0 ||
        object->getArray("hardware_stages")
                ->front()
                .getAsObject()
                ->getArray("buffer_pools")
                ->front()
                .getAsObject()
                ->getInteger("allocated_bytes") != 4 ||
        !object->getString("phase_schedule_digest")) {
      llvm::errs() << "large compact report lost domain/count/digest\n";
      ok = false;
    } else
      ++checks;
    largePlan.phases.back().dependencies.clear();
    if (checkGenericMapping(largeAnalysis, largePlan, error)) {
      llvm::errs() << "large plan accepted missing completion dependency\n";
      ok = false;
    } else
      ++checks;
  }
  llvm::outs() << "{\"passed\":" << (ok ? "true" : "false")
               << ",\"checks\":" << checks << "}\n";
  return ok ? 0 : 1;
}
