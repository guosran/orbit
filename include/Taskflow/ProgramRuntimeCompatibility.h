#ifndef ORBIT_TASKFLOW_PROGRAM_RUNTIME_COMPATIBILITY_H
#define ORBIT_TASKFLOW_PROGRAM_RUNTIME_COMPATIBILITY_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <string>

namespace orbit::taskflow {
// Hash the exact bytes at each parsing boundary, rather than a JSON proxy or
// a later pretty-printed module. The two texts are retained by the caller.
inline std::string programRuntimeTextSHA256(llvm::StringRef text) {
  llvm::SHA256 digest;
  digest.update(text);
  return llvm::toHex(digest.final(), true);
}

// Installed AIR emits a newer arith.trunci property absent from pinned MLIR.
// Parse generic syntax with no registered Arith dialect, remove only the exact
// semantic default, then let the caller's registered parser/checker validate.
// Non-default or unknown trunci properties must never be silently discarded.
inline bool normalizeProgramRuntimeDefaults(llvm::StringRef input,
                                            std::string &normalized,
                                            size_t &defaultsRemoved,
                                            std::string &error) {
  normalized.clear();
  defaultsRemoved = 0;
  error.clear();
  mlir::MLIRContext compatibilityContext;
  compatibilityContext.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      input, mlir::ParserConfig(&compatibilityContext, false));
  if (!module) {
    error = "invalid generic runtime IR before compatibility normalization";
    return false;
  }
  size_t accepted = 0;
  auto walked = module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() != "arith.trunci")
      return mlir::WalkResult::advance();
    if (operation->getDiscardableAttr("overflowFlags")) {
      error = "unsupported trunci overflowFlags attribute encoding";
      return mlir::WalkResult::interrupt();
    }
    auto properties = operation->getPropertiesAsAttribute();
    if (!properties)
      return mlir::WalkResult::advance();
    auto dictionary = mlir::dyn_cast<mlir::DictionaryAttr>(properties);
    auto overflow = dictionary ? mlir::dyn_cast_or_null<mlir::OpaqueAttr>(
                                     dictionary.get("overflowFlags"))
                               : nullptr;
    if (!dictionary || dictionary.size() != 1 || !overflow ||
        overflow.getDialectNamespace() != "arith" ||
        overflow.getAttrData() != "overflow<none>") {
      error = "unsupported trunci properties: only exact default "
              "overflowFlags = #arith.overflow<none> is compatible";
      return mlir::WalkResult::interrupt();
    }
    // Clearing the unregistered property removes its wrapper entirely. All
    // operands, result types, ordinary attributes and every other op survive.
    if (mlir::failed(operation->setPropertiesFromAttribute({}, [&]() {
          return operation->emitError("cannot clear default property");
        }))) {
      error = "cannot clear known default trunci property";
      return mlir::WalkResult::interrupt();
    }
    ++accepted;
    return mlir::WalkResult::advance();
  });
  if (walked.wasInterrupted())
    return false;
  if (!accepted) {
    normalized = input.str();
  } else {
    llvm::raw_string_ostream stream(normalized);
    module->print(stream, mlir::OpPrintingFlags().printGenericOpForm());
    stream << "\n";
  }
  defaultsRemoved = accepted;
  return true;
}
} // namespace orbit::taskflow
#endif
