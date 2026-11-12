#include "Taskflow/JSONUtils.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace orbit::taskflow;
namespace j = llvm::json;
int main() {
  unsigned checks = 0;
  auto require = [&](bool condition, const char *label) {
    ++checks;
    if (!condition)
      llvm::errs() << label << "\n";
    return condition;
  };
  auto original = llvm::cantFail(j::parse(
      R"({"generic_air":{"tile_sizes":[16,16,16],"capacity_bytes":24576,"cost_model":{"kind":"test"},"capabilities":{"implementations":[{"kind":"scalar","lanes":1}]}},"program_search":{"seconds":600}})"));
  const auto before = llvm::formatv("{0}", original).str();
  auto filtered = copyJSONWithoutNestedFields(original, "generic_air",
                                              {"tile_sizes", "cost_model"});
  // This is the minimal parse -> remove -> copy failure mode, exercised with
  // the public replacement helper rather than relying on vendor UB to crash.
  auto copy = filtered;
  const auto *air = copy.getAsObject()->getObject("generic_air");
  if (!require(air && !air->get("tile_sizes") && !air->get("cost_model"),
               "removed fields must not resurrect after a JSON copy") ||
      !require(air->getInteger("capacity_bytes") == 24576 &&
                   air->getObject("capabilities") != nullptr,
               "unrelated native capabilities and integers remain intact") ||
      !require(llvm::formatv("{0}", original).str() == before,
               "deriving a planner config must preserve the caller config"))
    return 1;
  // Existing callers may hand us objects with tombstones. Iteration skips
  // erased entries; recursively rebuilding must sanitize nested objects too.
  auto dirty = original;
  dirty.getAsObject()->getObject("generic_air")->erase("tile_sizes");
  auto clean = copyJSONWithoutFields(dirty, {"program_search"});
  auto copiedClean = clean;
  if (!require(!copiedClean.getAsObject()->get("program_search") &&
                   !copiedClean.getAsObject()
                        ->getObject("generic_air")
                        ->get("tile_sizes"),
               "nested tombstones must not reappear in rebuilt configs"))
    return 1;
  j::Array nested;
  nested.push_back(std::move(dirty));
  auto arrayCopy = copyJSONWithoutFields(j::Value(std::move(nested)));
  auto copiedArray = arrayCopy;
  if (!require(!copiedArray.getAsArray()
                    ->front()
                    .getAsObject()
                    ->getObject("generic_air")
                    ->get("tile_sizes"),
               "arrays also rebuild their nested objects"))
    return 1;
  std::string borrowed = "temporary source text";
  auto owned = copyJSONWithoutFields(j::Value(llvm::StringRef(borrowed)));
  borrowed.assign(borrowed.size(), 'x');
  if (!require(owned.getAsString() == "temporary source text",
               "rebuilt strings own their bytes"))
    return 1;
  llvm::outs() << "json ownership checks: " << checks << "\n";
  return 0;
}
