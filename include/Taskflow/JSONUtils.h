#ifndef ORBIT_TASKFLOW_JSON_UTILS_H
#define ORBIT_TASKFLOW_JSON_UTILS_H

#include "llvm/Support/JSON.h"
#include <initializer_list>

namespace orbit::taskflow {
// Rebuild instead of erasing fields and copying a json::Object. In the pinned
// LLVM, ObjectKey's copy assignment retains Owned when assigning a borrowed
// DenseMap tombstone. Copying that map resurrects the erased key and copies its
// already destroyed Value (parse -> erase -> copy is a standalone reproducer).
// Iteration skips tombstones; recursively rebuilding also sanitizes nested
// objects and owns borrowed strings. Never change the frozen LLVM dependency.
inline llvm::json::Value
copyJSONWithoutFields(const llvm::json::Value &source,
                      std::initializer_list<llvm::StringRef> omitted = {}) {
  if (const auto *object = source.getAsObject()) {
    llvm::json::Object result;
    for (const auto &entry : *object) {
      bool skip = false;
      for (auto key : omitted)
        skip |= llvm::StringRef(entry.first) == key;
      if (!skip)
        result[entry.first.str()] = copyJSONWithoutFields(entry.second);
    }
    return result;
  }
  if (const auto *array = source.getAsArray()) {
    llvm::json::Array result;
    for (const auto &entry : *array)
      result.push_back(copyJSONWithoutFields(entry));
    return result;
  }
  if (auto string = source.getAsString())
    return string->str();
  return llvm::json::Value(source);
}

inline llvm::json::Value
copyJSONWithoutNestedFields(const llvm::json::Value &source,
                            llvm::StringRef objectKey,
                            std::initializer_list<llvm::StringRef> omitted) {
  auto result = copyJSONWithoutFields(source);
  if (auto *root = result.getAsObject())
    if (auto *object = root->get(objectKey); object && object->getAsObject())
      *object = copyJSONWithoutFields(*object, omitted);
  return result;
}
} // namespace orbit::taskflow
#endif
