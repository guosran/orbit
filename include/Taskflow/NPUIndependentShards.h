#ifndef ORBIT_TASKFLOW_NPU_INDEPENDENT_SHARDS_H
#define ORBIT_TASKFLOW_NPU_INDEPENDENT_SHARDS_H
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <string>
namespace orbit::taskflow {
struct NPUShardBinding {int64_t rowOffset=0,rowCount=0,column=0,row=2;};
bool mergeIndependentAIRShards(llvm::ArrayRef<mlir::ModuleOp>,llvm::ArrayRef<NPUShardBinding>,llvm::ArrayRef<mlir::Type>,int64_t fullRows,mlir::OwningOpRef<mlir::ModuleOp> &,std::string &);
// A checked independent outer-axis partition. Each shard remains one execution
// group with its own static pools. The manifest binds the source and real AIR;
// the installed AIR compiler and final AIE checker remain mandatory.
bool lowerNPUIndependentProgram(mlir::ModuleOp,const llvm::json::Value &,int64_t shards,mlir::OwningOpRef<mlir::ModuleOp> &,llvm::json::Value &,std::string &);
bool checkNPUIndependentAIE(mlir::ModuleOp,const llvm::json::Value &,int64_t shards,mlir::ModuleOp actual,llvm::json::Value &,std::string &);
}
#endif
