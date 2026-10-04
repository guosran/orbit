//===- SourceIterationDomainPartitionProof.h --------------------*- C++ -*-===//
//
// Trusted, mapper-free proof hook for refreshing source-domain body bindings
// after a complete replica/tiling group has been validated.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_BACKEND_NEURA_SOURCE_ITERATION_DOMAIN_PARTITION_PROOF_H
#define AMOEBA_BACKEND_NEURA_SOURCE_ITERATION_DOMAIN_PARTITION_PROOF_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace mlir::amoeba::neura::joint_scheduling {

// Verifies the full current child task graph against the complete canonical
// parent using the C++ replica/tiling body, counter, output, join, interval,
// and state-carry proofs. Only after all source groups pass does it attach
// inherited source certificates and refresh current body bindings. It does no
// prediction, cost lookup, file I/O, or scheduling.
LogicalResult proveAndRefreshSourceIterationDomainPartition(
    func::FuncOp canonicalParent, func::FuncOp currentChild,
    std::string &error);

// Read-only counterpart for imported, restored, or replayed candidates. It
// repeats the same parent-to-full-child group/body proof and requires the
// current certificate, source-origin witness, partition marker, and exact
// current body binding to match. It never adds or repairs attributes.
LogicalResult verifySourceIterationDomainPartition(
    func::FuncOp canonicalParent, func::FuncOp currentChild,
    std::string &error);

} // namespace mlir::amoeba::neura::joint_scheduling

#endif // AMOEBA_BACKEND_NEURA_SOURCE_ITERATION_DOMAIN_PARTITION_PROOF_H
