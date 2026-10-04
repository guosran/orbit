// NeuraBackendPasses.td - Passes owned by the Neura backend

#ifndef AMOEBA_BACKEND_NEURA_PASSES_TD
#define AMOEBA_BACKEND_NEURA_PASSES_TD

include "mlir/Pass/PassBase.td"

def BindStaticScalarArgument
    : Pass<"bind-static-scalar-argument", "ModuleOp"> {
  let summary = "Bind a verified compile-time scalar function argument";
  let description = [{
    Replaces uses of one integer or index function argument with a typed
    arith.constant while preserving the function signature and recording the
    selected static value on the function.  The caller is responsible for
    proving that every invocation supplies this value.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Exact function symbol; inferred if unique.">,
    Option<"argumentIndex", "argument", "int64_t", /*default=*/"-1",
           "Zero-based scalar argument index.">,
    Option<"scalarValue", "value", "int64_t", /*default=*/"0",
           "Compile-time signed scalar value.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createBindStaticScalarArgumentPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect"];
}

def ImportInput0CallerNoAlias
    : Pass<"import-input0-caller-noalias", "ModuleOp"> {
  let summary = "Import noalias only from a verified input-0 caller allocation graph";
  let description = [{
    Checks every in-module call from the selected input-0 entry point, or,
    with prepared-external-caller, checks every call in the exact
    lower-generated caller evidence module while leaving the prepared target
    body untouched.  Each ranked memref operand must trace through
    identity-preserving casts or full static subviews to a distinct static or
    source-proven constant-sized allocation with the exact source logical
    shape.  A target is annotated only after every callsite passes;
    model metadata, C/C++ signatures, and unknown or external roots do not
    establish this contract.
  }];
  let options = [
    Option<"targetFunction", "target", "std::string", /*default=*/"\"\"",
           "Target func.func symbol.">,
    Option<"callerFunction", "caller", "std::string", /*default=*/"\"main\"",
           "Closed-world input-0 caller symbol.">,
    Option<"callerEvidence", "caller-evidence", "std::string",
           /*default=*/"\"\"", "Exact caller evidence path (no digest).">,
    Option<"logicalShapes", "logical-shapes", "std::string",
           /*default=*/"\"\"",
           "Semicolon-separated x-separated memref shapes.">,
    Option<"evidenceOutput", "evidence-output", "std::string",
           /*default=*/"\"\"", "Proof JSON output path.">,
    Option<"staticBound", "static-bound", "int64_t", /*default=*/"-1",
           "Source-proven scalar loop bound.">,
    Option<"input0Only", "input0-only", "bool", /*default=*/"false",
           "Required explicit input-0 scope marker.">,
    Option<"preparedExternalCaller", "prepared-external-caller", "bool",
           /*default=*/"false",
           "Use an external lower-generated caller to prove a prepared target.">,
    Option<"preparedInput", "prepared-input", "std::string",
           /*default=*/"\"\"",
           "Exact prepared module path used for the current input.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createImportInput0CallerNoAliasPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def ConvertTaskflowToNeura : Pass<"convert-taskflow-to-neura", "ModuleOp"> {
  let summary = "Convert Taskflow operations to neura.kernel";
  let description = [{
    Converts taskflow.hyperblock operations and their counters into
    neura.kernel operations suitable for CGRA tile-array mapping.
  }];
  let constructor =
      "::mlir::amoeba::neura::createConvertTaskflowToNeuraPass()";
  let dependentDialects = ["mlir::taskflow::TaskflowDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::affine::AffineDialect"];
}

def ConstructHyperblockFromTask
    : Pass<"construct-hyperblock-from-task", "func::FuncOp"> {
  let summary = "Construct Taskflow hyperblocks for the Neura backend";
  let description = [{
    Selects one of two transforms based on the Neura architecture.
    With counter support, the pass detects perfect nested loop bands.
    Without counter support, it wraps the innermost loop as a hyperblock.
  }];
  let constructor =
      "::mlir::amoeba::neura::createConstructHyperblockFromTaskPass()";
}

def BindSourceIterationDomain
    : Pass<"bind-source-iteration-domain", "ModuleOp"> {
  let summary = "Bind a captured source iteration domain to its current Neura control body";
  let description = [{
    Consumes the one-shot capture marker emitted by construct-hyperblock-from-task
    after source affine-loop bounds, carried state, and full expansion have been
    proved. The pass binds the certificate to the current canonical Neura kernel
    bodies. It never creates or refreshes a missing binding on imported IR.
    Incomplete source domains are retained as unsupported facts.
  }];
  let constructor =
      "::mlir::amoeba::neura::createBindSourceIterationDomainPass()";
  let dependentDialects = ["mlir::taskflow::TaskflowDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::arith::ArithDialect"];
}

def VerifySourceIterationDomainPartitions
    : Pass<"verify-source-iteration-domain-partitions", "ModuleOp"> {
  let summary = "Re-prove an imported task graph against immutable source domains";
  let constructor = "::mlir::amoeba::neura::createVerifySourceIterationDomainPartitionsPass()";
}

def ClassifyTaskAndCounter : Pass<"classify-task-and-counter", "ModuleOp"> {
  let summary = "Classify tasks and counters for Neura runtime management";
  let description = [{
    Annotates taskflow.task and taskflow.counter operations with the
    structural and runtime-management information consumed by Neura.
  }];
  let constructor =
      "::mlir::amoeba::neura::createClassifyTaskAndCounterPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def OrchestrateTasksOnAccelerators
    : Pass<"orchestrate-tasks-on-accelerators", "func::FuncOp"> {
  let summary = "Orchestrate Taskflow tasks on a Neura multi-CGRA grid";
  let description = [{
    Computes task resource allocation, placement, and scheduling for the
    Neura multi-CGRA architecture. Selects an orchestration strategy and
    invokes the shared scheduler. The routing-critical-path strategy is the
    default. The analytical-based-task-orchestration strategy consumes a
    materialized task candidate and preserves its spatial shape orientation.
  }];
  let options = [Option<
      "schedulingMode", "scheduling-mode", "std::string",
      /*default=*/"\"spatial-temporal\"",
      "Task scheduling mode: 'spatial' (one task per CGRA) or "
      "'spatial-temporal' (time-multiplexes CGRAs).">,
    Option<
      "orchestrationStrategy", "orchestration-strategy", "std::string",
      /*default=*/"\"routing-critical-path\"",
      "Task orchestration strategy: 'routing-critical-path' (default) or "
      "'analytical-based-task-orchestration' (requires a materialized "
      "joint-scheduling candidate).">,
    Option<
      "communicationMode", "communication-mode", "std::string",
      /*default=*/"\"none\"",
      "Task communication mode: 'none' or strict architecture-level "
      "'explicit'.">];
  let constructor =
      "::mlir::amoeba::neura::createOrchestrateTasksOnAcceleratorsPass()";
}

def AnalyzeRecResMii : Pass<"analyze-rec-res-mii", "ModuleOp"> {
  let summary = "Compute mapper-compatible RecMII and ResMII without mapping";
  let description = [{
    Computes the recurrence and resource bounds for exactly one standalone
    task DFG. x-tiles and y-tiles select the concrete mapper rectangle. The
    pinned ensemble predictor reads amoeba.rec_mii and amoeba.res_mii as
    module attributes. The pass never places or routes an operation.
  }];
  let options = [Option<"xTiles", "x-tiles", "int64_t", /*default=*/"0",
                         "Concrete mapper tile columns; must be positive.">,
                 Option<"yTiles", "y-tiles", "int64_t", /*default=*/"0",
                         "Concrete mapper tile rows; must be positive.">];
  let constructor =
      "::mlir::amoeba::neura::createAnalyzeRecResMiiPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::neura::NeuraDialect"];
}

def EnumerateAnalyticalTaskCandidates
    : Pass<"enumerate-analytical-task-candidates", "ModuleOp"> {
  let summary = "Freeze the full static task-shape Cartesian product";
  let description = [{
    Enumerates every ordered assignment of fixed-orientation rectangular
    per-task CGRA shapes that individually fit the physical grid, then writes
    a JSONL manifest. The production spatial-temporal scheduler determines
    whole-program feasibility and may reuse resources over time. Constant counter chains
    record their exact trip count, while dynamic or unresolved bounds are
    rejected. Counter bounds are read directly from their SSA definitions, so
    this pass does not depend on `classify-task-and-counter`. The pass does not
    score candidates or invoke the mapper. Fusion, fission, tiling, placement,
    and temporal order are fixed.
  }];
  let options =
      [Option<
           "functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function to process; inferred when exactly one exists.">,
       Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
              "Path to the candidate JSONL manifest.">,
       Option<
           "maxCandidates", "max-candidates", "int64_t",
           /*default=*/"1000000",
           "Fail instead of publishing a partial manifest above this count.">,
       Option<"maxCgrasPerTask", "max-cgras-per-task", "int64_t",
              /*default=*/"0",
              "Largest rectangular physical footprint per task; zero uses the full physical grid.">,
       Option<"searchPolicy", "search-policy", "std::string",
              /*default=*/"""",
              "Shape policy: complete-cartesian or bounded-changes-from-unit.">,
       Option<"shapePruningPolicy", "shape-pruning-policy", "std::string",
              /*default=*/"""",
              "Alias for search-policy for the shape-pruning contract.">,
       Option<"maxChangedTasks", "max-changed-tasks", "int64_t",
              /*default=*/"-1",
              "Maximum number of non-unit task shapes in bounded search.">,
       Option<"graphVariantId", "graph-variant-id", "std::string",
              /*default=*/"\"\"", "Optional graph variant identity.">,
       Option<"graphVariantSha256", "graph-variant-sha256", "std::string",
              /*default=*/"\"\"", "SHA-256 of graph-variant result IR bytes.">];
  let constructor =
      "::mlir::amoeba::neura::createEnumerateAnalyticalTaskCandidatesPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def EnumerateExactTaskSchedules
    : Pass<"enumerate-exact-task-schedules", "ModuleOp"> {
  let summary = "Enumerate legal event-normalized fixed-shape 4x4 schedules";
  let description = [{
    Branches over dependency-ready task orders and every in-bounds rectangle
    origin. Explicit mode replays production network reservations and branches
    at communication-ready and resource-release events, scoring each path.
    A branch with simultaneous cell overlap is rejected immediately. The
    all-integer policy covers every start cycle through the specified makespan
    horizon for one fixed graph and shape assignment. Event-normalized mode
    remains diagnostic; global graph/shape coverage and SRAM remain separate.
    Budget exhaustion yields an incomplete inventory.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function; inferred when unique.">,
    Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
           "Schedule inventory JSONL path.">,
    Option<"maxExpandedNodes", "max-expanded-nodes", "int64_t",
           /*default=*/"100000", "Branch budget.">,
    Option<"maxMilliseconds", "max-milliseconds", "int64_t",
           /*default=*/"0", "Optional wall time budget.">,
    Option<"communicationMode", "communication-mode", "std::string",
           /*default=*/"\"none\"",
           "none or production explicit inter-task network.">,
    Option<"startPolicy", "start-policy", "std::string",
           /*default=*/"\"event-normalized\"",
           "event-normalized or all-integer.">,
    Option<"maxMakespan", "max-makespan", "int64_t",
           /*default=*/"0", "All-integer inclusive makespan horizon.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createEnumerateExactTaskSchedulesPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def MaterializeAnalyticalTaskCandidate
    : Pass<"materialize-analytical-task-candidate", "ModuleOp"> {
  let summary = "Materialize one static task-shape candidate";
  let description = [{
    Validates a complete candidate manifest, selects one candidate by its
    sequential ID, and writes its cgra_count and cgra_shape decisions to
    Taskflow tasks. The analytical-based task orchestration strategy consumes
    and preserves the selected rectangle orientation. This pass does not
    invoke the mapper. Counter annotations must match the enumeration pipeline
    that produced the manifest.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function to process; inferred when exactly one exists.">,
    Option<"candidateFile", "candidates", "std::string", /*default=*/"\"\"",
           "Path to the candidate JSONL manifest.">,
    Option<"candidateId", "candidate-id", "std::string", /*default=*/"\"\"",
           "Sequential candidate ID to materialize.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createMaterializeAnalyticalTaskCandidatePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def MaterializeJointTaskReplicas
    : Pass<"materialize-joint-task-replicas", "ModuleOp"> {
  let summary = "Shard a proven terminal elementwise task into complete DFG replicas";
  let description = [{
    Clones a proved static terminal elementwise task into two independent
    execution shards along the selected counter axis after Taskflow-to-Neura conversion.
    Each clone retains the entire mapper-visible body. A completion join proves
    disjoint full output coverage; the operation does not create semantic tiles.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function; inferred when exactly one exists.">,
    Option<"taskName", "task", "std::string", /*default=*/"\"consumer\"",
           "Terminal elementwise task to shard.">,
    Option<"replicaCount", "replicas", "int64_t", /*default=*/"2",
           "Number of complete DFG instances; this proof supports two.">,
    Option<"shardAxis", "axis", "int64_t", /*default=*/"0",
           "Static output/counter axis split between complete DFG instances.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createMaterializeJointTaskReplicasPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def EnumerateReplicaActions
    : Pass<"enumerate-replica-actions", "ModuleOp"> {
  let summary = "Inventory each Taskflow task's legal, illegal, and unknown execution-shard replica factors";
  let description = [{
    Factor one is an identity action. Each factor 2, 4, and 8 on each output
    axis is admitted only after the source-owned replica materializer succeeds
    on a cloned whole module and the result verifies. This inventory is not
    joint graph closure or a completed ablation candidate search.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Function containing the complete Taskflow program.">,
    Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
           "Atomic JSONL replica action inventory output.">,
    Option<"maxActionRecords", "max-action-records", "int64_t",
           /*default=*/"100000", "Resource bound; a reached bound is incomplete.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createEnumerateReplicaActionsPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def MapJointSchedulingTasks
    : Pass<"map-joint-scheduling-tasks", "ModuleOp"> {
  let summary = "Map every selected task on its fixed oriented rectangle";
  let description = [{
    Runs InsertDataMov and the real Neura mapper independently for every
    materialized task. The selected per-task mapper rows and columns configure
    each invocation. Mapping failure, missing kernels, or a mismatched result
    fails closed. Successfully mapped operations are moved back into the task.
    The opt-in all-unit-baseline mode accepts only shape-space candidate-0
    with a complete source-owned iteration-domain certificate and an identity
    1x1 shape for every task. It derives structural startup from the C++ mapper
    feature graph and combines it with actual mapped II without reading an ML
    score file.
  }];
  let constructor =
      "::mlir::amoeba::neura::createMapJointSchedulingTasksPass()";
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Unique Taskflow function selected by the manifest.">,
    Option<"scoreFile", "scores", "std::string", /*default=*/"\"\"",
           "Validated score JSONL used to bind startup cycles.">,
    Option<"candidateId", "candidate-id", "std::string",
           /*default=*/"\"\"", "Selected candidate ID.">,
    Option<"allUnitBaseline", "all-unit-baseline", "bool",
           /*default=*/"false",
           "Use source-certified all-1x1 mapper durations without ML scores.">,
    Option<"expectedTraceSha256", "expected-trace-sha256", "std::string",
           /*default=*/"\"\"", "Expected selected schedule trace identity.">,
    Option<"expectedScoreSha256", "expected-score-sha256", "std::string",
           /*default=*/"\"\"", "SHA-256 of the exact score JSONL bytes.">
  ];
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def ScoreAnalyticalTaskCandidates
    : Pass<"score-analytical-task-candidates", "ModuleOp"> {
  let summary = "Score every candidate in a frozen spatial-binding manifest";
  let description = [{
    Validates and scores every candidate from a complete manifest using a
    provenance-bound external ML II/startup cost catalogue. Repeated queries
    share a stable task-body, architecture, and oriented-shape cache. Only
    after complete manifest traversal is a deterministic top-k shortlist
    emitted. The ranking score is the makespan produced by the production
    fixed-orientation spatial-temporal scheduler using predicted task
    durations. Communication remains outside this pass. A symbol-dynamic trip
    count must be concretely bound before ranking.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function to process; inferred when exactly one exists.">,
    Option<"candidateFile", "candidates", "std::string", /*default=*/"\"\"",
           "Path to the frozen candidate JSONL manifest.">,
    Option<"costFile", "cost-file", "std::string", /*default=*/"\"\"",
           "Path to the task-shape ML cost catalogue.">,
    Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
           "Path to the score JSONL manifest.">,
    Option<"topK", "top-k", "int64_t", /*default=*/"1",
           "Candidates to shortlist; zero selects every valid candidate.">,
    Option<"communicationMode", "communication-mode", "std::string",
           /*default=*/"\"none\"",
           "Task communication mode: 'none' or strict 'explicit'.">,
    Option<"graphVariantId", "graph-variant-id", "std::string",
           /*default=*/"\"\"", "Optional graph variant identity.">,
    Option<"graphVariantSha256", "graph-variant-sha256", "std::string",
           /*default=*/"\"\"", "SHA-256 of graph-variant result IR bytes.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createScoreAnalyticalTaskCandidatesPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def SearchJointNeighborhood : Pass<"search-joint-neighborhood", "ModuleOp"> {
  let summary = "Budgeted cost-guided complete-program neighborhood search";
  let options = [
    Option<"scoringWorkers", "scoring-workers", "int64_t",
           /*default=*/"1",
           "Number of private production-scheduler workers; one preserves serial semantics.">,
    Option<"maxPartitionFactor", "max-partition-factor", "int64_t",
           /*default=*/"4",
           "Independent authenticated cumulative tiling/replica cap: 4 or 8.">,
    Option<"historicalNativeWinner", "historical-native-winner", "std::string",
           /*default=*/"\"\"",
           "Optional source-owned historical native control selection JSONL.">
  ];
  let constructor = "::mlir::amoeba::neura::createSearchJointNeighborhoodPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def ReplayJointNeighborhoodActions
    : Pass<"replay-joint-neighborhood-actions", "ModuleOp"> {
  let summary = "Replay a typed neighborhood action path with source-domain proofs";
  let description = [{
    Replays only the supplied exact action sequence through the existing C++
    neighborhood enumerator and source-owned materializers. Every predecessor
    and resulting candidate is checked against the immutable source iteration
    domain. It emits the complete replayed module and source facts, without
    reading a cost catalogue, predicting, mapping, or ranking candidates.
  }];
  let options = [
    Option<"actionFile", "action-file", "std::string", /*default=*/"\"\"",
           "Exact typed action JSON file.">,
    Option<"canonicalInput", "canonical-input", "std::string", /*default=*/"\"\"",
           "Exact immutable source-certified canonical module path.">,
    Option<"candidateInput", "candidate-input", "std::string", /*default=*/"\"\"",
           "Exact current predecessor module path.">,
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function to process.">,
    Option<"stage", "stage", "std::string", /*default=*/"\"full-joint\"",
           "Cumulative neighborhood stage enabled for supplied actions.">,
    Option<"maxPartitionFactor", "max-partition-factor", "int64_t",
           /*default=*/"8", "Authenticated cumulative partition cap: 1, 2, 4, or 8.">,
    Option<"outputDir", "output-dir", "std::string", /*default=*/"\"\"",
           "New directory for the complete candidate, facts, and exact inputs.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createReplayJointNeighborhoodActionsPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def ProveStaticActiveTransferShapes
    : Pass<"prove-static-active-transfer-shapes", "ModuleOp"> {
  let summary = "Prove opt-in active transfer boxes without changing physical memory";
  let constructor =
      "::mlir::amoeba::neura::joint_scheduling::createProveStaticActiveTransferShapesPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::taskflow::TaskflowDialect",
                           "mlir::neura::NeuraDialect"];
}

def PrepareJointDiagnosticCandidate
    : Pass<"prepare-joint-diagnostic-candidate", "ModuleOp"> {
  let summary = "Emit one source-bound explicit joint scheduling diagnostic candidate";
  let constructor =
      "::mlir::amoeba::neura::createPrepareJointDiagnosticCandidatePass()";
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function to process; inferred only when explicitly supplied.">,
    Option<"parentCostFile", "parent-cost-file", "std::string", /*default=*/"\"\"",
           "Provenance-bound parent task-shape ML cost catalogue.">,
    Option<"sourceRepository", "source-repository", "std::string", /*default=*/"\"\"",
           "Source repository bound by the parent catalogue.">,
    Option<"sourceCommit", "source-commit", "std::string", /*default=*/"\"\"",
           "Source commit bound by the parent catalogue.">,
    Option<"architecturePath", "architecture-path", "std::string", /*default=*/"\"\"",
           "Architecture specification bound by the parent catalogue.">,
    Option<"candidateId", "candidate-id", "std::string", /*default=*/"\"\"",
           "Explicit diagnostic candidate identity.">,
    Option<"outputDir", "output-dir", "std::string", /*default=*/"\"\"",
           "Directory for the candidate, selection, and score artifacts.">,
    Option<"dispatchPolicy", "dispatch-policy", "std::string",
           /*default=*/"\"fixed\"",
           "Production dispatch policy: fixed or critical-path.">,
    Option<"taskShapes", "task-shapes", "std::string", /*default=*/"\"\"",
           "Optional semicolon-separated TaskName:RxC overrides; unspecified tasks use 1x1.">
  ];
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}


def ScoreExactJointTaskCandidates
    : Pass<"score-exact-joint-task-candidates", "ModuleOp"> {
  let summary = "Rank the exact factored shape and explicit communication space";
  let description = [{
    Traverses every shape tuple in an exact factored task-shape descriptor. For
    each tuple, the source-owned ML predicted II/startup contract supplies task
    latencies, and the production explicit communication model enumerates every
    legal four-by-four location, either the canonical fixed dispatch order or
    every dependency-ready dispatch order, and integer temporal start through a
    caller-supplied horizon. Immediate occupancy and
    communication conflicts are pruned by the C++ exact schedule core. A
    max-shape, node, time, or unproven horizon bound leaves the result
    incomplete and uncertified.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function; inferred when exactly one exists.">,
    Option<"candidateFile", "candidates", "std::string", /*default=*/"\"\"",
           "Exact factored task-shape descriptor.">,
    Option<"costFile", "cost-file", "std::string", /*default=*/"\"\"",
           "Provenance-bound ML II/startup cost catalogue.">,
    Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
           "Global exact-joint score JSONL.">,
    Option<"topK", "top-k", "int64_t", /*default=*/"5",
           "Global schedules to retain after complete traversal.">,
    Option<"maxShapeCandidates", "max-shape-candidates", "int64_t",
           /*default=*/"0", "Zero is unlimited; exhaustion is incomplete.">,
    Option<"maxExpandedNodes", "max-expanded-nodes", "int64_t",
           /*default=*/"1000000", "Global exact schedule branch budget.">,
    Option<"maxMilliseconds", "max-milliseconds", "int64_t",
           /*default=*/"0", "Global wall-time budget; zero is unlimited.">,
    Option<"maxMakespan", "max-makespan", "int64_t", /*default=*/"0",
           "Inclusive all-integer temporal horizon.">,
    Option<"dispatchPolicy", "dispatch-policy", "std::string",
           /*default=*/"\"all-ready\"",
           "Dispatch order: fixed canonical smallest-ready task or all-ready.">,
    Option<"horizonProven", "horizon-proven", "bool", /*default=*/"false",
           "Only true for a separately proven finite horizon.">,
    Option<"graphVariantId", "graph-variant-id", "std::string",
           /*default=*/"\"\"", "Semantic graph variant identity.">,
    Option<"sourceRepository", "source-git-repository", "std::string",
           /*default=*/"\"\"", "Cost catalogue source repository.">,
    Option<"sourceCommit", "source-git-commit", "std::string",
           /*default=*/"\"\"", "Cost catalogue source commit.">,
    Option<"scoreCacheDirectory", "score-cache-dir", "std::string",
           /*default=*/"\"\"",
           "Optional complete exact-score cache directory; incomplete results "
           "are never reused.">,
    Option<"scoreCachePeerDirectory", "score-cache-peer-directory", "std::string",
           /*default=*/"\"\"", "Optional legacy peer score directory.">,
    Option<"scoreCachePeerProbe", "score-cache-peer-probe", "bool",
           /*default=*/"false", "Probe/import a peer score without searching.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createScoreExactJointTaskCandidatesPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def FuseTask : Pass<"fuse-task", "func::FuncOp"> {
  let summary = "Fuse Taskflow tasks using Neura profitability metrics";
  let description = [{
    Applies producer-consumer and sibling fusion strategies. Profitability is
    evaluated with Neura lowering and MII metrics.
  }];
  let constructor = "::mlir::amoeba::neura::createFuseTaskPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect",
                           "mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def ExtractJointTaskGraphFacts
    : Pass<"extract-joint-task-graph-facts", "ModuleOp"> {
  let summary = "Extract Taskflow graph facts and legal rewrite actions";
  let description = [{
    Writes deterministic task facts, dependencies, shape needs, a canonical
    structural key, and conservative fusion, fission, and tensor-axis tiling
    actions. Graph enumeration reruns this pass after every rewrite.
  }];
  let constructor =
      "::mlir::amoeba::neura::createExtractJointTaskGraphFactsPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def EnumerateJointSemanticRewrite
    : Pass<"enumerate-joint-semantic-rewrites", "ModuleOp"> {
  let summary = "Enumerate source-bound Taskflow tiling and fusion graph rewrites";
  let description = [{
    Recognizes a static producer-consumer Taskflow program and publishes a
    deterministic, hardware-independent graph rewrite DAG. Legalization and
    candidate identity are owned by this MLIR pass.
  }];
  let constructor =
      "::mlir::amoeba::neura::createEnumerateJointSemanticRewritePass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def EnumerateApplicationRewriteActions
    : Pass<"enumerate-application-rewrite-actions", "ModuleOp"> {
  let summary = "Record complete Taskflow application rewrite action alphabet";
  let description = [{
    Emits a fail-closed inventory of no-op and M/N/K tiling and sibling or
    producer-consumer fusion action attempts. Unknown materialization is never
    promoted to a legal application rewrite.
  }];
  let constructor =
      "::mlir::amoeba::neura::createEnumerateApplicationRewriteActionsPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def EnumerateJointGraphClosure
    : Pass<"enumerate-joint-graph-closure", "ModuleOp"> {
  let summary = "Enumerate bounded C++ MLIR joint graph rewrite paths";
  let description = [{
    Clones the current ModuleOp, invokes source-owned materializers for M/N/K
    tiling, execution-shard replicas, and producer-consumer or sibling fusion,
    and re-extracts structural graph facts after each successful path.
    The graph rewrite scope is single actions and the two paths fuse-then-tile
    and tile-then-fuse. Unsupported actions and resource limits remain
    incomplete.
  }];
  let options = [
    Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
           "JSONL closure output." >,
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function selected by the closure." >,
    Option<"stageName", "stage", "std::string",
           /*default=*/"\"full-joint\"",
           "One of the five cumulative stage names." >,
    Option<"maxGraphs", "max-graphs", "int64_t", /*default=*/"1024",
           "Maximum unique structural graph facts." >,
    Option<"maxActions", "max-actions", "int64_t", /*default=*/"10000",
           "Maximum source-owned materializer attempts." >
  ];
  let constructor =
      "::mlir::amoeba::neura::createEnumerateJointGraphClosurePass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def PruneJointGraphCandidates
    : Pass<"prune-joint-graph-candidates", "ModuleOp"> {
  let summary = "Validate and prune completed graph enumeration before joint search";
  let description = [{
    Reads the complete closure artifact after enumeration has ended, checks
    persisted graph artifacts and dependencies, and optionally reuses the
    exact scorer's cached-cost graph bound. Unknown proofs retain the graph;
    an incomplete upstream closure remains incomplete. This pass does not
    alter the rewrite frontier or claim top-five ranking completeness.
  }];
  let constructor =
      "::mlir::amoeba::neura::createPruneJointGraphCandidatesPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def MaterializeJointSemanticRewrite
    : Pass<"materialize-joint-semantic-rewrite", "ModuleOp"> {
  let summary = "Materialize a selected source-bound Taskflow graph rewrite";
  let description = [{
    Applies a selected hardware-independent M/N/K task tiling and graph
    fusion candidate to Taskflow IR after source and legality validation.
  }];
  let constructor =
      "::mlir::amoeba::neura::createMaterializeJointSemanticRewritePass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def TileOutputM : Pass<"tile-output-m", "ModuleOp"> {
  let summary = "Tile a source-bound Taskflow output along M";
  let constructor = "::mlir::amoeba::neura::createTileOutputMPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def TileOutputN : Pass<"tile-output-n", "ModuleOp"> {
  let summary = "Tile a source-bound Taskflow output along N";
  let constructor = "::mlir::amoeba::neura::createTileOutputNPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def TileReductionKSequential
    : Pass<"tile-reduction-k-sequential", "ModuleOp"> {
  let summary = "Split K into sequential Taskflow state-carry blocks";
  let constructor =
      "::mlir::amoeba::neura::createTileReductionKSequentialPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def TileReductionKParallelLinear
    : Pass<"tile-reduction-k-parallel-linear", "ModuleOp"> {
  let summary = "Split K into private partials with linear reduction topology";
  let constructor =
      "::mlir::amoeba::neura::createTileReductionKParallelLinearPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def TileReductionKParallelTree
    : Pass<"tile-reduction-k-parallel-tree", "ModuleOp"> {
  let summary = "Split K into private partials with balanced-tree topology";
  let constructor =
      "::mlir::amoeba::neura::createTileReductionKParallelTreePass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def FuseProducerConsumer
    : Pass<"fuse-producer-consumer", "ModuleOp"> {
  let summary = "Fuse a proven Taskflow producer-consumer pair";
  let constructor =
      "::mlir::amoeba::neura::createFuseProducerConsumerPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def FuseTileLocalPair : Pass<"fuse-tile-local-pair", "ModuleOp"> {
  let summary = "Fuse corresponding producer-consumer output tiles";
  let constructor = "::mlir::amoeba::neura::createFuseTileLocalPairPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def FuseReductionConsumer
    : Pass<"fuse-reduction-consumer", "ModuleOp"> {
  let summary = "Fuse a completed numeric reduction with its consumer";
  let constructor =
      "::mlir::amoeba::neura::createFuseReductionConsumerPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def MaterializeNumericReduction
    : Pass<"materialize-numeric-reduction", "ModuleOp"> {
  let summary = "Materialize a numeric reduction over private K partials";
  let constructor =
      "::mlir::amoeba::neura::createMaterializeNumericReductionPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def MaterializeCompletionJoin
    : Pass<"materialize-completion-join", "ModuleOp"> {
  let summary = "Join disjoint Taskflow completion states without arithmetic";
  let constructor =
      "::mlir::amoeba::neura::createMaterializeCompletionJoinPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def MaterializeNeuraJointRewrite
    : Pass<"materialize-neura-joint-rewrite", "ModuleOp"> {
  let summary = "Materialize static M/N tiling after Taskflow-to-Neura lowering";
  let description = [{
    Applies the verified post-conversion vertical slice of the ORBIT joint
    rewrite pipeline.  A selected Taskflow task must contain one static
    neura.kernel and a matching outer/inner counter chain.  The pass clones
    complete task/kernel instances, narrows the selected M/N counter in both
    IR layers, and replaces the original completion state with a checked
    taskflow.join.  Producer-consumer fusion is available in retained mode
    and in a separately selectable forwarded mode that removes exactly one
    proven intermediate store/load pair.  K reduction is handled by the separate post-Neura K reduction pass; no
    Python rewrite is involved.
  }];
  let options = [
    Option<"taskName", "task-name", "std::string", /*default=*/"\"\"",
           "Exact Taskflow task name to rewrite." >,
    Option<"tileAxis", "tile-axis", "int64_t", /*default=*/"-1",
           "Static M/N counter axis (0 or 1)." >,
    Option<"tileFactor", "tile-factor", "int64_t", /*default=*/"0",
           "Number of disjoint static domain parts." >,
    Option<"fusionMode", "fusion-mode", "std::string",
           /*default=*/"\"none\"",
           "none, producer-consumer-retained, or producer-consumer-forwarded." >,
    Option<"firstTaskName", "first-task-name", "std::string",
           /*default=*/"\"\"",
           "Producer Taskflow task name for fusion." >,
    Option<"secondTaskName", "second-task-name", "std::string",
           /*default=*/"\"\"",
           "Consumer Taskflow task name for fusion." >
  ];
  let constructor =
      "::mlir::amoeba::neura::createMaterializeNeuraJointRewritePass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def MaterializeNeuraKReduction
    : Pass<"materialize-neura-k-reduction", "ModuleOp"> {
  let summary = "Materialize canonical static i32 GEMM K reduction after Neura lowering";
  let description = [{
    Consumes the mixed Taskflow plus Neura IR after
    --convert-taskflow-to-neura.  The pass accepts only a statically bounded
    2-D i32 GEMM producer with one scf.for K accumulator and emits either an
    explicit state-carry chain or private partials with a linear or balanced
    numeric reduction.  Completion joins are never used as arithmetic.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function name." >,
    Option<"kMode", "k-mode", "std::string", /*default=*/"\"sequential\"",
           "sequential, parallel-linear, or parallel-tree." >,
    Option<"tileSize", "tile-size", "int64_t", /*default=*/"0",
           "K block width; zero derives a default or uses k-factor." >,
    Option<"kFactor", "k-factor", "int64_t", /*default=*/"0",
           "Number of K blocks when tile-size is zero." >,
    Option<"taskName", "task-name", "std::string", /*default=*/"\"\"",
           "Select the checked LLaMA Task_1 application recurrence." >
  ];
  let constructor =
      "::mlir::amoeba::neura::createMaterializeNeuraKReductionPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def LowerJointTaskflowToHostSCF
    : Pass<"lower-joint-taskflow-to-host-scf", "ModuleOp"> {
  let summary = "Lower executable ORBIT Taskflow graphs to host SCF";
  let description = [{
    Inlines dependency-ordered Taskflow bodies, lowers static or dynamic
    counters to nested scf.for operations, and aliases completion joins only
    after the Taskflow verifier proves their region coverage. Unsupported
    bodies and operations are rejected without publishing a partial lowering.
  }];
  let options = [Option<
      "functionName", "function", "std::string", /*default=*/"\"\"",
      "Only lower this function; empty lowers every Taskflow function.">];
  let constructor =
      "::mlir::amoeba::neura::createLowerJointTaskflowToHostSCFPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def BindExactMemrefShape : Pass<"bind-exact-memref-shape", "ModuleOp"> {
  let summary = "Bind an exact caller memref shape to one function argument";
  let description = [{
    Attaches amoeba.logical_transfer_shape to the selected func.func
    argument after validating its ranked memref type, rank, and static
    dimensions. The pass never publishes a partial contract.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Exact func.func symbol name to annotate.">,
    Option<"argumentNumber", "argument", "int64_t", /*default=*/"-1",
           "Zero-based function argument index to annotate.">,
    Option<"dimensions", "dimensions", "std::string", /*default=*/"\"\"",
           "Comma-separated positive caller extents, e.g. 8,100.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createBindExactMemrefShapePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect"];
}

def FissionTask : Pass<"fission-task", "func::FuncOp"> {
  let summary = "Split one value-only task at a legal pure DFG cut";
  let constructor = "::mlir::amoeba::neura::createFissionTaskPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def FissionRayCarriedReduction
    : Pass<"fission-ray-carried-reduction", "func::FuncOp"> {
  let summary = "Split the exact Ray carried reduction across an affine lane interval";
  let description = [{
    Materializes a source-owned two-stage split of the Ray Task_13 reduction.
    It recognizes only the original scalar-seeded, two-result min/index loop,
    requires caller-proved disjoint inputs and outputs, and partitions the
    complete static lane range into prefix and suffix tasks connected by
    explicit scratch state. Unsupported bodies fail closed. Run after
    convert-affine-to-taskflow and before construct-hyperblock-from-task.
  }];
  let options = [
    Option<"taskName", "task-name", "std::string", /*default=*/"\"Task_13\"",
           "Exact source task name to split.">,
    Option<"splitAt", "split-at", "int64_t", /*default=*/"4",
           "Exclusive lane bound for the first reduction stage.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createFissionRayCarriedReductionPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def TileTask : Pass<"tile-task", "func::FuncOp"> {
  let summary = "Split a static tensor task into k parts along one axis";
  let constructor = "::mlir::amoeba::neura::createTileTaskPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def ResourceAwareTaskOptimization
    : Pass<"resource-aware-task-optimization", "func::FuncOp"> {
  let summary =
      "Optimize Taskflow resource allocation for the Neura CGRA backend";
  let description = [{
    Balances pipeline latency and fuses independent tasks for Neura CGRA
    utilization. Compiled estimation runs the Neura lowering and mapping
    flow; analytical estimation uses ResMII/RecMII estimates.
  }];
  let options = [Option<
      "estimationMode", "estimation-mode", "std::string",
      /*default=*/"\"compiled\"",
      "Profiling mode: 'compiled' runs Neura lowering and mapping; "
      "'analytical' uses ResMII/RecMII estimates.">];
  let constructor =
      "::mlir::amoeba::neura::createResourceAwareTaskOptimizationPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::func::FuncDialect"];
}

def PredictAnalyticalTaskCostCatalog
    : Pass<"predict-analytical-task-cost-catalog", "ModuleOp"> {
  let summary =
      "Predict an exploratory C++ formal max-four task-shape catalogue";
  let description = [{
    Extracts each current route-expanded Neura task body, computes the pinned
    156 mapper features in C++, evaluates the JSON MLP ensemble, and publishes
    a complete task-shape cost catalogue. An optional feature-file is accepted
    only as an exact current-body oracle. Cache identity is canonical body
    text plus mapper shape and exact architecture/model source text; trip count
    is never an ML feature or cache identity. Exploratory artifacts remain
    production_ready false and unsupported input fails closed.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function name.">,
    Option<"featureFile", "feature-file", "std::string", /*default=*/"\"\"",
           "Optional C++ feature oracle; must match the current module.">,
    Option<"featureOutputFile", "feature-output", "std::string",
           /*default=*/"\"\"", "Optional generated current-module features.">,
    Option<"spaceFile", "space-file", "std::string", /*default=*/"\"\"",
           "Exhaustive candidate-space header.">,
    Option<"ensembleFile", "ensemble-file", "std::string", /*default=*/"\"\"",
           "Formal JSON ensemble.">,
    Option<"checkpointDirectory", "checkpoint-dir", "std::string",
           /*default=*/"\"\"", "Formal JSON checkpoint directory.">,
    Option<"architectureContract", "architecture-contract", "std::string",
           /*default=*/"\"\"", "Neura architecture and shape contract.">,
    Option<"architecturePath", "architecture-path", "std::string",
           /*default=*/"\"\"", "Target architecture path.">,
    Option<"architectureTransferCatalog", "architecture-transfer-catalog",
           "std::string", /*default=*/"\"\"",
           "Optional validated rebound catalogue with architecture transfer evidence.">,
    Option<"sourceRepository", "source-git-repository", "std::string",
           /*default=*/"\"\"", "Source repository.">,
    Option<"sourceCommit", "source-git-commit", "std::string",
           /*default=*/"\"\"", "Source Git commit.">,
    Option<"graphVariantId", "graph-variant-id", "std::string",
           /*default=*/"\"\"", "Graph variant identity.">,
    Option<"modelNamespace", "model-namespace", "std::string",
           /*default=*/"\"formal-max4-nohash-v2-exploratory\"",
           "Output catalogue namespace.">,
    Option<"cacheFile", "cache", "std::string", /*default=*/"\"\"",
           "Persistent structural ML cache.">,
    Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
           "Output task-shape catalogue.">,
    Option<"allowUnsupportedAboveModelCeiling",
           "allow-unsupported-above-model-ceiling", "bool",
           /*default=*/"false",
           "Record source-proved lower bounds above the model interval as "
           "explicit unsupported task-shape rows.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createPredictAnalyticalTaskCostCatalogPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect",
                           "mlir::neura::NeuraDialect"];
}

#endif // AMOEBA_BACKEND_NEURA_PASSES_TD

def RebindAnalyticalTaskCostCatalog
    : Pass<"rebind-analytical-task-cost-catalog", "ModuleOp"> {
  let summary = "Rebind an ML II catalogue to an exact factored task space";
  let description = [{
    Validates exact task/oriented mapper-shape coverage and explicit source,
    architecture, graph, and model provenance before changing candidate_count.
    Predictor rows, exploratory status, and production_ready flags are copied
    losslessly. Missing predictions and ambiguous model provenance fail closed.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function to validate.">,
    Option<"costFile", "cost-file", "std::string", /*default=*/"\"\"",
           "Input predictor catalogue.">,
    Option<"spaceFile", "space-file", "std::string", /*default=*/"\"\"",
           "Exact factored task-shape JSONL descriptor.">,
    Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
           "Rebound catalogue output.">,
    Option<"sourceRepository", "source-git-repository", "std::string",
           /*default=*/"\"\"", "Expected source Git repository.">,
    Option<"sourceCommit", "source-git-commit", "std::string",
           /*default=*/"\"\"", "Expected source Git commit.">,
    Option<"architecturePath", "architecture-path", "std::string",
           /*default=*/"\"\"", "Expected architecture path.">,
    Option<"graphVariantId", "graph-variant-id", "std::string",
           /*default=*/"\"\"", "Expected graph variant identity.">,
    Option<"modelNamespace", "model-namespace", "std::string",
           /*default=*/"\"\"", "Expected predictor namespace.">,
    Option<"model", "model", "std::string", /*default=*/"\"\"",
           "Expected predictor model identity.">,
    Option<"modelSchema", "model-schema", "std::string",
           /*default=*/"\"\"", "Expected predictor model schema.">,
    Option<"featureContract", "feature-contract-id", "std::string",
           /*default=*/"\"\"", "Expected predictor feature contract.">,
    Option<"featureExtractor", "feature-extractor", "std::string",
           /*default=*/"\"\"", "Expected predictor feature extractor.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createRebindAnalyticalTaskCostCatalogPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def InheritReplicaAnalyticalTaskCostCatalog
    : Pass<"inherit-replica-analytical-task-cost-catalog", "ModuleOp"> {
  let summary =
      "Inherit ML II/startup costs after proving replica body equality";
  let description = [{
    Reuses a canonical task-shape ML catalogue for unchanged tasks and
    materialized replicas only after C++ proves mapper-visible Neura body
    equality. Counter bound changes are isolated to the ranker trip-count
    calculation; semantic body or identity changes fail closed.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Child taskflow function.">,
    Option<"parentCostFile", "parent-cost-file", "std::string",
           /*default=*/"\"\"", "Canonical parent ML catalogue.">,
    Option<"parentModuleFile", "parent-module", "std::string",
           /*default=*/"\"\"", "Canonical parent MLIR module.">,
    Option<"childSpaceFile", "child-space-file", "std::string",
           /*default=*/"\"\"", "Child factored task-shape manifest.">,
    Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
           "Inherited child catalogue.">,
    Option<"sourceRepository", "source-git-repository", "std::string",
           /*default=*/"\"\"", "Expected source Git repository.">,
    Option<"sourceCommit", "source-git-commit", "std::string",
           /*default=*/"\"\"", "Expected source Git commit.">,
    Option<"architecturePath", "architecture-path", "std::string",
           /*default=*/"\"\"", "Expected architecture path.">,
    Option<"parentGraphVariantId", "parent-graph-variant-id", "std::string",
           /*default=*/"\"\"", "Canonical parent graph identity.">,
    Option<"modelNamespace", "model-namespace", "std::string",
           /*default=*/"\"\"", "Expected predictor namespace.">,
    Option<"model", "model", "std::string", /*default=*/"\"\"",
           "Expected predictor model.">,
    Option<"modelSchema", "model-schema", "std::string",
           /*default=*/"\"\"", "Expected predictor model schema.">,
    Option<"featureContract", "feature-contract-id", "std::string",
           /*default=*/"\"\"", "Expected feature contract.">,
    Option<"featureExtractor", "feature-extractor", "std::string",
           /*default=*/"\"\"", "Expected feature extractor.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createInheritReplicaAnalyticalTaskCostCatalogPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::affine::AffineDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect",
                           "mlir::neura::NeuraDialect"];
}

def GlobalStageRanker : Pass<"select-global-stage-top-five", "ModuleOp"> {
  let summary = "Select one certified global top five across stage graph variants";
  let options = [
    Option<"manifestFile", "manifest", "std::string", /*default=*/"\"\"",
           "Manifest binding the C++ graph closure and exact score files.">,
    Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
           "Global top-five JSONL output.">,
    Option<"maxDiagnostics", "max-diagnostics", "int64_t", /*default=*/"5",
           "Maximum uncertified diagnostic selections for incomplete inputs.">
  ];
  let constructor = "::mlir::amoeba::neura::createGlobalStageRankerPass()";
}

// Original decisions are held fixed; this pass performs no orchestration search.
def OriginalAmoebaProfileCostAdapter : Pass<"adapt-original-amoeba-profile-costs", "ModuleOp"> {
  let summary = "Bind original AMOEBA mapper profiles to exact current kernel bodies";
  let constructor = "mlir::amoeba::neura::createOriginalAmoebaProfileCostAdapterPass()";
}

def RetimingOriginalAmoebaFixedDecisions : Pass<"retime-original-amoeba-fixed-decisions", "ModuleOp"> {
  let summary = "Retime captured original AMOEBA decisions with explicit communication";
  let constructor = "mlir::amoeba::neura::createRetimingOriginalAmoebaFixedDecisionsPass()";
}
