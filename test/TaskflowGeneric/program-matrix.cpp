#include "Taskflow/GenericMapping.h"
#include "Taskflow/NativeTransforms.h"
#include "Taskflow/ProgramGraphAIR.h"
#include "Taskflow/ProgramGraphAIRLoop.h"
#include "Taskflow/ProgramGraphAIRSelection.h"
#include "Taskflow/ProgramMatrixImplementation.h"
#include "Taskflow/ProgramMatrixRuntime.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

using namespace orbit::taskflow;
int main(int argc, char **argv) {
  if (argc != 3 && argc != 4 && argc != 5) {
    llvm::errs()
        << "usage: program-matrix-test source.mlir NEW-output-directory\n";
    return 2;
  }
  int64_t tile = 16;
  bool selected =
      argc >= 4 && llvm::StringRef(argv[3]).starts_with("--selection-tile=");
  bool compact =
      argc >= 4 && llvm::StringRef(argv[3]).starts_with("--loop-tile=");
  bool runtimeInput = argc == 4 && !compact && !selected &&
                      !llvm::StringRef(argv[3]).starts_with("--tile=");
  if (argc >= 4 && !runtimeInput &&
      llvm::StringRef(argv[3])
          .drop_front(selected  ? 17
                      : compact ? 12
                                : 7)
          .getAsInteger(10, tile))
    return 2;
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  unsigned checks = 0;
  std::string error;
  auto require = [&](bool value, const char *label) {
    if (!value) {
      llvm::errs() << label << ": " << error << "\n";
      return false;
    }
    ++checks;
    error.clear();
    return true;
  };
  GenericAnalysis original;
  ProgramMatrixTaskContract matrix;
  if (!require(module && analyzeGenericProgram(*module, original, error),
               "analyze explicit BF16 source") ||
      !require(analyzeProgramMatrixTask(original, 0, matrix, error),
               "prove exact source arithmetic and affine addressing")) {
    return 1;
  }
  auto contract = original.function->getAttr("orbit.matrix_contract");
  original.function->removeAttr("orbit.matrix_contract");
  if (!require(!analyzeProgramMatrixTask(original, 0, matrix, error),
               "strict source without numerical opt-in cannot select matrix")) {
    return 1;
  }
  original.function->setAttr("orbit.matrix_contract", contract);
  mlir::affine::AffineLoadOp first;
  original.tasks[0].operation->walk([&](mlir::affine::AffineLoadOp op) {
    if (!first && op.getType().isBF16()) {
      first = op;
    }
  });
  auto a = first->getOperand(1), b = first->getOperand(2);
  first->setOperand(1, b);
  first->setOperand(2, a);
  if (!require(!analyzeProgramMatrixTask(original, 0, matrix, error),
               "transposed source addressing is rejected")) {
    return 1;
  }
  first->setOperand(1, a);
  first->setOperand(2, b);
  if (!require(distributeNativeInitialization(*module, error),
               "source initialization is explicitly distributed")) {
    return 1;
  }
  module->getOperation()->setAttr(
      "orbit.implementation",
      mlir::StringAttr::get(&context, programMatrixImplementation));
  GenericAnalysis source;
  GenericMappingPlan plan;
  llvm::json::Value config = llvm::json::Object{
      {"generic_air",
       llvm::json::Object{
           {"tile_sizes", llvm::json::Array{tile, tile, tile}},
           {"capabilities",
            llvm::json::Object{
                {"implementations",
                 llvm::json::Array{
                     llvm::json::Object{{"kind", "scalar"}, {"lanes", 1}},
                     llvm::json::Object{{"kind", programMatrixImplementation},
                                        {"lanes", 1}}}}}}}}};
  if (!require(
          analyzeGenericProgram(*module, source, error) &&
              planGenericMapping(source, config, plan, error),
          "native public planner accepts individually proved matrix tasks") ||
      !require(
          checkGenericMapping(source, plan, error),
          "independent native checker accepts source-bound implementation")) {
    return 1;
  }
  size_t matrices = 0, scalars = 0;
  for (auto &implementation : plan.taskImplementations) {
    matrices += implementation.kind == programMatrixImplementation;
    scalars += implementation.kind == "scalar";
  }
  if (!require(matrices >= 1 && scalars >= 1,
               "initializer/epilogue/conversion tasks stay scalar")) {
    return 1;
  }
  std::vector<ProgramGraphAIRSelection> selections;
  llvm::json::Value unionProof = nullptr;
  if (selected) {
    if (!require(buildProgramGraphAIRSelections(plan.communication, 8,
                                                selections, unionProof, error),
                 "prove complete safe configuration selection union"))
      return 1;
    std::vector<MappingRegion> imported;
    bool repeatedInput = false;
    for (const auto &event : plan.communication.events) {
      if (event.kind != "transfer")
        continue;
      const auto &allocation =
          plan.communication.allocations
              [plan.communication.copies[event.reads.front()].allocation];
      auto memory =
          llvm::find_if(plan.communication.memories, [&](const auto &value) {
            return value.id == allocation.memory;
          });
      if (memory->kind != "system-bo")
        continue;
      const auto &region = *event.transferRegion;
      repeatedInput |= llvm::any_of(imported, [&](const auto &previous) {
        return previous.storage == region.storage &&
               previous.offsets == region.offsets &&
               previous.sizes == region.sizes;
      });
      imported.push_back(region);
    }
    if (repeatedInput) {
      ProgramGraphAIRSelection unsafe{programGraphIdentity(plan.communication),
                                      0, plan.communication.events.size()};
      if (!require(!checkProgramGraphAIRSelection(plan.communication, unsafe,
                                                  error) &&
                       (error.find("repeated host input rectangle") !=
                            std::string::npos ||
                        error.find("multiple host output transfers") !=
                            std::string::npos),
                   "same input region cannot be grouped across output tiles"))
        return 1;
    }
    auto bad = selections.front();
    bad.end -= 1;
    if (!require(
            !checkProgramGraphAIRSelection(plan.communication, bad, error),
            "cut through live local result before host export is rejected"))
      return 1;
    auto overlap = selections;
    overlap.push_back(selections.back());
    if (!require(!checkProgramGraphAIRSelectionUnion(
                     plan.communication, overlap, unionProof, error),
                 "overlapping configuration union is rejected"))
      return 1;
    if (!require(checkProgramGraphAIRSelectionUnion(
                     plan.communication, selections, unionProof, error),
                 "negative union tests preserve complete valid coverage"))
      return 1;
  }
  ProgramGraphAIRLoopPlan loopPlan;
  if (compact && !require(buildProgramGraphAIRLoopPlan(plan.communication,
                                                       loopPlan, error),
                          "prove complete selected event loop template"))
    return 1;
  mlir::OwningOpRef<mlir::ModuleOp> air;
  llvm::json::Value manifest = nullptr;
  if (!require(
          lowerProgramGraphToAIR(plan.communication, air, manifest, error,
                                 compact ? &loopPlan : nullptr,
                                 selected ? &selections.front() : nullptr) &&
              mlir::succeeded(mlir::verify(*air)),
          "selected graph lowers with exact copies and matrix kernel call")) {
    return 1;
  }
  bool preservesSelectedFeeds = false;
  air->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "air.launch")
      preservesSelectedFeeds = op->hasAttr("air.preserve_shim_dma_order");
  });
  if (!require(preservesSelectedFeeds,
               "selected DMA occurrences opt out of shim channel folding"))
    return 1;
  auto receipt = programMatrixAIRManifest(*air);
  if (!require(*receipt.getAsObject()->getInteger("calls") >= 1 &&
                   *receipt.getAsObject()->getInteger(
                       "physical_scratch_bytes") > 0 &&
                   *receipt.getAsObject()->getInteger(
                       "additional_host_transfer_bytes") == 0,
               "layout scratch and external calls have an explicit receipt")) {
    return 1;
  }
  std::string output = argv[2];
  if (llvm::sys::fs::exists(output) ||
      llvm::sys::fs::create_directories(output)) {
    llvm::errs() << "use a new writable output directory\n";
    return 1;
  }
  auto write = [&](const std::string &name, llvm::StringRef contents) {
    std::error_code ec;
    llvm::raw_fd_ostream stream(output + "/" + name, ec);
    if (ec) {
      return false;
    }
    stream << contents;
    return true;
  };
  std::string ir;
  llvm::raw_string_ostream stream(ir);
  air->print(stream, mlir::OpPrintingFlags().printGenericOpForm());
  if (!write("program.air.mlir", ir) ||
      !write("manifest.json", llvm::formatv("{0:2}\n", manifest).str()) ||
      !write("plan.json",
             llvm::formatv("{0:2}\n", serializeGenericMapping(source, plan))
                 .str())) {
    return 1;
  }
  for (auto &kernel : *receipt.getAsObject()->getArray("kernels")) {
    auto *row = kernel.getAsObject();
    if (!write(row->getString("symbol")->str() + ".cc",
               programMatrixKernelSource(*row->getInteger("m"),
                                         *row->getInteger("k"),
                                         *row->getInteger("n")))) {
      return 1;
    }
  }
  if (selected) {
    if (!write("configuration-union.json",
               llvm::formatv("{0:2}\n", unionProof).str()))
      return 1;
    std::vector<int64_t> completeABI;
    for (const auto &entry :
         *manifest.getAsObject()->getArray("abi_storage_indices"))
      completeABI.push_back(*entry.getAsInteger());
    auto fullContract = serializeProgramMatrixRuntimeContract(
        plan.communication, completeABI, error);
    if (fullContract.getAsNull() ||
        !write("full-runtime-contract.json",
               llvm::formatv("{0:2}\n", fullContract).str()))
      return 1;
    size_t coveredTransfers = 0;
    for (size_t i = 0; i < selections.size(); ++i) {
      auto name = "configuration-" + std::to_string(i);
      if (llvm::sys::fs::create_directories(output + "/" + name))
        return 1;
      mlir::OwningOpRef<mlir::ModuleOp> fragment;
      llvm::json::Value physical = nullptr;
      if (!require(
              lowerProgramGraphAIRSelection(plan.communication, selections[i],
                                            fragment, physical, error),
              "re-emit exact selected source events with full ABI and slots"))
        return 1;
      std::vector<int64_t> abi;
      for (const auto &entry :
           *physical.getAsObject()->getArray("abi_storage_indices"))
        abi.push_back(*entry.getAsInteger());
      auto sealed = serializeProgramMatrixRuntimeContract(
          plan.communication, abi, selections[i], error);
      if (!require(!sealed.getAsNull(),
                   "seal selected contract without replanning"))
        return 1;
      const auto &rows = *sealed.getAsObject()->getArray("transfers");
      if (!require(llvm::count_if(rows,
                                  [](const auto &row) {
                                    return row.getAsObject()->getBoolean(
                                               "host_input") == false;
                                  }) <= 1,
                   "every selection has at most one checked host output"))
        return 1;
      coveredTransfers += sealed.getAsObject()->getArray("transfers")->size();
      std::string text;
      llvm::raw_string_ostream fs(text);
      fragment->print(fs, mlir::OpPrintingFlags().printGenericOpForm());
      fs << '\n';
      if (!write(name + "/program.air.mlir", text) ||
          !write(name + "/physical.json",
                 llvm::formatv("{0:2}\n", physical).str()) ||
          !write(name + "/runtime-contract.json",
                 llvm::formatv("{0:2}\n", sealed).str()))
        return 1;
    }
    size_t expected =
        llvm::count_if(plan.communication.events, [](const auto &event) {
          return event.kind == "transfer";
        });
    if (!require(coveredTransfers == expected,
                 "selected configuration contracts preserve every full-program "
                 "DMA exactly once"))
      return 1;
  }
  if (runtimeInput || (selected && argc == 5)) {
    mlir::ParserConfig parserConfig(&context, /*verifyAfterParse=*/false);
    auto actual = mlir::parseSourceFile<mlir::ModuleOp>(argv[selected ? 4 : 3],
                                                        parserConfig);
    std::vector<int64_t> abi;
    for (const auto &entry :
         *manifest.getAsObject()->getArray("abi_storage_indices")) {
      abi.push_back(*entry.getAsInteger());
    }
    llvm::json::Value proof = nullptr;
    auto sealed =
        selected ? serializeProgramMatrixRuntimeContract(
                       plan.communication, abi, selections.front(), error)
                 : serializeProgramMatrixRuntimeContract(plan.communication,
                                                         abi, error);
    if (!require(!sealed.getAsNull(),
                 "export checked source-bound runtime contract"))
      return 1;
    if (!write("runtime-contract.json", llvm::formatv("{0:2}\n", sealed).str()))
      return 1;

    auto *transferRows = sealed.getAsObject()->getArray("transfers");
    const size_t outputCount =
        llvm::count_if(*transferRows, [](const auto &row) {
          return row.getAsObject()->getBoolean("host_input") == false;
        });
    if (outputCount > 1) {
      mlir::OwningOpRef<mlir::ModuleOp> unsupported;
      if (!require(actual &&
                       !realizeProgramMatrixRuntime(
                           sealed, *actual, unsupported, proof, error) &&
                       error.find("multiple host output transfers") !=
                           std::string::npos,
                   "legacy multiple-output module is explicitly unsupported"))
        return 1;
      llvm::outs() << "matrix contract checks: " << checks
                   << "; legacy multiple-output runtime rejected; hardware not "
                      "executed\n";
      return 0;
    }
    if (selected) {
      if (!require(actual && !checkProgramMatrixRuntime(sealed, *actual, proof,
                                                        error),
                   "coalesced packet inputs are rejected before realization"))
        return 1;
    } else if (!require(actual &&
                            !checkProgramMatrixRuntime(plan.communication, abi,
                                                       *actual, proof, error),
                        "unfixed feedback prefetch fails independent ordering"))
      return 1;
    mlir::OwningOpRef<mlir::ModuleOp> patched;
    if (!require(
            realizeProgramMatrixRuntime(sealed, *actual, patched, proof, error),
            "realize selected feedback completion ordering") ||
        !require(checkProgramMatrixRuntime(sealed, *patched, proof, error),
                 "independent patched runtime ordering"))
      return 1;

    if (!require(
            checkProgramMatrixRuntime(sealed, *patched, proof, error),
            "sealed contract checker agrees without rebuilding selected graph"))
      return 1;
    auto mutated = llvm::json::parse(llvm::formatv("{0}", sealed).str());
    if (!mutated)
      return 1;
    auto &firstTransfer = (*mutated->getAsObject()->getArray("transfers"))[0];
    (*firstTransfer.getAsObject())["offset"] = int64_t(1);
    if (!require(!checkProgramMatrixRuntime(*mutated, *patched, proof, error),
                 "altered selected BD address is rejected"))
      return 1;
    mutated = llvm::json::parse(llvm::formatv("{0}", sealed).str());
    if (!mutated)
      return 1;
    auto *deps = mutated->getAsObject()->getArray("event_dependencies");
    (*deps)[0] = llvm::json::Array{int64_t(0)};
    if (!require(!checkProgramMatrixRuntime(*mutated, *patched, proof, error),
                 "cyclic contract dependency is rejected"))
      return 1;
    if (!require(checkProgramMatrixRuntime(sealed, *patched, proof, error),
                 "negative tests leave original proof unchanged"))
      return 1;
    if (selected) {
      auto changed = llvm::json::parse(llvm::formatv("{0}", sealed).str());
      if (!changed)
        return 1;
      auto *rows = changed->getAsObject()->getArray("transfers");
      std::vector<size_t> sameStream;
      for (size_t i = 0; i < rows->size(); ++i)
        if ((*rows)[i].getAsObject()->getInteger("storage") == int64_t(0))
          sameStream.push_back(i);
      if (!require(sameStream.size() == 2,
                   "selected fixture contains two K input tiles"))
        return 1;
      auto *first = (*rows)[sameStream[0]].getAsObject();
      auto *second = (*rows)[sameStream[1]].getAsObject();
      int64_t a = *first->getInteger("offset"),
              b = *second->getInteger("offset");
      (*first)["offset"] = b;
      (*second)["offset"] = a;
      if (!require(!checkProgramMatrixRuntime(*changed, *patched, proof, error),
                   "swapped K blocks fail exact FIFO proof"))
        return 1;
      (*first)["offset"] = a;
      (*second)["offset"] = a;
      if (!require(!checkProgramMatrixRuntime(*changed, *patched, proof, error),
                   "repeated K0 cannot replace selected K1"))
        return 1;
      (*second)["offset"] = b;
      auto truncated = *second->getArray("sizes");
      truncated[0] = int64_t(16);
      (*second)["sizes"] = std::move(truncated);
      if (!require(!checkProgramMatrixRuntime(*changed, *patched, proof, error),
                   "partial selected transfer is rejected"))
        return 1;
      if (!require(checkProgramMatrixRuntime(sealed, *patched, proof, error),
                   "FIFO negatives leave accepted descriptor intact"))
        return 1;
    }
    std::vector<mlir::Operation *> outputs;
    patched->walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef() != "aiex.dma_configure_task_for")
        return;
      auto props = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
          op->getPropertiesAsAttribute());
      auto alloc = props ? props.getAs<mlir::SymbolRefAttr>("alloc")
                         : mlir::SymbolRefAttr{};
      if (!alloc)
        return;
      bool outputTask = false;
      patched->walk([&](mlir::Operation *shim) {
        if (shim->getName().getStringRef() != "aie.shim_dma_allocation")
          return;
        auto attributes = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
            shim->getPropertiesAsAttribute());
        if (attributes &&
            attributes.getAs<mlir::StringAttr>("sym_name").getValue() ==
                alloc.getLeafReference() &&
            attributes.getAs<mlir::IntegerAttr>("channel_dir").getInt() == 0)
          outputTask = true;
      });
      if (outputTask)
        outputs.push_back(op);
    });
    if (outputs.size() >= 2) {
      auto oldProperties = outputs[1]->getPropertiesAsAttribute();
      mlir::NamedAttrList alias(
          mlir::cast<mlir::DictionaryAttr>(oldProperties));
      auto first = mlir::cast<mlir::DictionaryAttr>(
          outputs[0]->getPropertiesAsAttribute());
      alias.set("alloc", first.get("alloc"));
      if (mlir::failed(outputs[1]->setPropertiesFromAttribute(
              alias.getDictionary(&context),
              [&] { return outputs[1]->emitError(); })))
        return 1;
      mlir::Operation *firstStart = nullptr, *secondStart = nullptr;
      for (auto *user : outputs[0]->getResult(0).getUsers())
        if (user->getName().getStringRef() == "aiex.dma_start_task")
          firstStart = user;
      for (auto *user : outputs[1]->getResult(0).getUsers())
        if (user->getName().getStringRef() == "aiex.dma_start_task")
          secondStart = user;
      outputs[1]->moveBefore(firstStart);
      secondStart->moveBefore(firstStart);
      if (!require(!checkProgramMatrixRuntime(sealed, *patched, proof, error),
                   "multiple output tasks sharing a FIFO with swapped starts "
                   "are rejected"))
        return 1;
      outputs[1]->moveAfter(firstStart);
      secondStart->moveAfter(outputs[1]);
      if (mlir::failed(outputs[1]->setPropertiesFromAttribute(
              oldProperties, [&] { return outputs[1]->emitError(); })))
        return 1;
    }
    // These mutants preserve host addresses and lengths. They independently
    // exercise packet framing, concrete core storage and channel entry order.
    mlir::Operation *hostPacket = nullptr, *coreStart = nullptr;
    std::vector<mlir::Operation *> coreInputs;
    patched->walk([&](mlir::Operation *op) {
      auto name = op->getName().getStringRef();
      auto props = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
          op->getPropertiesAsAttribute());
      if (name == "aie.dma_start" && props &&
          props.getAs<mlir::IntegerAttr>("channel_dir").getInt() == 0)
        coreStart = op;
      if (name != "aie.dma_bd" || !props || !props.get("packet"))
        return;
      if (op->getParentOp()->getName().getStringRef() ==
          "aiex.dma_configure_task_for") {
        if (!hostPacket)
          hostPacket = op;
      } else {
        auto type =
            mlir::dyn_cast<mlir::MemRefType>(op->getOperand(0).getType());
        if (type && type.getElementType().isBF16())
          coreInputs.push_back(op);
      }
    });
    if (!require(hostPacket && coreStart && coreInputs.size() >= 2,
                 "actual packet routes expose distinct A/B core consumers"))
      return 1;
    auto packetProperties = hostPacket->getPropertiesAsAttribute();
    mlir::NamedAttrList noPacket(
        mlir::cast<mlir::DictionaryAttr>(packetProperties));
    noPacket.erase("packet");
    if (mlir::failed(hostPacket->setPropertiesFromAttribute(
            noPacket.getDictionary(&context),
            [&] { return hostPacket->emitError(); })))
      return 1;
    if (!require(!checkProgramMatrixRuntime(sealed, *patched, proof, error),
                 "missing host packet attribute is rejected"))
      return 1;
    if (mlir::failed(hostPacket->setPropertiesFromAttribute(
            packetProperties, [&] { return hostPacket->emitError(); })))
      return 1;
    auto aBuffer = coreInputs[0]->getOperand(0),
         bBuffer = coreInputs[1]->getOperand(0);
    coreInputs[0]->setOperand(0, bBuffer);
    coreInputs[1]->setOperand(0, aBuffer);
    if (!require(!checkProgramMatrixRuntime(sealed, *patched, proof, error),
                 "same-length A/B core destination swap is rejected"))
      return 1;
    coreInputs[0]->setOperand(0, aBuffer);
    coreInputs[1]->setOperand(0, bBuffer);
    auto *head = coreStart->getSuccessor(0);
    auto *wrongHead = head->back().getSuccessor(0);
    coreStart->setSuccessor(wrongHead, 0);
    if (!require(!checkProgramMatrixRuntime(sealed, *patched, proof, error),
                 "wrong first core DMA BD is rejected"))
      return 1;
    coreStart->setSuccessor(head, 0);
    auto handle = hostPacket->getParentOp()->getResult(0);
    mlir::Operation *inputAwait = nullptr, *inputFree = nullptr;
    for (auto *user : handle.getUsers()) {
      if (user->getName().getStringRef() == "aiex.dma_await_task")
        inputAwait = user;
      if (user->getName().getStringRef() == "aiex.dma_free_task")
        inputFree = user;
    }
    if (!require(inputAwait && inputFree,
                 "realized packet has completion and release"))
      return 1;
    inputFree->moveBefore(inputAwait);
    if (!require(!checkProgramMatrixRuntime(sealed, *patched, proof, error),
                 "free-before-completion is rejected"))
      return 1;
    inputFree->moveAfter(inputAwait);
    std::string brokenText;
    llvm::raw_string_ostream brokenStream(brokenText);
    patched->print(brokenStream, mlir::OpPrintingFlags().printGenericOpForm());
    auto broken = mlir::parseSourceString<mlir::ModuleOp>(
        brokenText, mlir::ParserConfig(&context, false));
    mlir::Operation *missing = nullptr;
    broken->walk([&](mlir::Operation *op) {
      if (missing ||
          op->getName().getStringRef() != "aiex.dma_configure_task_for")
        return;
      for (auto &inner : op->getRegion(0).front()) {
        auto props = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
            inner.getPropertiesAsAttribute());
        if (props && props.get("packet"))
          missing = op;
      }
    });
    if (!require(missing != nullptr,
                 "missing-descriptor mutation finds a selected packet"))
      return 1;
    for (auto *user :
         llvm::make_early_inc_range(missing->getResult(0).getUsers()))
      user->erase();
    missing->erase();
    if (!require(!checkProgramMatrixRuntime(sealed, *broken, proof, error),
                 "missing input descriptor returns rejection without abort"))
      return 1;
    if (!require(checkProgramMatrixRuntime(sealed, *patched, proof, error),
                 "all packet/core mutants leave accepted module unchanged"))
      return 1;

    mlir::Operation *memory = nullptr;
    patched->walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef() == "aie.mem")
        memory = op;
    });
    if (!require(memory != nullptr,
                 "actual final AIE contains core DMA memory"))
      return 1;
    std::vector<mlir::Block *> extras;
    while (memory->getRegion(0).getBlocks().size() <= 16) {
      auto *block = new mlir::Block();
      memory->getRegion(0).push_back(block);
      extras.push_back(block);
    }
    if (!require(!checkProgramMatrixRuntime(sealed, *patched, proof, error),
                 "seventeen static core DMA blocks are rejected"))
      return 1;
    for (auto *block : extras)
      block->erase();
    mlir::Operation *lock = nullptr;
    patched->walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef() == "aie.lock")
        lock = op;
    });
    if (!require(lock != nullptr, "actual final AIE contains explicit locks"))
      return 1;
    auto properties = lock->getPropertiesAsAttribute();
    auto dictionary = mlir::cast<mlir::DictionaryAttr>(properties);
    mlir::NamedAttrList replacement(dictionary);
    replacement.set("lockID", mlir::IntegerAttr::get(
                                  mlir::IntegerType::get(&context, 32), 16));
    if (mlir::failed(lock->setPropertiesFromAttribute(
            replacement.getDictionary(&context),
            [&] { return lock->emitError(); })))
      return 1;
    if (!require(!checkProgramMatrixRuntime(sealed, *patched, proof, error),
                 "out-of-range physical lock ID is rejected"))
      return 1;
    if (mlir::failed(lock->setPropertiesFromAttribute(
            properties, [&] { return lock->emitError(); })))
      return 1;
    if (!require(
            checkProgramMatrixRuntime(sealed, *patched, proof, error),
            "static-resource mutations do not modify the accepted artifact"))
      return 1;
    std::string fixed;
    llvm::raw_string_ostream fs(fixed);
    patched->print(fs, mlir::OpPrintingFlags().printGenericOpForm());
    fs << '\n';
    if (!write("runtime.generic.mlir", fixed) ||
        !write("runtime-proof.json", llvm::formatv("{0:2}\n", proof).str()))
      return 1;
  }
  llvm::outs() << "matrix contract checks: " << checks
               << "; AIR emitted; hardware not executed\n";
  return 0;
}
