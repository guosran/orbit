//===- ExactScoreCache.h ----------------------------------------*- C++ -*-===//
//
// A content-addressed (by an exact structural key, never by a digest) cache
// for complete exact-joint score files. The entry directory number is only a
// collision-free locator. The key.txt contents are always compared byte for
// byte before an entry can be reused.
//
// This file is intentionally header-only so it can be added beside
// ScoreExactJointTaskCandidatesPass.cpp without changing the library target's
// source list.
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_EXACT_SCORE_CACHE_H
#define AMOEBA_EXACT_SCORE_CACHE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <sys/file.h>
#include <unistd.h>

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

/// Explicit proof supplied by the scorer when importing a complete result
/// whose exact key predates the production fixed-point fields.  A missing
/// budget is never inferred: the caller must explicitly declare the legacy
/// compatibility upper bound (currently 64).
struct ExactScoreCacheMigrationOptions {
  llvm::StringRef legacyKey;
  llvm::StringRef productionContract;
  int64_t effectiveProductionFixedPointIterations = 0;
  int64_t explicitLegacyMaxIterations = -1;
  llvm::StringRef proofReason;
};

struct ExactScoreCacheMigrationProof {
  int64_t budgetBefore = 0;
  int64_t budgetAfter = 0;
  bool legacyUpperBoundDeclared = false;
};

/// A lease is either a cache hit or the sole owner allowed to publish one
/// complete result for an exact key. A pending/incomplete result is never
/// considered reusable. Waiting has no wall-time limit: a second search for
/// the same key waits for the first owner to finish instead of duplicating the
/// exact search.
class ExactScoreCacheLease {
public:
  ExactScoreCacheLease() = default;
  ExactScoreCacheLease(const ExactScoreCacheLease &) = delete;
  ExactScoreCacheLease &operator=(const ExactScoreCacheLease &) = delete;

  ExactScoreCacheLease(ExactScoreCacheLease &&other) noexcept {
    moveFrom(std::move(other));
  }
  ExactScoreCacheLease &operator=(ExactScoreCacheLease &&other) noexcept {
    if (this != &other) {
      release();
      moveFrom(std::move(other));
    }
    return *this;
  }

  ~ExactScoreCacheLease() { release(); }

  /// Returns false only when the cache itself cannot be used. The caller may
  /// continue the normal search in that case because caching is an
  /// optimization, never a legality or completeness condition.
  bool begin(llvm::StringRef cacheDirectory, llvm::StringRef exactKey,
             llvm::StringRef architecturePath, llvm::StringRef graphVariantId,
             llvm::StringRef sourceGraphId, llvm::StringRef stage,
             llvm::StringRef outputPath,
             std::string &error) {
    return begin(cacheDirectory, exactKey, architecturePath, graphVariantId,
                 sourceGraphId, stage, outputPath,
                 ExactScoreCacheMigrationOptions{}, error);
  }

  /// As above, with an optional, explicitly authorized migration from the
  /// exact legacy key.  Migration is accepted only for a complete certified
  /// production result; otherwise the caller owns a normal new-key search.
  bool begin(llvm::StringRef cacheDirectory, llvm::StringRef exactKey,
             llvm::StringRef architecturePath, llvm::StringRef graphVariantId,
             llvm::StringRef sourceGraphId, llvm::StringRef stage,
             llvm::StringRef outputPath,
             const ExactScoreCacheMigrationOptions &migration,
             std::string &error) {
    if (cacheDirectory.empty())
      return true;
    cacheDirectory_ = cacheDirectory.str();
    key_ = exactKey.str();
    architecturePath_ = architecturePath.str();
    graphVariantId_ = graphVariantId.str();
    sourceGraphId_ = sourceGraphId.str();
    stage_ = stage.str();
    outputPath_ = outputPath.str();
    migrationLegacyKey_ = migration.legacyKey.str();
    migrationContract_ = migration.productionContract.str();
    migrationEffectiveIterations_ =
        migration.effectiveProductionFixedPointIterations;
    migrationExplicitLegacyMaxIterations_ =
        migration.explicitLegacyMaxIterations;
    migrationProofReason_ = migration.proofReason.str();

    std::error_code ec = llvm::sys::fs::create_directories(cacheDirectory_);
    if (ec) {
      error = "cannot create score cache directory " + cacheDirectory_ +
              ": " + ec.message();
      reset();
      return false;
    }

    const std::string indexLock = cacheDirectory_ + "/index.lock";
    if (!acquireFileLock(indexLock, indexFd_, error)) {
      reset();
      return false;
    }

    uint64_t nextId = 0;
    const std::string nextIdPath = cacheDirectory_ + "/next-id";
    std::string nextText;
    if (readText(nextIdPath, nextText) && !nextText.empty()) {
      llvm::StringRef text(nextText);
      text = text.trim();
      if (text.getAsInteger(10, nextId))
        nextId = 0;
    }

    uint64_t scanNextId = 0;
    while (llvm::sys::fs::exists(entryPath(scanNextId))) {
      if (scanNextId == std::numeric_limits<uint64_t>::max())
        break;
      ++scanNextId;
    }
    nextId = std::max(nextId, scanNextId);

    std::string matchedEntry;
    for (uint64_t id = 0; id < nextId; ++id) {
      const std::string entry = entryPath(id);
      std::string storedKey;
      if (!readText(entry + "/key.txt", storedKey) || storedKey != key_)
        continue;
      matchedEntry = entry;
      break;
    }

    std::string legacyScore;
    std::string legacySourceEntry;
    ExactScoreCacheMigrationProof legacyProof;
    bool migrateLegacy = false;
    if (matchedEntry.empty() && !migrationLegacyKey_.empty() &&
        migrationEffectiveIterations_ > 0) {
      for (uint64_t id = 0; id < nextId; ++id) {
        const std::string entry = entryPath(id);
        std::string storedKey;
        if (!readText(entry + "/key.txt", storedKey) ||
            storedKey != migrationLegacyKey_)
          continue;
        std::string status;
        if (!readText(entry + "/status", status) ||
            llvm::StringRef(status).trim() != "complete")
          break;
        if (!readText(entry + "/result.jsonl", legacyScore))
          break;
        std::string validationError;
        if (!validateLegacyProductionScore(
                legacyScore, migrationContract_,
                migrationEffectiveIterations_,
                migrationExplicitLegacyMaxIterations_, legacyProof,
                validationError))
          break;
        legacySourceEntry = entry;
        migrateLegacy = true;
        break;
      }
    }

    if (matchedEntry.empty()) {
      matchedEntry = entryPath(nextId);
      while (llvm::sys::fs::exists(matchedEntry)) {
        if (nextId == std::numeric_limits<uint64_t>::max()) {
          closeFileLock(indexFd_);
          error = "score cache entry numbering exhausted";
          reset();
          return false;
        }
        matchedEntry = entryPath(++nextId);
      }
      std::error_code createError =
          llvm::sys::fs::create_directory(matchedEntry);
      if (createError) {
        closeFileLock(indexFd_);
        error = "cannot create score cache entry " + matchedEntry + ": " +
                createError.message();
        reset();
        return false;
      }
      if (!writeTextAtomically(matchedEntry + "/key.txt", key_, error) ||
          !writeTextAtomically(matchedEntry + "/status", "pending\n", error)) {
        closeFileLock(indexFd_);
        reset();
        return false;
      }
      if (nextId == std::numeric_limits<uint64_t>::max()) {
        closeFileLock(indexFd_);
        error = "score cache entry numbering exhausted";
        reset();
        return false;
      }
      ++nextId;
      if (!writeTextAtomically(nextIdPath, std::to_string(nextId), error)) {
        closeFileLock(indexFd_);
        reset();
        return false;
      }
    }

    closeFileLock(indexFd_);
    entryPath_ = std::move(matchedEntry);
    const std::string entryLock = entryPath_ + "/entry.lock";
    if (!acquireFileLock(entryLock, entryFd_, error)) {
      reset();
      return false;
    }
    std::string status;
    readText(entryPath_ + "/status", status);
    status = llvm::StringRef(status).trim().str();
    if (status == "complete") {
      std::string cached;
      if (readText(entryPath_ + "/result.jsonl", cached) &&
          isCompleteScore(cached)) {
        if (!materialize(cached, architecturePath_, graphVariantId_,
                         sourceGraphId_, stage_, outputPath_,
                         /*peerImported=*/false,
                         error)) {
          release();
          return false;
        }
        hit_ = true;
        closeFileLock(entryFd_);
        entryPath_.clear();
        return true;
      }
      // A malformed complete entry is treated as an unpublishable old or
      // interrupted entry. The owner repairs it in place under this lock.
    }
    if (migrateLegacy) {
      owner_ = true;
      if (!materialize(legacyScore, architecturePath_, graphVariantId_,
                       sourceGraphId_, stage_, outputPath_,
                       /*peerImported=*/false, &legacyProof,
                       migrationContract_, migrationProofReason_,
                       legacySourceEntry, error)) {
        release();
        return false;
      }
      std::string migratedResult;
      if (!readText(outputPath_, migratedResult) ||
          !writeTextAtomically(entryPath_ + "/result.jsonl", migratedResult,
                               error)) {
        release();
        return false;
      }
      std::string evidence;
      llvm::raw_string_ostream evidenceStream(evidence);
      llvm::json::Object migrationRecord;
      migrationRecord["schema"] = "orbit-exact-score-cache-migration-v1";
      migrationRecord["source_entry"] = legacySourceEntry;
      migrationRecord["budget_before"] = legacyProof.budgetBefore;
      migrationRecord["budget_after"] = legacyProof.budgetAfter;
      migrationRecord["budget_before_kind"] =
          legacyProof.legacyUpperBoundDeclared
              ? "explicit-legacy-upper-bound"
              : "recorded-fixed-point-budget";
      migrationRecord["success_prefix_only"] = true;
      migrationRecord["key_match"] = "byte-exact-legacy-key";
      migrationRecord["legacy_key_byte_exact"] = true;
      migrationRecord["legacy_key_length"] =
          static_cast<int64_t>(migrationLegacyKey_.size());
      migrationRecord["new_key_length"] =
          static_cast<int64_t>(key_.size());
      migrationRecord["proof_reason"] = migrationProofReason_;
      migrationRecord["production_scheduler_contract"] = migrationContract_;
      if (legacyProof.legacyUpperBoundDeclared)
        migrationRecord["compatibility_option"] =
            "legacy-production-cache-max-iterations=64";
      evidenceStream << llvm::json::Value(std::move(migrationRecord)) << "\n";
      evidenceStream.flush();
      if (!writeTextAtomically(entryPath_ + "/migration.json", evidence,
                               error) ||
          !writeTextAtomically(entryPath_ + "/status", "complete\n",
                               error)) {
        release();
        return false;
      }
      owner_ = false;
      published_ = true;
      migrated_ = true;
      hit_ = true;
      closeFileLock(entryFd_);
      entryPath_.clear();
      return true;
    }
    if (!writeTextAtomically(entryPath_ + "/status", "pending\n", error)) {
      release();
      return false;
    }
    owner_ = true;
    return true;
  }

  bool enabled() const { return !cacheDirectory_.empty(); }
  bool hit() const { return hit_; }
  bool owner() const { return owner_; }
  bool migrated() const { return migrated_; }

  /// Publish only a final score file whose own footer proves complete/top-k.
  /// The generated output is copied into the entry before status becomes
  /// complete, so readers can never observe a complete marker with a partial
  /// result.
  bool publish(llvm::StringRef generatedOutput, std::string &error) {
    if (!owner_)
      return true;
    std::string result;
    if (!readText(generatedOutput, result)) {
      error = "cannot read completed score output for cache publication: " +
              generatedOutput.str();
      return false;
    }
    if (!isCompleteScore(result)) {
      error = "refusing to cache a score output without complete/top-k proof";
      return false;
    }
    if (!writeTextAtomically(entryPath_ + "/result.jsonl", result, error))
      return false;
    if (!writeTextAtomically(entryPath_ + "/status", "complete\n", error))
      return false;
    owner_ = false;
    published_ = true;
    // The result is durable and status was published last; no reader needs
    // the writer lock after this point. Releasing here also permits a
    // same-process validation/replay object to observe the newly complete
    // entry without waiting for the producer lease's destructor.
    closeFileLock(entryFd_);
    entryPath_.clear();
    return true;
  }

  /// Adopt a separately produced complete score after the caller has checked
  /// that its module, shape/cost inputs, architecture, and search options are
  /// identical to this lease's key.  The peer bytes are materialized through
  /// the same current-path/source binding rewrite as an ordinary cache hit;
  /// incomplete peer output can never reach publish().
  bool publishImportedPeer(llvm::StringRef peerScore, std::string &error) {
    if (!owner_) {
      error = "cannot import a peer score without an active cache owner";
      return false;
    }
    if (!isCompleteScore(peerScore)) {
      error = "refusing to import an incomplete peer score";
      return false;
    }
    if (!materialize(peerScore, architecturePath_, graphVariantId_,
                     sourceGraphId_, stage_, outputPath_,
                     /*peerImported=*/true,
                     error))
      return false;
    return publish(outputPath_, error);
  }

  /// Mark the entry non-reusable before releasing the lock. This is used by
  /// callers that deliberately stop after an incomplete/bounded search.
  void discard() {
    if (owner_)
      writeTextAtomically(entryPath_ + "/status", "incomplete\n", ignored_);
    owner_ = false;
  }

  void release() {
    if (entryPath_.empty()) {
      reset();
      return;
    }
    if (owner_ && !published_)
      writeTextAtomically(entryPath_ + "/status", "incomplete\n", ignored_);
    reset();
  }

private:
  static bool readText(llvm::StringRef path, std::string &text) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(path);
    if (!buffer)
      return false;
    text = (*buffer)->getBuffer().str();
    return true;
  }

  static bool writeTextAtomically(llvm::StringRef path, llvm::StringRef text,
                                  std::string &error) {
    llvm::SmallString<256> pattern(path);
    pattern += ".tmp-%%%%%%";
    llvm::SmallString<256> temporary;
    int descriptor = -1;
    std::error_code ec =
        llvm::sys::fs::createUniqueFile(pattern, descriptor, temporary);
    if (ec) {
      error = "cannot create score cache temporary " + temporary.str().str() +
              ": " + ec.message();
      return false;
    }
    {
      llvm::raw_fd_ostream stream(descriptor, /*shouldClose=*/true);
      stream.write(text.data(), text.size());
      stream.flush();
      if (stream.has_error()) {
        error = "cannot write score cache file " + path.str();
        llvm::sys::fs::remove(temporary);
        return false;
      }
    }
    ec = llvm::sys::fs::rename(temporary, path);
    if (ec) {
      error = "cannot publish score cache file " + path.str() + ": " +
              ec.message();
      llvm::sys::fs::remove(temporary);
      return false;
    }
    return true;
  }

  static bool acquireFileLock(llvm::StringRef path, int &fd,
                              std::string &error) {
    fd = ::open(path.str().c_str(), O_CREAT | O_RDWR, 0600);
    if (fd < 0) {
      error = "cannot open score cache lock " + path.str();
      return false;
    }
    while (::flock(fd, LOCK_EX) != 0) {
      if (errno == EINTR)
        continue;
      error = "cannot acquire score cache lock " + path.str();
      ::close(fd);
      fd = -1;
      return false;
    }
    return true;
  }

  static void closeFileLock(int &fd) {
    if (fd >= 0) {
      ::flock(fd, LOCK_UN);
      ::close(fd);
      fd = -1;
    }
  }

  std::string entryPath(uint64_t id) const {
    return cacheDirectory_ + "/entry-" + std::to_string(id);
  }

  static bool isCompleteScore(llvm::StringRef text) {
    bool sawHeader = false;
    bool sawFooter = false;
    unsigned scoreRows = 0;
    llvm::MemoryBufferRef buffer(text, "cached-score");
    llvm::line_iterator lines(buffer, true);
    for (; !lines.is_at_end(); ++lines) {
      llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*lines);
      if (!parsed)
        return false;
      const llvm::json::Object *object = parsed->getAsObject();
      if (!object)
        return false;
      std::optional<llvm::StringRef> schema = object->getString("schema");
      if (!schema || *schema != "amoeba-exact-joint-task-scores-v1")
        return false;
      std::optional<llvm::StringRef> type = object->getString("record_type");
      if (!type)
        return false;
      if (*type == "header") {
        if (sawHeader || sawFooter)
          return false;
        sawHeader = true;
        continue;
      }
      if (!sawHeader || sawFooter)
        return false;
      if (*type == "score") {
        ++scoreRows;
        continue;
      }
      if (*type != "footer")
        return false;
      if (sawFooter)
        return false;
      sawFooter = true;
      if (object->getString("status") !=
              std::optional<llvm::StringRef>("complete") ||
          object->getBoolean("incomplete") != std::optional<bool>(false) ||
          object->getBoolean("top_k_certified") != std::optional<bool>(true) ||
          object->getInteger("top_k_requested").value_or(0) < 5)
        return false;
      std::optional<llvm::StringRef> certificate =
          object->getString("ranking_certificate");
      if (!certificate || *certificate == "best-found-uncertified")
        return false;
    }
    return sawHeader && sawFooter && scoreRows >= 5;
  }

  static bool hasFailedProductionTuple(const llvm::json::Object &object) {
    // Keep this list explicit.  A complete result from a failed production
    // tuple is never a valid success-prefix migration, even if its footer
    // happens to carry a complete-looking status.
    return object.getString("failed_production_shape_candidate_id") ||
           object.getString("failed_production_candidate_id") ||
           object.getString("failed_production_phase") ||
           object.getArray("failed_production_task_costs") ||
           object.getBoolean("production_scheduler_failed").value_or(false);
  }

  static bool validateLegacyProductionScore(
      llvm::StringRef text, llvm::StringRef expectedContract,
      int64_t effectiveIterations, int64_t explicitLegacyMaxIterations,
      ExactScoreCacheMigrationProof &proof, std::string &error) {
    if (effectiveIterations <= 0) {
      error = "legacy production migration has no effective iteration budget";
      return false;
    }
    if (!isCompleteScore(text)) {
      error = "legacy score is not complete and top-k certified";
      return false;
    }

    std::optional<llvm::json::Object> header;
    std::optional<llvm::json::Object> footer;
    llvm::MemoryBufferRef buffer(text, "legacy-score");
    llvm::line_iterator lines(buffer, true);
    for (; !lines.is_at_end(); ++lines) {
      llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*lines);
      if (!parsed)
        return false;
      const llvm::json::Object *object = parsed->getAsObject();
      if (!object)
        return false;
      auto type = object->getString("record_type");
      if (!type)
        return false;
      if (*type == "header")
        header = *object;
      else if (*type == "footer")
        footer = *object;
    }
    if (!header || !footer) {
      error = "legacy score has no header/footer for migration proof";
      return false;
    }

    const llvm::json::Object &headerObject = *header;
    const llvm::json::Object &footerObject = *footer;
    auto requireStringPair = [&](llvm::StringRef field,
                                 llvm::StringRef expected) {
      auto headerValue = headerObject.getString(field);
      auto footerValue = footerObject.getString(field);
      return headerValue && footerValue && *headerValue == expected &&
             *footerValue == expected;
    };
    if (!requireStringPair("temporal_model", "production-scheduler") ||
        !requireStringPair("schedule_space", "production-scheduler") ||
        headerObject.getString("communication_mode") !=
            std::optional<llvm::StringRef>("explicit") ||
        (footerObject.get("communication_mode") &&
         footerObject.getString("communication_mode") !=
             std::optional<llvm::StringRef>("explicit")) ||
        !requireStringPair("scheduler_backend",
                           "orchestrate-tasks-on-accelerators") ||
        footerObject.getString("ranking_certificate") !=
            std::optional<llvm::StringRef>(
                "proven-shape-space-under-production-scheduler")) {
      error = "legacy score production strategy is not the current scheduler";
      return false;
    }
    auto headerDispatch = headerObject.getString("dispatch_policy");
    auto footerDispatch = footerObject.getString("dispatch_policy");
    if (!headerDispatch || !footerDispatch ||
        *headerDispatch != *footerDispatch || headerDispatch->empty()) {
      error = "legacy score dispatch policy is missing or inconsistent";
      return false;
    }
    auto requireFalse = [&](const llvm::json::Object &object,
                            llvm::StringRef field) {
      return object.getBoolean(field) == std::optional<bool>(false);
    };
    if (!requireFalse(headerObject, "global_cutoff_requested") ||
        !requireFalse(footerObject, "global_cutoff_requested") ||
        !requireFalse(headerObject, "score_cache_disabled_for_global_cutoff") ||
        !requireFalse(footerObject, "score_cache_disabled_for_global_cutoff")) {
      error = "legacy score carries a global-cutoff result";
      return false;
    }
    if (headerObject.getInteger("max_makespan") !=
            std::optional<int64_t>(0) ||
        (footerObject.get("max_makespan") &&
         footerObject.getInteger("max_makespan") !=
             std::optional<int64_t>(0)) ||
        headerObject.getBoolean("placement_coverage_limited") !=
            std::optional<bool>(true) ||
        footerObject.getBoolean("placement_coverage_limited") !=
            std::optional<bool>(true) ||
        hasFailedProductionTuple(headerObject) ||
        hasFailedProductionTuple(footerObject)) {
      error = "legacy score lacks the required production success-prefix proof";
      return false;
    }

    auto headerContract =
        headerObject.getString("production_scheduler_contract");
    auto footerContract =
        footerObject.getString("production_scheduler_contract");
    if (headerContract || footerContract) {
      if (!headerContract || !footerContract ||
          *headerContract != *footerContract ||
          *headerContract != expectedContract) {
        error = "legacy score production contract is inconsistent";
        return false;
      }
    }

    auto headerIterations = headerObject.getInteger(
        "production_fixed_point_max_iterations");
    auto footerIterations = footerObject.getInteger(
        "production_fixed_point_max_iterations");
    if (headerIterations || footerIterations) {
      if (!headerIterations || !footerIterations ||
          *headerIterations != *footerIterations || *headerIterations <= 0 ||
          *headerIterations > effectiveIterations) {
        error = "legacy score fixed-point budget is missing, inconsistent, or newer";
        return false;
      }
      proof.budgetBefore = *headerIterations;
      proof.budgetAfter = effectiveIterations;
      proof.legacyUpperBoundDeclared = false;
      return true;
    }

    // The old result did not record the budget.  Accept only the explicitly
    // named local compatibility declaration for the known legacy upper bound
    // 64, and only when the current effective budget is also 64.
    if (explicitLegacyMaxIterations != 64 || effectiveIterations != 64) {
      error = "legacy score omits its fixed-point budget without the explicit 64 compatibility declaration";
      return false;
    }
    proof.budgetBefore = explicitLegacyMaxIterations;
    proof.budgetAfter = effectiveIterations;
    proof.legacyUpperBoundDeclared = true;
    return true;
  }

  static bool materialize(llvm::StringRef cached, llvm::StringRef architecture,
                          llvm::StringRef graphVariantId,
                          llvm::StringRef sourceGraphId,
                          llvm::StringRef stage,
                          llvm::StringRef output, bool peerImported,
                          std::string &error) {
    return materialize(cached, architecture, graphVariantId, sourceGraphId,
                       stage, output, peerImported, nullptr, "", "", "",
                       error);
  }

  static bool materialize(
      llvm::StringRef cached, llvm::StringRef architecture,
      llvm::StringRef graphVariantId, llvm::StringRef sourceGraphId,
      llvm::StringRef stage, llvm::StringRef output, bool peerImported,
      const ExactScoreCacheMigrationProof *migration,
      llvm::StringRef migrationContract, llvm::StringRef migrationReason,
      llvm::StringRef migrationSourceEntry, std::string &error) {
    llvm::MemoryBufferRef buffer(cached, "cached-score");
    llvm::line_iterator lines(buffer, true);
    std::string rewritten;
    llvm::raw_string_ostream stream(rewritten);
    for (; !lines.is_at_end(); ++lines) {
      llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*lines);
      if (!parsed) {
        error = "cached score contains invalid JSON";
        return false;
      }
      const llvm::json::Object *source = parsed->getAsObject();
      if (!source) {
        error = "cached score contains a non-object record";
        return false;
      }
      llvm::json::Object object = *source;
      std::optional<llvm::StringRef> type = object.getString("record_type");
      if (type && (*type == "header" || *type == "footer")) {
        object["architecture_path"] = architecture.str();
        object["graph_variant_id"] = graphVariantId.str();
        object["source_graph_id"] = sourceGraphId.str();
        if (!stage.empty())
          object["stage"] = stage.str();
        else
          object.erase("stage");
        object["score_cache_enabled"] = true;
        object["score_cache_hit"] = true;
        object["score_cache_reused"] = true;
        object["score_cache_new_expanded_nodes"] = 0;
        if (peerImported)
          object["score_cache_peer_imported"] = true;
        if (migration) {
          object["production_scheduler_contract"] = migrationContract.str();
          object["production_fixed_point_max_iterations"] =
              migration->budgetAfter;
          object["score_cache_migrated"] = true;
          object["score_cache_migration_source_entry"] =
              migrationSourceEntry.str();
          object["score_cache_migration_budget_before"] =
              migration->budgetBefore;
          object["score_cache_migration_budget_after"] =
              migration->budgetAfter;
          object["score_cache_migration_budget_before_kind"] =
              migration->legacyUpperBoundDeclared
                  ? "explicit-legacy-upper-bound"
                  : "recorded-fixed-point-budget";
          object["score_cache_migration_success_prefix_only"] = true;
          object["score_cache_migration_reason"] = migrationReason.str();
          if (migration->legacyUpperBoundDeclared)
            object["score_cache_migration_compatibility"] =
                "legacy-production-cache-max-iterations=64";
        }
      }
      stream << llvm::json::Value(std::move(object)) << "\n";
    }
    stream.flush();
    return writeTextAtomically(output, rewritten, error);
  }

  void moveFrom(ExactScoreCacheLease &&other) {
    cacheDirectory_ = std::move(other.cacheDirectory_);
    key_ = std::move(other.key_);
    architecturePath_ = std::move(other.architecturePath_);
    graphVariantId_ = std::move(other.graphVariantId_);
    sourceGraphId_ = std::move(other.sourceGraphId_);
    stage_ = std::move(other.stage_);
    outputPath_ = std::move(other.outputPath_);
    migrationLegacyKey_ = std::move(other.migrationLegacyKey_);
    migrationContract_ = std::move(other.migrationContract_);
    migrationEffectiveIterations_ = other.migrationEffectiveIterations_;
    migrationExplicitLegacyMaxIterations_ =
        other.migrationExplicitLegacyMaxIterations_;
    migrationProofReason_ = std::move(other.migrationProofReason_);
    entryPath_ = std::move(other.entryPath_);
    ignored_ = std::move(other.ignored_);
    indexFd_ = other.indexFd_;
    entryFd_ = other.entryFd_;
    other.indexFd_ = -1;
    other.entryFd_ = -1;
    owner_ = other.owner_;
    hit_ = other.hit_;
    published_ = other.published_;
    migrated_ = other.migrated_;
    other.owner_ = false;
    other.hit_ = false;
    other.published_ = false;
    other.migrated_ = false;
    other.entryPath_.clear();
  }

  void reset() {
    closeFileLock(indexFd_);
    closeFileLock(entryFd_);
    cacheDirectory_.clear();
    key_.clear();
    architecturePath_.clear();
    graphVariantId_.clear();
    sourceGraphId_.clear();
    stage_.clear();
    outputPath_.clear();
    migrationLegacyKey_.clear();
    migrationContract_.clear();
    migrationEffectiveIterations_ = 0;
    migrationExplicitLegacyMaxIterations_ = -1;
    migrationProofReason_.clear();
    entryPath_.clear();
    owner_ = false;
    hit_ = false;
    published_ = false;
    migrated_ = false;
  }

  std::string cacheDirectory_;
  std::string key_;
  std::string architecturePath_;
  std::string graphVariantId_;
  std::string sourceGraphId_;
  std::string stage_;
  std::string outputPath_;
  std::string migrationLegacyKey_;
  std::string migrationContract_;
  int64_t migrationEffectiveIterations_ = 0;
  int64_t migrationExplicitLegacyMaxIterations_ = -1;
  std::string migrationProofReason_;
  std::string entryPath_;
  std::string ignored_;
  int indexFd_ = -1;
  int entryFd_ = -1;
  bool owner_ = false;
  bool hit_ = false;
  bool published_ = false;
  bool migrated_ = false;
};

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_EXACT_SCORE_CACHE_H
