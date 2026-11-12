#include "Taskflow/SyncCost.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <tuple>

using namespace orbit::taskflow;
namespace {
using O = llvm::json::Object;
using A = llvm::json::Array;
using J = llvm::json::Value;
bool fail(std::string &e, const std::string &s) {
  e = "sync API cost: " + s;
  return false;
}
bool digest(llvm::StringRef s) {
  return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
bool knownState(const std::string &s) {
  static const std::set<std::string> states = {
      "host-write-observed",        "host-write-since-sync",
      "host-read-observed",         "host-read-since-sync",
      "post-TO-no-access",          "post-FROM-no-access",
      "after-device-read-complete", "after-device-write-complete"};
  return states.count(s);
}
auto key(const SyncCostDomain &d) {
  return std::tie(d.site, d.argumentOrdinal, d.callForm, d.observerBoundary,
                  d.direction, d.runtimeIdentity, d.observerIdentity,
                  d.conditionIdentity);
}
bool validate(const SyncAPICostModel &m, std::string &e) {
  if (m.modelId.empty() || !digest(m.parameterFileSHA256) || m.rows.empty())
    return fail(e,
                "model requires a label, exact-file SHA256 and nonempty rows");
  for (size_t i = 0; i < m.rows.size(); ++i) {
    const auto &r = m.rows[i];
    const auto &d = r.domain;
    if (d.site.empty() || d.runtimeIdentity.empty() ||
        d.observerIdentity.empty() || d.conditionIdentity.empty() ||
        (d.callForm != "whole-implicit" && d.callForm != "whole-explicit" &&
         d.callForm != "region-explicit") ||
        (d.observerBoundary != "minimal-api" &&
         d.observerBoundary != "detailed-compatibility" &&
         d.observerBoundary != "batch-outer") ||
        (d.direction != "TO_DEVICE" && d.direction != "FROM_DEVICE") ||
        d.minBytes <= 0 || d.maxBytes < d.minBytes || d.minCount <= 0 ||
        d.maxCount < d.minCount || d.alignmentBytes <= 0 ||
        d.allowedStates.empty() || r.provenance.empty() ||
        !digest(r.rawManifestSHA256) || !std::isfinite(r.entryNs) ||
        !std::isfinite(r.callNs) || !std::isfinite(r.cachelineNs) ||
        r.entryNs < 0 || r.callNs < 0 || r.cachelineNs < 0 ||
        (r.entryNs == 0 && r.callNs == 0 && r.cachelineNs == 0))
      return fail(e, "invalid domain, calibration evidence, units or "
                     "nonnegative coefficients");
    std::set<std::string> states;
    for (const auto &s : d.allowedStates)
      if (!knownState(s) || !states.insert(s).second)
        return fail(
            e, "allowed_states must contain distinct observable known states");
    if (r.formula == "batch-entry-plus-cachelines") {
      if (d.site != "prebound-contiguous-sync-loop" ||
          d.argumentOrdinal != -1 || d.observerBoundary != "batch-outer")
        return fail(e, "batch formula requires its preregistered loop "
                       "site/ordinal/observer");
      if (d.alignmentBytes < 64 || d.alignmentBytes % 64 != 0)
        return fail(e, "batch API-size line formula requires at least 64-byte "
                       "aligned range starts");
      for (const auto &s : d.allowedStates)
        if (s != (d.direction == "TO_DEVICE" ? "post-TO-no-access"
                                             : "post-FROM-no-access"))
          return fail(
              e, "batch formula is restricted to unchanged post-sync state");
    } else if (r.formula == "fixed-api-site") {
      if ((d.site != "installed-bound-bo-input-loop" &&
           d.site != "installed-bound-bo-readback-loop") ||
          d.argumentOrdinal < 0 || d.callForm != "whole-implicit" ||
          d.observerBoundary != "detailed-compatibility" || d.minCount != 1 ||
          d.maxCount != 1 || r.entryNs != 0 || r.cachelineNs != 0 ||
          (d.site == "installed-bound-bo-input-loop" &&
           d.direction != "TO_DEVICE") ||
          (d.site == "installed-bound-bo-readback-loop" &&
           d.direction != "FROM_DEVICE"))
        return fail(e, "fixed formula requires exact installed single-call ABI "
                       "site/ordinal and no byte slope");
    } else
      return fail(
          e, "unsupported formula; opaque compute/wait cannot be priced here");
    for (size_t j = 0; j < i; ++j) {
      const auto &other = m.rows[j].domain;
      if (key(d) != key(other) || d.minBytes > other.maxBytes ||
          other.minBytes > d.maxBytes || d.minCount > other.maxCount ||
          other.minCount > d.maxCount)
        continue;
      for (const auto &s : d.allowedStates)
        if (std::find(other.allowedStates.begin(), other.allowedStates.end(),
                      s) != other.allowedStates.end())
          return fail(e, "duplicate overlapping calibration rows");
    }
  }
  return true;
}
J rowJSON(const SyncCostRow &r) {
  const auto &d = r.domain;
  A states;
  for (const auto &s : d.allowedStates)
    states.push_back(s);
  return O{{"domain", O{{"site", d.site},
                        {"argument_ordinal", d.argumentOrdinal},
                        {"call_form", d.callForm},
                        {"observer_boundary", d.observerBoundary},
                        {"direction", d.direction},
                        {"runtime_identity", d.runtimeIdentity},
                        {"observer_identity", d.observerIdentity},
                        {"condition_identity", d.conditionIdentity},
                        {"allowed_states", std::move(states)},
                        {"min_bytes", d.minBytes},
                        {"max_bytes", d.maxBytes},
                        {"min_count", d.minCount},
                        {"max_count", d.maxCount},
                        {"alignment_bytes", d.alignmentBytes}}},
           {"formula", r.formula},
           {"entry_ns", r.entryNs},
           {"call_ns", r.callNs},
           {"cacheline_ns", r.cachelineNs},
           {"provenance", r.provenance},
           {"raw_manifest_sha256", r.rawManifestSHA256}};
}
} // namespace

bool orbit::taskflow::parseSyncAPICostModel(llvm::StringRef bytes,
                                            SyncAPICostModel &output,
                                            std::string &error) {
  auto parsed = llvm::json::parse(bytes);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return fail(error, "invalid model JSON");
  }
  auto *top = parsed->getAsObject();
  if (!top)
    return fail(error, "model must be an object");
  if (top->getString("schema") != "orbit.sync-api.model.v14" ||
      top->getString("units") != "ns")
    return fail(error, "unsupported schema or units (required ns)");
  SyncAPICostModel m;
  auto id = top->getString("model_id");
  auto *rows = top->getArray("rows");
  if (!id || !rows)
    return fail(error, "missing model_id or rows");
  m.modelId = id->str();
  llvm::SHA256 sha;
  sha.update(bytes);
  m.parameterFileSHA256 = llvm::toHex(sha.final(), true);
  for (const auto &value : *rows) {
    auto *o = value.getAsObject();
    if (!o)
      return fail(error, "row must be an object");
    auto *d = o->getObject("domain");
    if (!d)
      return fail(error, "row missing domain");
    SyncCostRow r;
    const std::array<std::pair<const char *, std::string *>, 7> domainStrings =
        {{{"site", &r.domain.site},
          {"call_form", &r.domain.callForm},
          {"observer_boundary", &r.domain.observerBoundary},
          {"direction", &r.domain.direction},
          {"runtime_identity", &r.domain.runtimeIdentity},
          {"observer_identity", &r.domain.observerIdentity},
          {"condition_identity", &r.domain.conditionIdentity}}};
    for (const auto &item : domainStrings) {
      auto v = d->getString(item.first);
      if (!v)
        return fail(error, "domain missing string " + std::string(item.first));
      *item.second = v->str();
    }
    const std::array<std::pair<const char *, int64_t *>, 6> domainIntegers = {
        {{"argument_ordinal", &r.domain.argumentOrdinal},
         {"min_bytes", &r.domain.minBytes},
         {"max_bytes", &r.domain.maxBytes},
         {"min_count", &r.domain.minCount},
         {"max_count", &r.domain.maxCount},
         {"alignment_bytes", &r.domain.alignmentBytes}}};
    for (const auto &item : domainIntegers) {
      auto v = d->getInteger(item.first);
      if (!v)
        return fail(error, "domain missing integer " + std::string(item.first));
      *item.second = *v;
    }
    auto *states = d->getArray("allowed_states");
    if (!states)
      return fail(error, "missing allowed_states");
    for (const auto &v : *states) {
      auto s = v.getAsString();
      if (!s)
        return fail(error, "state must be a string");
      r.domain.allowedStates.push_back(s->str());
    }
    const std::array<std::pair<const char *, std::string *>, 3> rowStrings = {
        {{"formula", &r.formula},
         {"provenance", &r.provenance},
         {"raw_manifest_sha256", &r.rawManifestSHA256}}};
    for (const auto &item : rowStrings) {
      auto v = o->getString(item.first);
      if (!v)
        return fail(error, "row missing string " + std::string(item.first));
      *item.second = v->str();
    }
    const std::array<std::pair<const char *, double *>, 3> coefficients = {
        {{"entry_ns", &r.entryNs},
         {"call_ns", &r.callNs},
         {"cacheline_ns", &r.cachelineNs}}};
    for (const auto &item : coefficients) {
      auto v = o->getNumber(item.first);
      if (!v)
        return fail(error, "row missing numeric " + std::string(item.first));
      *item.second = *v;
    }
    m.rows.push_back(std::move(r));
  }
  if (!validate(m, error))
    return false;
  output = std::move(m);
  return true;
}

bool orbit::taskflow::evaluateSyncAPICost(const SyncAPICostModel &model,
                                          const SyncCostRequest &request,
                                          SyncAPICostEvaluation &out,
                                          std::string &error) {
  out = {};
  out.modelId = model.modelId;
  out.parameterFileSHA256 = model.parameterFileSHA256;
  if (!validate(model, error))
    return false;
  const auto &q = request.query;
  const auto &r = q.range;
  if (request.observedExtentBytes <= 0 || r.offset < 0 || r.bytes <= 0 ||
      r.offset > request.observedExtentBytes ||
      r.bytes > request.observedExtentBytes - r.offset || request.count <= 0 ||
      request.site.empty() || !q.context.callForm ||
      !q.context.observerBoundary ||
      (request.direction != SyncDirection::ToDevice &&
       request.direction != SyncDirection::FromDevice))
    return fail(error, "request requires observed extent, valid range/count, "
                       "site and explicit call/observer context");
  bool full = r.offset == 0 && r.bytes == request.observedExtentBytes;
  if (*q.context.callForm == SyncCallForm::ImplicitFullBuffer) {
    if (!full)
      return fail(error,
                  "implicit invocation is not the observed full BO range");
    out.callForm = "whole-implicit";
  } else if (*q.context.callForm == SyncCallForm::ExplicitRange)
    out.callForm = full ? "whole-explicit" : "region-explicit";
  else
    return fail(error, "invalid invocation form");
  auto boundary =
      std::string(syncObserverBoundaryName(*q.context.observerBoundary));
  if (boundary == "unknown")
    return fail(error, "invalid observer boundary");
  if (request.conditionIdentity.empty()) {
    out.reason = "execution condition identity is unobserved/undeclared";
    return true;
  }
  if (!q.known || !q.homogeneous || q.pending || !knownState(q.state)) {
    out.reason = "state is unknown, mixed, pending or unsupported";
    return true;
  }
  // Queries normally originate in SyncState. Keep structural evidence checks
  // fail-closed when crossing a serialized/public cost-provider interface.
  int64_t end = r.offset;
  bool producer = true;
  if (q.segments.empty()) {
    out.reason = "known state has no range evidence";
    return true;
  }
  auto stateKey = [](const SyncRangeState &s) {
    return std::make_tuple(s.known, s.hostVisible, s.deviceVisible,
                           s.contentEpoch, s.hostWriteEpoch, s.hostReadEpoch,
                           s.deviceReadEpoch, s.deviceWriteEpoch,
                           s.toCompleteEpoch, s.fromCompleteEpoch);
  };
  for (const auto &s : q.segments) {
    if (!s.known || s.range.bo != r.bo || s.range.generation != r.generation ||
        stateKey(s) != stateKey(q.segments.front()) || s.range.offset != end ||
        s.range.bytes <= 0 || s.range.bytes > r.offset + r.bytes - end) {
      out.reason = "range evidence does not cover one known queried interval";
      return true;
    }
    end += s.range.bytes;
    producer &= s.deviceWriteEpoch > 0 &&
                s.contentEpoch == s.deviceWriteEpoch && s.deviceVisible;
  }
  if (end != r.offset + r.bytes) {
    out.reason = "range evidence is incomplete";
    return true;
  }
  for (size_t i = 0; i < model.rows.size(); ++i) {
    const auto &row = model.rows[i];
    const auto &d = row.domain;
    if (d.site != request.site ||
        d.argumentOrdinal != request.argumentOrdinal ||
        d.callForm != out.callForm || d.observerBoundary != boundary ||
        d.direction != (request.direction == SyncDirection::ToDevice
                            ? "TO_DEVICE"
                            : "FROM_DEVICE") ||
        d.runtimeIdentity != q.context.runtimeIdentity ||
        d.observerIdentity != q.context.observerIdentity ||
        d.conditionIdentity != request.conditionIdentity ||
        r.bytes < d.minBytes || r.bytes > d.maxBytes ||
        request.count < d.minCount || request.count > d.maxCount ||
        r.bytes % d.alignmentBytes || r.offset % d.alignmentBytes ||
        std::find(d.allowedStates.begin(), d.allowedStates.end(), q.state) ==
            d.allowedStates.end())
      continue;
    if (request.direction == SyncDirection::FromDevice && !producer) {
      out.reason =
          "FROM calibration requires an observed successful device producer";
      return true;
    }
    // ceil division avoids signed overflow at INT64_MAX. The coefficient is an
    // API size term, not a measurement of physical cacheline transfers.
    int64_t lines = r.bytes / 64 + (r.bytes % 64 != 0);
    double value = double(request.count) * row.callNs;
    if (row.formula == "batch-entry-plus-cachelines")
      value = row.entryNs + double(request.count) *
                                (row.callNs + row.cachelineNs * double(lines));
    if (!std::isfinite(value)) {
      out.reason = "finite API estimate overflow";
      return true;
    }
    out.durationNs = value;
    out.matchedRow = i;
    out.status = "predicted";
    out.reason = "calibrated API-observed boundary only; no "
                 "compute/wait/internal-DMA or physical-traffic charge";
    return true;
  }
  out.reason = "no calibrated row matches exact site, identities, form, state, "
               "size, count and alignment";
  return true;
}

llvm::json::Value
orbit::taskflow::serializeSyncAPICostModel(const SyncAPICostModel &m) {
  A rows;
  for (const auto &r : m.rows)
    rows.push_back(rowJSON(r));
  return O{{"schema", "orbit.sync-api.model.v14"},
           {"model_id", m.modelId},
           {"units", "ns"},
           {"parameter_file_sha256", m.parameterFileSHA256},
           {"rows", std::move(rows)},
           {"covered_cost", "boundary-api-observed"},
           {"excluded_costs",
            A{"internal-DMA", "compute", "wait", "physical-DDR-traffic"}}};
}
llvm::json::Value
orbit::taskflow::serializeSyncCostRequest(const SyncCostRequest &r) {
  return O{{"site", r.site},
           {"argument_ordinal", r.argumentOrdinal},
           {"count", r.count},
           {"condition_identity", r.conditionIdentity},
           {"observed_extent_bytes", r.observedExtentBytes},
           {"direction", r.direction == SyncDirection::ToDevice
                             ? "TO_DEVICE"
                             : "FROM_DEVICE"},
           {"query", serializeSyncQuery(r.query)}};
}
llvm::json::Value orbit::taskflow::serializeSyncAPICostEvaluation(
    const SyncAPICostEvaluation &e) {
  return O{
      {"schema", "orbit.sync-api.cost.v14"},
      {"model_id", e.modelId},
      {"parameter_file_sha256", e.parameterFileSHA256},
      {"status", e.status},
      {"reason", e.reason},
      {"call_form", e.callForm},
      {"units", "ns"},
      {"duration_ns", e.durationNs ? J(*e.durationNs) : J(nullptr)},
      {"matched_row", e.matchedRow ? J(int64_t(*e.matchedRow)) : J(nullptr)},
      {"physical_bytes", nullptr},
      {"complete_program_cost", false},
      {"covered_cost", "boundary-api-observed"},
      {"excluded_costs",
       A{"internal-DMA", "compute", "wait", "physical-DDR-traffic"}}};
}
