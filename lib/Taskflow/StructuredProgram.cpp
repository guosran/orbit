#include "Taskflow/StructuredProgram.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <set>
#include <sstream>
using namespace orbit::taskflow;
namespace {
bool fail(std::string &e, const std::string &s) {
  e = s;
  return false;
}
bool name(const std::string &s) {
  return !s.empty() &&
         (std::isalpha(static_cast<unsigned char>(s[0])) || s[0] == '_') &&
         std::all_of(s.begin(), s.end(), [](char c) {
           return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
         });
}
bool onlyKeys(const llvm::json::Object &object,
              std::initializer_list<llvm::StringRef> allowed) {
  for (const auto &entry : object)
    if (std::find(allowed.begin(), allowed.end(), entry.first.str()) ==
        allowed.end())
      return false;
  return true;
}
std::string str(const llvm::json::Object &o, llvm::StringRef k) {
  auto s = o.getString(k);
  return s ? s->str() : "";
}
bool strings(const llvm::json::Object &o, llvm::StringRef k,
             std::vector<std::string> &out) {
  auto a = o.getArray(k);
  if (!a)
    return false;
  for (auto &v : *a) {
    auto s = v.getAsString();
    if (!s)
      return false;
    out.push_back(s->str());
  }
  return true;
}
bool range(const llvm::json::Object &o, llvm::StringRef k, Range &r) {
  auto a = o.getArray(k);
  if (!a || a->size() != 2)
    return false;
  auto l = (*a)[0].getAsInteger(), h = (*a)[1].getAsInteger();
  if (!l || !h)
    return false;
  r = {*l, *h};
  return true;
}
llvm::json::Array jsonStrings(const std::vector<std::string> &s) {
  llvm::json::Array a;
  for (auto &v : s)
    a.push_back(v);
  return a;
}
struct Rect {
  Range m, n;
};
bool overlap(Rect a, Rect b) {
  return a.m.lo < b.m.hi && b.m.lo < a.m.hi && a.n.lo < b.n.hi &&
         b.n.lo < a.n.hi;
}
bool equal(Rect a, Rect b) { return a.m == b.m && a.n == b.n; }
std::vector<Rect> subtract(Rect a, Rect b) {
  if (!overlap(a, b))
    return {a};
  Range m{std::max(a.m.lo, b.m.lo), std::min(a.m.hi, b.m.hi)},
      n{std::max(a.n.lo, b.n.lo), std::min(a.n.hi, b.n.hi)};
  std::vector<Rect> r;
  auto add = [&](Range x, Range y) {
    if (x.lo < x.hi && y.lo < y.hi)
      r.push_back({x, y});
  };
  add({a.m.lo, m.lo}, a.n);
  add({m.hi, a.m.hi}, a.n);
  add(m, {a.n.lo, n.lo});
  add(m, {n.hi, a.n.hi});
  return r;
}
bool covered(Rect r, const std::vector<Rect> &regions) {
  std::vector<Rect> left{r};
  for (auto b : regions) {
    std::vector<Rect> next;
    for (auto a : left) {
      auto p = subtract(a, b);
      next.insert(next.end(), p.begin(), p.end());
    }
    left = std::move(next);
  }
  return left.empty();
}
const Buffer &buffer(const Program &p, const std::string &id) {
  return *std::find_if(p.buffers.begin(), p.buffers.end(),
                       [&](auto &b) { return b.id == id; });
}
std::string type(const Buffer &b) {
  return "memref<" + std::to_string(b.rows) + "x" + std::to_string(b.cols) +
         "xf32>";
}
} // namespace
bool orbit::taskflow::parsePlan(const llvm::json::Value &v, Program &p,
                                std::string &e) {
  auto o = v.getAsObject();
  if (!o || str(*o, "schema") != "orbit.taskflow.plan.v1")
    return fail(e, "unsupported plan schema (expected orbit.taskflow.plan.v1)");
  if (!onlyKeys(*o, {"schema", "source_id", "candidate_id", "transforms",
                     "buffers", "tasks", "outputs"}))
    return fail(e, "unsupported top-level plan field");
  p.sourceId = str(*o, "source_id");
  p.candidateId = str(*o, "candidate_id");
  if (auto a = o->getArray("transforms"))
    p.transforms = *a;
  auto bs = o->getArray("buffers"), ts = o->getArray("tasks");
  if (!bs || !ts || !strings(*o, "outputs", p.outputs))
    return fail(e, "buffers, tasks and outputs arrays required");
  for (auto &x : *bs) {
    auto b = x.getAsObject();
    Range shape;
    if (!b || !range(*b, "shape", shape))
      return fail(e, "buffer shape must be two static integers");
    if (!onlyKeys(*b, {"id", "shape", "dtype", "role"}))
      return fail(e, "unsupported buffer field");
    p.buffers.push_back(
        {str(*b, "id"), str(*b, "dtype"), str(*b, "role"), shape.lo, shape.hi});
  }
  for (auto &x : *ts) {
    auto t = x.getAsObject();
    Task task;
    if (!t || !strings(*t, "depends_on", task.dependsOn))
      return fail(e, "task depends_on array required");
    if (!onlyKeys(*t, {"id", "depends_on", "operations"}))
      return fail(e, "unsupported task field");
    task.id = str(*t, "id");
    auto ops = t->getArray("operations");
    if (!ops)
      return fail(e, "task operations required");
    for (auto &y : *ops) {
      auto c = y.getAsObject();
      Compute op;
      if (!c || !strings(*c, "inputs", op.inputs) || !range(*c, "m", op.m) ||
          !range(*c, "n", op.n))
        return fail(e, "operation inputs and m/n ranges required");
      op.kind = str(*c, "kind");
      op.output = str(*c, "output");
      if (op.kind == "matmul"
              ? !onlyKeys(*c, {"kind", "inputs", "output", "m", "n", "k",
                               "accumulate"})
              : !onlyKeys(*c, {"kind", "inputs", "output", "m", "n"}))
        return fail(e, "unsupported operation field");
      if (op.kind == "matmul") {
        auto acc = c->getBoolean("accumulate");
        if (!range(*c, "k", op.k) || !acc)
          return fail(e, "matmul requires k and accumulate");
        op.accumulate = *acc;
      }
      task.operations.push_back(std::move(op));
    }
    p.tasks.push_back(std::move(task));
  }
  return validate(p, e);
}
llvm::json::Value orbit::taskflow::printPlan(const Program &p) {
  llvm::json::Array bs, ts;
  for (auto &b : p.buffers)
    bs.push_back(
        llvm::json::Object{{"id", b.id},
                           {"shape", llvm::json::Array{b.rows, b.cols}},
                           {"dtype", b.dtype},
                           {"role", b.role}});
  for (auto &t : p.tasks) {
    llvm::json::Array ops;
    for (auto &c : t.operations) {
      llvm::json::Object o{{"kind", c.kind},
                           {"inputs", jsonStrings(c.inputs)},
                           {"output", c.output},
                           {"m", llvm::json::Array{c.m.lo, c.m.hi}},
                           {"n", llvm::json::Array{c.n.lo, c.n.hi}}};
      if (c.kind == "matmul") {
        o["k"] = llvm::json::Array{c.k.lo, c.k.hi};
        o["accumulate"] = c.accumulate;
      }
      ops.push_back(std::move(o));
    }
    ts.push_back(llvm::json::Object{{"id", t.id},
                                    {"depends_on", jsonStrings(t.dependsOn)},
                                    {"operations", std::move(ops)}});
  }
  return llvm::json::Object{{"schema", "orbit.taskflow.plan.v1"},
                            {"source_id", p.sourceId},
                            {"candidate_id", p.candidateId},
                            {"transforms", llvm::json::Array(p.transforms)},
                            {"buffers", std::move(bs)},
                            {"tasks", std::move(ts)},
                            {"outputs", jsonStrings(p.outputs)}};
}
bool orbit::taskflow::validate(const Program &p, std::string &e) {
  if (p.buffers.empty() || p.tasks.empty() || p.outputs.empty())
    return fail(e, "empty programs are unsupported");
  std::map<std::string, const Buffer *> bs;
  for (auto &b : p.buffers) {
    if (!name(b.id) || !bs.emplace(b.id, &b).second || b.rows <= 0 ||
        b.cols <= 0 || b.rows > std::numeric_limits<int64_t>::max() / b.cols ||
        b.dtype != "f32" ||
        (b.role != "input" && b.role != "temporary" && b.role != "output"))
      return fail(e,
                  "invalid/duplicate buffer or unsupported shape/dtype/role: " +
                      b.id);
  }
  std::map<std::string, std::set<std::string>> ancestors;
  std::map<std::string, std::vector<Rect>> initialized;
  for (auto &b : p.buffers)
    if (b.role == "input")
      initialized[b.id].push_back({{0, b.rows}, {0, b.cols}});
  // Rectangle subtraction proves coverage without allocating an element-sized
  // bitmap (the same verifier handles full Qwen FFN dimensions). Dependence
  // ancestry is checked independently: textual task order is not a dependency.
  struct Access {
    std::string task, storage;
    Rect region;
    bool write;
  };
  std::vector<Access> accesses;
  // A reduction has exactly one zero initialization and monotonically
  // increasing K intervals. No parallel partial sums or reassociation are
  // admitted.
  struct Reduction {
    std::string storage;
    Rect region;
    int64_t next = 0, total = 0;
    bool started = false;
  };
  std::vector<Reduction> reductions;
  for (auto &t : p.tasks) {
    if (!name(t.id) || ancestors.count(t.id) || t.operations.empty())
      return fail(e, "invalid/duplicate task or empty body: " + t.id);
    std::set<std::string> deps;
    for (auto &d : t.dependsOn) {
      auto it = ancestors.find(d);
      if (it == ancestors.end())
        return fail(e, "dependency must reference an earlier task: " + d);
      deps.insert(d);
      deps.insert(it->second.begin(), it->second.end());
    }
    ancestors[t.id] = deps;
    auto access = [&](const std::string &id, Rect r, bool write) -> bool {
      for (auto &a : accesses)
        if (a.task != t.id && a.storage == id && (a.write || write) &&
            overlap(a.region, r) && !deps.count(a.task))
          return fail(e, "missing RAW/WAR/WAW dependency: " + a.task + " -> " +
                             t.id + " on " + id);
      if (!write && !covered(r, initialized[id]))
        return fail(e,
                    "read before full initialization: " + id + " in " + t.id);
      accesses.push_back({t.id, id, r, write});
      if (write)
        initialized[id].push_back(r);
      return true;
    };
    for (auto &c : t.operations) {
      if (!bs.count(c.output))
        return fail(e, "unknown output storage: " + c.output);
      auto &out = *bs[c.output];
      Rect r{c.m, c.n};
      if (out.role == "input" || c.m.lo < 0 || c.n.lo < 0 || c.m.hi <= c.m.lo ||
          c.n.hi <= c.n.lo || c.m.hi > out.rows || c.n.hi > out.cols)
        return fail(e, "invalid output slice or write to input: " + c.output);
      if (c.kind != "fill" && c.kind != "matmul" && c.kind != "relu" &&
          c.kind != "swiglu")
        return fail(e, "unsupported operation: " + c.kind);
      unsigned expected = c.kind == "fill"                             ? 0
                          : c.kind == "relu"                           ? 1
                          : (c.kind == "matmul" || c.kind == "swiglu") ? 2
                                                                       : 99;
      if (c.inputs.size() != expected)
        return fail(e, "unsupported operation or arity: " + c.kind);
      for (auto &id : c.inputs)
        if (!bs.count(id))
          return fail(e, "unknown input storage: " + id);
      if (c.kind == "fill") {
        for (auto &q : reductions)
          if (q.storage == c.output && overlap(q.region, r))
            return fail(e,
                        "overlapping or repeated reduction initialization: " +
                            c.output);
        reductions.push_back({c.output, r});
      }
      if (c.kind == "matmul") {
        auto &a = *bs[c.inputs[0]], &b = *bs[c.inputs[1]];
        if (c.inputs[0] == c.output || c.inputs[1] == c.output ||
            a.cols != b.rows || a.rows != out.rows || b.cols != out.cols ||
            !c.accumulate || c.k.lo < 0 || c.k.hi <= c.k.lo || c.k.hi > a.cols)
          return fail(e, "unsupported matmul dimensions, alias, bounds or "
                         "non-accumulating contract");
        auto it =
            std::find_if(reductions.begin(), reductions.end(), [&](auto &q) {
              return q.storage == c.output && equal(q.region, r);
            });
        if (it == reductions.end() || it->next != c.k.lo ||
            (it->started && it->total != a.cols))
          return fail(e, "matmul needs exactly one matching fill and "
                         "contiguous increasing K slices");
        it->started = true;
        it->next = c.k.hi;
        it->total = a.cols;
        if (!access(c.inputs[0], {c.m, c.k}, false) ||
            !access(c.inputs[1], {c.k, c.n}, false) ||
            !access(c.output, r, false))
          return false;
      } else if (c.kind != "fill") {
        for (auto &id : c.inputs) {
          auto &b = *bs[id];
          if (b.rows != out.rows || b.cols != out.cols)
            return fail(e, "elementwise shape mismatch");
          for (auto &q : reductions)
            if (q.storage == id && overlap(q.region, r) && q.started &&
                q.next != q.total)
              return fail(e, "consumer reads an incomplete reduction");
          if (!access(id, r, false))
            return false;
        }
      }
      if (c.kind == "matmul")
        for (auto &id : c.inputs)
          for (auto &q : reductions)
            if (q.storage == id && q.started && q.next != q.total)
              return fail(e, "matmul consumes incomplete reduction");
      if (!access(c.output, r, true))
        return false;
    }
  }
  for (auto &q : reductions)
    if (q.started && q.next != q.total)
      return fail(e, "incomplete K reduction on " + q.storage);
  std::set<std::string> outputs;
  for (auto &id : p.outputs) {
    if (!bs.count(id) || !outputs.insert(id).second ||
        bs[id]->role != "output" ||
        !covered({{0, bs[id]->rows}, {0, bs[id]->cols}}, initialized[id]))
      return fail(e, "output not fully covered or invalid: " + id);
  }
  return true;
}
std::string orbit::taskflow::fingerprint(const Program &p) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << printPlan(p);
  auto digest = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(s.data()), s.size()));
  static const char hex[] = "0123456789abcdef";
  std::string out;
  for (auto c : digest) {
    out += hex[c >> 4];
    out += hex[c & 15];
  }
  return out;
}
bool orbit::taskflow::transform(Program &p, const TransformOptions &o,
                                std::string &e) {
  if (!validate(p, e))
    return false;
  if (o.tileM < 0 || o.tileN < 0 || o.tileK < 0 || (o.fuse && o.fission))
    return fail(
        e, "negative tile sizes or simultaneous fuse/fission are unsupported");
  if (p.sourceId.empty())
    p.sourceId = fingerprint(p);
  std::vector<Task> stages;
  std::map<std::string, std::vector<std::string>> mapped;
  // Task fusion retains both real computations and their storage identities in
  // one isolated body. Its inverse below splits the trailing elementwise stage.
  if (o.fuse) {
    std::set<std::string> removed;
    std::map<std::string, std::string> alias;
    for (size_t i = 0; i < p.tasks.size(); ++i) {
      auto &producer = p.tasks[i];
      if (producer.operations.empty() ||
          producer.operations.back().kind != "matmul")
        continue;
      for (size_t j = i + 1; j < p.tasks.size(); ++j) {
        auto &consumer = p.tasks[j];
        if (consumer.dependsOn.size() != 1 ||
            consumer.dependsOn[0] != producer.id ||
            consumer.operations.size() != 1)
          continue;
        auto &a = producer.operations.back();
        auto &b = consumer.operations[0];
        if (b.kind != "relu" || b.inputs[0] != a.output || !(a.m == b.m) ||
            !(a.n == b.n))
          continue;
        unsigned uses = 0;
        for (auto &task : p.tasks)
          for (auto &d : task.dependsOn)
            uses += d == producer.id;
        if (uses != 1)
          return fail(e, "fusion requires a single consumer");
        producer.operations.push_back(b);
        removed.insert(consumer.id);
        alias[consumer.id] = producer.id;
        break;
      }
    }
    if (removed.empty())
      return fail(e, "no supported matmul -> relu edge to fuse");
    for (auto &t : p.tasks)
      if (!removed.count(t.id)) {
        for (auto &d : t.dependsOn)
          if (alias.count(d))
            d = alias[d];
        stages.push_back(t);
      }
    p.tasks = std::move(stages);
    p.transforms.push_back(llvm::json::Object{{"kind", "fuse_matmul_relu"}});
  }
  if (o.fission) {
    bool changed = false;
    stages.clear();
    mapped.clear();
    for (auto &orig : p.tasks) {
      Task t = orig;
      std::vector<std::string> deps;
      for (auto &d : t.dependsOn) {
        auto it = mapped.find(d);
        if (it != mapped.end())
          deps.insert(deps.end(), it->second.begin(), it->second.end());
        else
          deps.push_back(d);
      }
      t.dependsOn = deps;
      if (t.operations.size() > 1 && t.operations.back().kind == "relu" &&
          t.operations[t.operations.size() - 2].kind == "matmul") {
        Compute relu = t.operations.back();
        t.operations.pop_back();
        Task consumer{t.id + "_relu", {t.id}, {relu}};
        mapped[t.id] = {consumer.id};
        stages.push_back(t);
        stages.push_back(consumer);
        changed = true;
      } else {
        mapped[t.id] = {t.id};
        stages.push_back(t);
      }
    }
    if (!changed)
      return fail(e, "fission supports only trailing relu after matmul");
    p.tasks = std::move(stages);
    p.transforms.push_back(llvm::json::Object{{"kind", "fission_matmul_relu"}});
  }
  // M/N create task instances; K creates sequential affine loops inside each
  // task. Consumer tasks conservatively depend on every producer tile. These
  // completion edges never perform, or stand in for, a numerical reduction.
  if (o.tileM || o.tileN || o.tileK) {
    stages.clear();
    mapped.clear();
    for (auto &orig : p.tasks) {
      Rect region{orig.operations.front().m, orig.operations.front().n};
      for (auto &c : orig.operations)
        if (!(c.m == region.m) || !(c.n == region.n))
          return fail(
              e,
              "task tiling requires operations with a common output rectangle");
      int64_t tm = o.tileM ? o.tileM : region.m.hi - region.m.lo,
              tn = o.tileN ? o.tileN : region.n.hi - region.n.lo;
      std::vector<std::string> deps;
      for (auto &d : orig.dependsOn) {
        auto &children = mapped[d];
        deps.insert(deps.end(), children.begin(), children.end());
      }
      for (int64_t m = region.m.lo; m < region.m.hi;
           m += std::min(tm, region.m.hi - m))
        for (int64_t n = region.n.lo; n < region.n.hi;
             n += std::min(tn, region.n.hi - n)) {
          Task t{orig.id + "_m" + std::to_string(m) + "_n" + std::to_string(n),
                 deps,
                 {}};
          for (auto c : orig.operations) {
            c.m = {m, m + std::min(tm, region.m.hi - m)};
            c.n = {n, n + std::min(tn, region.n.hi - n)};
            if (c.kind == "matmul" && o.tileK) {
              for (int64_t k = c.k.lo; k < c.k.hi;
                   k += std::min(o.tileK, c.k.hi - k)) {
                auto part = c;
                part.k = {k, k + std::min(o.tileK, c.k.hi - k)};
                t.operations.push_back(part);
              }
            } else
              t.operations.push_back(c);
          }
          mapped[orig.id].push_back(t.id);
          stages.push_back(t);
        }
    }
    p.tasks = std::move(stages);
    p.transforms.push_back(llvm::json::Object{{"kind", "tile"},
                                              {"tile_m", o.tileM},
                                              {"tile_n", o.tileN},
                                              {"tile_k", o.tileK}});
  }
  p.candidateId.clear();
  p.candidateId = fingerprint(p);
  return validate(p, e);
}
std::string orbit::taskflow::emitTaskflow(const Program &p) {
  std::ostringstream s;
  s << "module {\n  func.func @main(";
  for (size_t i = 0; i < p.buffers.size(); ++i) {
    auto &b = p.buffers[i];
    if (i)
      s << ", ";
    s << "%" << b.id << ": " << type(b);
  }
  s << ") attributes {llvm.emit_c_interface} {\n";
  // Buffers are distinct externally allocated storage objects. Every task uses
  // original storage roots plus explicit value-completion dependencies. The
  // completion value is not a numeric result and never merges matrix elements.
  std::map<std::string, std::string> token;
  unsigned serial = 0;
  for (auto &t : p.tasks) {
    std::vector<std::string> writes, reads;
    auto add = [](auto &v, const std::string &x) {
      if (std::find(v.begin(), v.end(), x) == v.end())
        v.push_back(x);
    };
    for (auto &c : t.operations)
      add(writes, c.output);
    for (auto &c : t.operations)
      for (auto &id : c.inputs)
        if (std::find(writes.begin(), writes.end(), id) == writes.end())
          add(reads, id);
    std::vector<std::string> operands = reads;
    operands.insert(operands.end(), writes.begin(), writes.end());
    auto list = [&](const std::vector<std::string> &ids, bool types,
                    const std::string &prefix) {
      for (size_t i = 0; i < ids.size(); ++i) {
        if (i)
          s << ", ";
        s << (types ? type(buffer(p, ids[i])) : prefix + ids[i]);
      }
    };
    s << "    %" << t.id << ":" << writes.size() + 1 << " = taskflow.task @"
      << t.id;
    if (!reads.empty()) {
      s << " will_reads(";
      list(reads, false, "%");
      s << " : ";
      list(reads, true, "");
      s << ")";
    }
    s << " will_writes(";
    list(writes, false, "%");
    s << " : ";
    list(writes, true, "");
    s << ")";
    if (!t.dependsOn.empty()) {
      s << " value_inputs(";
      for (size_t i = 0; i < t.dependsOn.size(); ++i) {
        if (i)
          s << ", ";
        s << token[t.dependsOn[i]];
      }
      s << " : ";
      for (size_t i = 0; i < t.dependsOn.size(); ++i) {
        if (i)
          s << ", ";
        s << "i1";
      }
      s << ")";
    }
    s << " [";
    if (!reads.empty()) {
      s << "original_read_memrefs(";
      list(reads, false, "%");
      s << " : ";
      list(reads, true, "");
      s << "), ";
    }
    s << "original_write_memrefs(";
    list(writes, false, "%");
    s << " : ";
    list(writes, true, "");
    s << ")] : (";
    list(operands, true, "");
    for (size_t i = 0; i < t.dependsOn.size(); ++i)
      s << ", i1";
    s << ") -> (";
    list(writes, true, "");
    s << ", i1) {\n    ^bb0(";
    for (size_t i = 0; i < operands.size(); ++i) {
      if (i)
        s << ", ";
      s << "%b_" << operands[i] << ": " << type(buffer(p, operands[i]));
    }
    for (size_t i = 0; i < t.dependsOn.size(); ++i)
      s << ", %dep" << i << ": i1";
    s << "):\n";
    s << "      %zero = arith.constant 0.0 : f32\n      %one = arith.constant "
         "1.0 : f32\n      %done = arith.constant true\n";
    for (auto &c : t.operations) {
      auto id = std::to_string(serial++);
      std::string i = "%i" + id, j = "%j" + id, k = "%k" + id;
      s << "      affine.for " << i << " = " << c.m.lo << " to " << c.m.hi
        << " {\n        affine.for " << j << " = " << c.n.lo << " to " << c.n.hi
        << " {\n";
      auto load = [&](std::string v, const std::string &b, std::string x,
                      std::string y) {
        s << "          %" << v << id << " = affine.load %b_" << b << "[" << x
          << ", " << y << "] : " << type(buffer(p, b)) << "\n";
      };
      auto store = [&](std::string v) {
        s << "          affine.store " << v << ", %b_" << c.output << "[" << i
          << ", " << j << "] : " << type(buffer(p, c.output)) << "\n";
      };
      if (c.kind == "fill")
        store("%zero");
      if (c.kind == "matmul") {
        s << "          affine.for " << k << " = " << c.k.lo << " to " << c.k.hi
          << " {\n";
        load("a", c.inputs[0], i, k);
        load("b", c.inputs[1], k, j);
        load("old", c.output, i, j);
        s << "          %mul" << id << " = arith.mulf %a" << id << ", %b" << id
          << " : f32\n          %sum" << id << " = arith.addf %old" << id
          << ", %mul" << id << " : f32\n";
        store("%sum" + id);
        s << "          }\n";
      }
      if (c.kind == "relu") {
        load("a", c.inputs[0], i, j);
        s << "          %relu" << id << " = arith.maximumf %a" << id
          << ", %zero : f32\n";
        store("%relu" + id);
      }
      if (c.kind == "swiglu") {
        load("a", c.inputs[0], i, j);
        load("b", c.inputs[1], i, j);
        s << "          %neg" << id << " = arith.negf %a" << id
          << " : f32\n          %exp" << id << " = math.exp %neg" << id
          << " : f32\n          %den" << id << " = arith.addf %one, %exp" << id
          << " : f32\n          %silu" << id << " = arith.divf %a" << id
          << ", %den" << id << " : f32\n          %swiglu" << id
          << " = arith.mulf %silu" << id << ", %b" << id << " : f32\n";
        store("%swiglu" + id);
      }
      s << "        }\n      }\n";
    }
    s << "      taskflow.yield done_writes(";
    list(writes, false, "%b_");
    s << " : ";
    list(writes, true, "");
    s << ") values(%done : i1)\n    }\n";
    token[t.id] = "%" + t.id + "#" + std::to_string(writes.size());
  }
  s << "    return\n  }\n}\n";
  return s.str();
}
