#ifndef ORBIT_TASKFLOW_SEARCH_H
#define ORBIT_TASKFLOW_SEARCH_H
#include "Taskflow/StructuredProgram.h"
#include "Taskflow/GenericMapping.h"
#include <algorithm>
#include <optional>
#include <set>
namespace orbit::taskflow {
// Extracted from JointNeighborhoodSearchPass::selectBeam. Both legacy Neura
// and structured search share scored-only, unique-key, diversity selection.
template<class State, class Less, class Scored, class Key, class Configuration>
std::vector<State> selectSearchBeam(std::vector<State> candidates, int64_t width,
    int64_t diversity, Less less, Scored scored, Key key, Configuration configuration) {
  std::stable_sort(candidates.begin(), candidates.end(), less);
  std::vector<size_t> indices;
  std::set<std::string> keys, configurations;
  const size_t costSlots = size_t(std::max<int64_t>(0, width-diversity));
  for (size_t i=0; i<candidates.size() && indices.size()<costSlots; ++i) {
    if (!scored(candidates[i]) || key(candidates[i]).empty() || !keys.insert(key(candidates[i])).second) continue;
    indices.push_back(i); configurations.insert(configuration(candidates[i]));
  }
  for (size_t i=0; i<candidates.size() && indices.size()<size_t(width); ++i) {
    if (!scored(candidates[i]) || key(candidates[i]).empty() || keys.count(key(candidates[i])) || !configurations.insert(configuration(candidates[i])).second) continue;
    keys.insert(key(candidates[i])); indices.push_back(i);
  }
  for (size_t i=0; i<candidates.size() && indices.size()<size_t(width); ++i) {
    if (!scored(candidates[i]) || key(candidates[i]).empty() || !keys.insert(key(candidates[i])).second) continue;
    indices.push_back(i);
  }
  std::vector<State> selected;
  for (auto i:indices) selected.push_back(std::move(candidates[i]));
  return selected;
}
struct SearchOptions {
  std::string backend, evaluator, outputDir, targetConfigFile, checkpoint;
  int64_t maxRounds=2, maxCandidates=9, beamWidth=1;
  bool resume=false;
  std::string nativeSourceMLIR; // optional actual-body candidates; same search driver

};
struct SearchAction {
  std::string name;
  TransformOptions options;
  std::optional<ProgramMappingChoice> mapping;
  std::string implementation;
  int64_t vectorLanes = 1;
  int64_t partitionExtent = -1; // -1 unchanged, 0 single command, >0 program fission
};
// Target recommendations affect proposals only, never semantic legality.
struct SearchPreferences { int64_t m=1, n=1, k=1; };
std::vector<SearchAction> generateSearchActions(const Program &, const SearchPreferences & = {});
std::string canonicalSearchIdentity(const Program &);
bool runStructuredSearch(const Program &, const SearchOptions &, std::string &error);
}
#endif
