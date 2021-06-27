#pragma once
#include <map>
#include <vector>
#include <utility>
#include "_main.hxx"
#include "vertices.hxx"

using std::pair;
using std::map;
using std::vector;
using std::make_pair;
using std::move;




template <class G, class H>
auto inIdenticals(const G& x, const H& xt) {
  map<size_t, pair<vector<int>, int>> m;
  auto vis = createContainer(x, bool());
  vector<int> es;
  vector2d<int> a;
  // For all vertices.
  for (int u : x.vertices()) {
    if (x.degree(u) < 2) continue;
    m.clear();
    // Scan matching edges by in-vertices hash.
    for (int v : x.edges(u)) {
      if (vis[v]) continue;
      vis[v] = true;
      size_t h = hashValue(es, xt.edges(v));
      if (!m.count(h)) m[h] = make_pair(vector<int>(), v);
      else m[h].first.push_back(v);
    }
    // Save in-identicals.
    for (auto&& [_, p] : m) {
      if (p.first.empty()) continue;
      p.first.insert(p.first.begin(), p.second);
      a.push_back(move(p.first));
    }
  }
  return a;
}
