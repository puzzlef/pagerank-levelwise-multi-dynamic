#pragma once
#include <vector>
#include "_main.hxx"
#include "vertices.hxx"
#include "dfs.hxx"

using std::vector;



// COMPONENTS
// ----------
// Finds Strongly Connected Components (SCC) using Kosaraju's algorithm.

template <class G, class H>
auto components(const G& x, const H& xt) {
  vector<vector<int>> a;
  // original dfs
  auto vis = createContainer(x, bool());
  vector<int> vs;
  for (int u : x.vertices())
    if (!vis[u]) dfsEndLoop(vs, vis, x, u);
  // transpose dfs
  fill(vis, false);
  while (!vs.empty()) {
    int u = vs.back(); vs.pop_back();
    if (vis[u]) continue;
    a.push_back(vector<int>());
    dfsLoop(a.back(), vis, xt, u);
  }
  return a;
}




// COMPONENTS-IDS
// --------------
// Get component id of each vertex.

template <class G>
auto componentIds(const G& x, const vector<vector<int>>& comps) {
  auto a = createContainer(x, int()); int i = 0;
  for (const auto& comp : comps) {
    for (int u : comp)
      a[u] = i;
    i++;
  }
  return a;
}




// COMPONENTS-EQUAL
// ----------------

template <class G>
bool componentsEqual(const G& x, const vector<int>& xc, const G& y, const vector<int>& yc) {
  if (xc != yc) return false;
  for (int i=0, I=xc.size(); i<I; i++)
    if (!verticesEqual(x, xc[i], y, yc[i])) return false;
  return true;
}

template <class G, class H>
bool componentsEqual(const G& x, const H& xt, const vector<int>& xc, const G& y, const H& yt, const vector<int>& yc) {
  return componentsEqual(x, xc, y, yc) && componentsEqual(xt, xc, yt, yc);
}
