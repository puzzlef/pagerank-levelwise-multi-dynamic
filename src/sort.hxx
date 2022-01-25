#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "dfs.hxx"
#include "components.hxx"

using std::vector;
using std::unordered_map;
using std::reverse;
using std::swap;




// TOPOLOGICAL-SORT
// ----------------
// Arrrange vertices in dependency order.
// Top level vertices may not always come first.

template <class G>
auto topologicalSort(const G& x) {
  vector<int> a;
  auto vis = createContainer(x, bool());
  for (int u : x.vertices())
    if (!vis[u]) dfsEndLoop(a, vis, x, u);
  reverse(a.begin(), a.end());
  return a;
}




// LEVELWISE-SORT
// --------------
// Arrange vertices in dependency and level order.
// Top level vertices always come first.

template <class G, class F>
void levelwiseFrontierDo(const G& x, const vector<int>& frnt, vector<int>& ideg, F fn) {
  for (int u : frnt) {
    for (int v : x.edges(u))
      if (--ideg[v]==0) fn(v);
  }
}
template <class G, class H, class F>
void levelwiseFrontiersDo(const G& x, const H& xt, F fn) {
  vector<int> frnt, frnu, a;
  vector<int> ideg(x.span());
  for (int u : x.vertices()) {
    ideg[u] = x.degree(u);
    if (x.degree(u)==0) frnt.push_back(u);
  }
  while (!frnt.empty()) {
    fn(frnt);
    frnu.clear();
    levelwiseFrontierDo(x, frnt, ideg, [&](int v) { frnu.push_back(v); });
    swap(frnu, frnt);
  }
}
template <class G, class H>
inline auto levelwiseSort(const G& x, const H& xt) {
  vector<int> a;
  levelwiseFrontiersDo(x, xt, [&](const int& frnt) { append(a, frnt); });
  return a;
}




// LEVELWISE-GROUPS
// ----------------
// Arrange groups of vertices in dependency and level order.
// Vertices belonging to the same level come in a group.

template <class G, class H>
inline auto levelwiseGroups(const G& x, const H& xt) {
  vector2d<int> a;
  levelwiseFrontiersDo(x, xt, [&](const auto& frnt) { a.push_back(frnt); });
  return a;
}
template <class G, class H>
inline auto levelwiseGroupIndices(const G& x, const H& xt) {
  vector<int> a(x.span()); int i = 0;
  levelwiseFrontiersDo(x, xt, [&](const auto& frnt) { fillAt(a, i++, frnt); });
  return a;
}




// TOPOLOGICAL-COMPONENTS
// ----------------------
// Get components in topological order.

template <class G>
void topologicalComponentsTo(vector2d<int>& cs, const G& b) {
  auto bks = topologicalSort(b);
  reorderDirty(cs, bks);
}

template <class G>
auto topologicalComponentsFrom(const vector2d<int>& cs, const G& b) {
  auto bks = topologicalSort(b);
  return copyAt(cs, bks);
}

template <class G, class H>
auto topologicalComponents(const G& x, const H& xt) {
  auto cs = components(x, xt);
  auto b  = blockgraph(x, cs);
  topologicalComponentsTo(cs, b);
  return cs;
}




// LEVELWISE-COMPONENTS
// --------------------

template <class G, class H>
inline void levelwiseComponentsTo(vector2d<int>& cs, const G& b, const H& bt) {
  auto bks = levelwiseSort(b, bt);
  reorderDirty(cs, bks);
}
template <class G, class H>
inline auto levelwiseComponentsFrom(const vector2d<int>& cs, const G& b, const H& bt) {
  auto bks = levelwiseSort(b, bt);
  return copyAt(cs, bks);
}
template <class G, class H>
inline auto levelwiseComponents(const G& x, const H& xt) {
  auto cs = components(x, xt);
  auto b  = blockgraph(x, cs);
  auto bt = transpose(x);
  levelwiseComponentsTo(cs, b, bt);
  return cs;
}




// LEVELWISE-GROUPED-COMPONENTS
// ----------------------------

template <class G, class H>
auto levelwiseGroupedComponentsFrom(const vector2d<int>& cs, const G& b, const H& bt) {
  vector2d<int> a;
  auto bgs = levelwiseGroups(b, bt);
  for (const auto& g : bgs)
    a.push_back(joinAt(cs, g));
  return a;
}
template <class G, class H>
inline auto levelwiseGroupedComponents(const G& x, const H& xt) {
  auto cs = components(x, xt);
  auto b  = blockgraph(x, cs);
  auto bt = transpose(b);
  return levelwiseGroupedComponentsFrom(cs, b, bt);
}
