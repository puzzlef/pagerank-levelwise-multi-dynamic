#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "dfs.hxx"
#include "components.hxx"

using std::vector;
using std::reverse;




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

template <class H>
void levelwiseSortAdd(vector<int>& a, vector<bool>& vis, const H& xt) {
  auto fn = [&](int u) { return vis[u]; };
  for (int u : xt.vertices()) {
    if (vis[u] || !allOf(xt.edges(u), fn)) continue;
    a.push_back(u); vis[u] = true;
  }
}


template <class H>
auto levelwiseSort(const H& xt) {
  vector<int> a;
  auto vis = createContainer(xt, bool());
  while (a.size() < xt.order())
    levelwiseSortAdd(a, vis, xt);
  return a;
}




// LEVELWISE-GROUPS
// ----------------
// Arrange groups of vertices in dependency and level order.
// Vertices belonging to the same level come in a group.

template <class H>
auto levelwiseGroups(const H& xt) {
  vector2d<int> a;
  auto vis = createContainer(xt, bool());
  while (a.size() < xt.order()) {
    a.push_back({});
    levelwiseSortAdd(a.back(), vis, xt);
  }
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

template <class H>
void levelwiseComponentsTo(vector2d<int>& cs, const H& bt) {
  auto bks = levelwiseSort(bt);
  reorderDirty(cs, bks);
}

template <class H>
auto levelwiseComponentsFrom(const vector2d<int>& cs, const H& bt) {
  auto bks = levelwiseSort(bt);
  return copyAt(cs, bks);
}

template <class G, class H>
auto levelwiseComponents(const G& x, const H& xt) {
  auto cs = components(x, xt);
  auto b  = blockgraph(x, cs);
  auto bt = transpose(x);
  levelwiseComponentsTo(cs, bt);
  return cs;
}




// LEVELWISE-GROUPS
// ----------------

template <class H>
auto levelwiseGroupsFrom(const vector2d<int>& cs, const H& bt) {
  vector2d<int> a;
  auto bgs = levelwiseGroups(bt);
  for (const auto& g : bgs)
    a.push_back(joinAt(cs, g));
  return a;
}

template <class G, class H>
auto levelwiseGroups(const G& x, const H& xt) {
  auto cs = components(x, xt);
  auto b  = blockgraph(x, cs);
  auto bt = transpose(b);
  return levelwiseGroupsFrom(cs, bt);
}
