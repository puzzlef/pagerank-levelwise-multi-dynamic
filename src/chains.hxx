#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"

using std::vector;
using std::reverse;




template <class G, class H>
void chainTraverse(vector<int>& a, vector<bool>& vis, const G& x, const H& xt, int v) {
  while (!vis[v] && x.degree(v)==1 && xt.degree(v)==1) {
    vis[v] = true;
    a.push_back(v);
    v = edge(x, v);
  }
}


template <class G, class H>
auto chains(const G& x, const H& xt) {
  vector2d<int> a;
  auto vis = createContainer(x, bool());
  for (int v : x.vertices()) {
    vector<int> b;
    if (vis[v] || xt.degree(v)!=1 || x.degree(v)!=1) continue;
    int u = edge(xt, v), w = edge(x, v);
    if ((vis[u] || xt.degree(u)!=1) && (vis[w] || x.degree(w)!=1)) continue;
    chainTraverse(b, vis, xt, x, u); reverse(b.begin(), b.end());
    chainTraverse(b, vis, x, xt, v);
    a.push_back(b);
  }
  return a;
}
