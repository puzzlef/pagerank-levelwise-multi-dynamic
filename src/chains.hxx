#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"

using std::vector;
using std::reverse;




template <class G, class H>
int chainTraverse(vector<int>& a, vector<bool>& vis, const G& x, const H& xt, int v) {
  while (!vis[v] && x.degree(v)==1 && xt.degree(v)==1) {
    vis[v] = true;
    a.push_back(v);
    v = edge(x, v);
  }
  return v;
}


template <class G, class H, class J>
auto chainsFromSize(const G& x, const H& xt, const J& ks, int n) {
  auto vis = createContainer(x, bool());
  vector2d<int> a; vector<int> b;
  for (int v : ks) {
    b.clear();
    // Traverse back.
    int u = chainTraverse(b, vis, xt, x, v);
    if (x.degree(u)==1) b.push_back(u);
    reverse(b.begin(), b.end());
    if (b.size()>0) b.pop_back();
    // Traverse front.
    int w = chainTraverse(b, vis, x, xt, v);
    if (xt.degree(w)==1) b.push_back(w);
    // Record chain.
    if (b.size()<n) continue;
    a.push_back(b);
  }
  return a;
}

template <class G, class H>
auto chainsFromSize(const G& x, const H& xt, int n) {
  return chainsFromSize(x, xt, x.vertices(), n);
}

template <class G, class H, class J>
auto chains(const G& x, const H& xt, const J& ks) {
  return chainsFromSize(x, xt, ks, 2);
}

template <class G, class H>
auto chains(const G& x, const H& xt) {
  return chains(x, xt, x.vertices());
}
