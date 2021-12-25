#pragma once
#include <map>
#include <vector>
#include <utility>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"

using std::map;
using std::vector;
using std::move;
using std::sort;




template <class G, class J>
auto edgeIdenticalsFromSize(const G& x, const J& ks, int n) {
  using vec = vector<int>;
  map<vec, vec> m; vec es;
  // Find groups of identicals.
  for (int u : ks) {
    write(es, x.edges(u)); sort(es);
    m[es].push_back(u);
  }
  // Copy identicals from given size in sorted order.
  vector2d<int> a;
  for (auto& p : m) {
    auto& is = p.second;
    if (is.size()<n) continue;
    sort(is); a.push_back(move(is));
  }
  return a;
}

template <class G>
auto edgeIdenticalsFromSize(const G& x, int n) {
  return edgeIdenticalsFromSize(x, x.vertices(), n);
}

template <class G, class J>
auto edgeIdenticals(const G& x, const J& ks) {
  return edgeIdenticalsFromSize(x, ks, 2);
}

template <class G>
auto edgeIdenticals(const G& x) {
  return edgeIdenticals(x, x.vertices());
}
