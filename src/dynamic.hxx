#pragma once
#include <utility>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"

using std::vector;
using std::unordered_set;
using std::max;
using std::make_pair;




// ADJUST-RANKS
// ------------

template <class T, class J>
void adjustRanks(vector<T>& a, const vector<T>& r, J&& ks0, J&& ks1, T radd, T rmul, T rset) {
  auto ksNew = setDifference(ks1, ks0);
  for (int k : ks0)
    a[k] = (r[k]+radd)*rmul;
  for (int k : ksNew)
    a[k] = rset;
}

template <class T, class J>
auto adjustRanks(int N, const vector<T>& r, J&& ks0, J&& ks1, T radd, T rmul, T rset) {
  vector<T> a(N); adjustRanks(a, r, ks0, ks1, radd, rmul, rset);
  return a;
}




// CHANGED-VERTICES
// ----------------

template <class G, class F>
void changedVerticesForEach(const G& x, const G& y, F fn) {
  for (int u : y.vertices())
    if (!x.hasVertex(u) || !verticesEqual(x, u, y, u)) fn(u);
}

template <class G, class H, class F>
void changedVerticesForEach(const G& x, const H& xt, const G& y, const H& yt, F fn) {
  for (int u : y.vertices())
    if (!x.hasVertex(u) || !verticesEqual(x, xt, u, y, yt, u)) fn(u);
}

template <class G>
auto changedVertices(const G& x, const G& y) {
  vector<int> a; changedVerticesForEach(x, y, [&](int u) { a.push_back(u); });
  return a;
}

template <class G, class H>
auto changedVertices(const G& x, const H& xt, const G& y, const H& yt) {
  vector<int> a; changedVerticesForEach(x, xt, y, yt, [&](int u) { a.push_back(u); });
  return a;
}




// AFFECTED-VERTICES
// -----------------

template <class G, class F>
void affectedVerticesForEach(const G& x, const G& y, F fn) {
  auto visx = createContainer(x, bool());
  auto visy = createContainer(y, bool());
  auto fny  = [&](int u) { if (!visx[u]) fn(u); };
  changedVerticesForEach(x, y, [&](int u) { dfsDoLoop(visx, x, u, fn); });
  changedVerticesForEach(x, y, [&](int u) { dfsDoLoop(visy, y, u, fny); });
}

template <class G, class H, class F>
void affectedVerticesForEach(const G& x, const H& xt, const G& y, const H& yt, F fn) {
  auto vis = createContainer(y, bool());
  changedVerticesForEach(x, xt, y, yt, [&](int u) { dfsDoLoop(vis, y, u, fn); });
}

template <class G>
auto affectedVertices(const G& x, const G& y) {
  vector<int> a; affectedVerticesForEach(x, y, [&](int u) { a.push_back(u); });
  return a;
}

template <class G, class H>
auto affectedVertices(const G& x, const H& xt, const G& y, const H& yt) {
  vector<int> a; affectedVerticesForEach(x, xt, y, yt, [&](int u) { a.push_back(u); });
  return a;
}




// DYNAMIC-VERTICES
// ----------------

template <class G, class FA>
auto dynamicVerticesBy(const G& y, FA fa) {
  vector<int> a; unordered_set<int> aff;
  fa([&](int u) { a.push_back(u); aff.insert(u); });
  for (int u : y.vertices())
    if (aff.count(u)==0) a.push_back(u);
  // printf("dynamicVerticesBy: aff: %d org: %d\n", aff.size(), a.size()-aff.size());
  // printf("dynamicVerticesBy: "); print(a); printf("\n");
  return make_pair(a, aff.size());
}

template <class G>
auto dynamicVertices(const G& x, const G& y) {
  return dynamicVerticesBy(y, [&](auto fn) {
    affectedVerticesForEach(x, y, fn);
  });
}

template <class G, class H>
auto dynamicVertices(const G& x, const H& xt, const G& y, const H& yt) {
  return dynamicVerticesBy(y, [&](auto fn) {
    affectedVerticesForEach(x, xt, y, yt, fn);
  });
}
