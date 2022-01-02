#pragma once
#include <vector>
#include <iterator>
#include <algorithm>
#include <random>
#include "_main.hxx"

using std::vector;
using std::uniform_real_distribution;
using std::transform;
using std::back_inserter;




// EDGES
// -----

template <class G, class F, class D>
auto edges(const G& x, int u, F fm, D fp) {
  vector<int> a;
  append(a, x.edges(u));
  auto ie = a.end(), ib = a.begin();
  fp(ib, ie); transform(ib, ie, ib, fm);
  return a;
}

template <class G, class F>
auto edges(const G& x, int u, F fm) {
  return edges(x, u, fm, [](auto ib, auto ie) {});
}

template <class G>
auto edges(const G& x, int u) {
  return edges(x, u, [](int v) { return v; });
}




// EDGE
// ----

template <class G, class F>
auto edge(const G& x, int u, F fm) {
  for (int v : x.edges(u))
    return fm(v);
  return -1;
}

template <class G>
auto edge(const G& x, int u) {
  return edge(x, u, [](int v) { return v; });
}




// EDGE-DATA
// ---------

template <class G, class J, class F, class D>
auto edgeData(const G& x, const J& ks, F fm, D fp) {
  using E = decltype(fm(0, 0));
  vector<E> a;
  vector<int> b;
  for (int u : ks) {
    b.clear(); append(b, x.edges(u));
    auto ie = b.end(), ib = b.begin();
    fp(ib, ie); transform(ib, ie, back_inserter(a), [&](int v) { return fm(u, v); });
  }
  return a;
}

template <class G, class J, class F>
auto edgeData(const G& x, const J& ks, F fm) {
  return edgeData(x, ks, fm, [](auto ib, auto ie) {});
}

template <class G, class J>
auto edgeData(const G& x, const J& ks) {
  return edgeData(x, ks, [&](int u, int v) { return x.edgeData(u, v); });
}

template <class G>
auto edgeData(const G& x) {
  return edgeData(x, x.vertices());
}




// EDGES-VISITED
// -------------

template <class G>
bool allEdgesVisited(const G& x, int u, const vector<bool>& vis) {
  for (int v : x.edges(u))
    if (!vis[v]) return false;
  return true;
}

template <class G>
bool someEdgesVisited(const G& x, int u, const vector<bool>& vis) {
  for (int v : x.edges(u))
    if (vis[v]) return true;
  return false;
}




// ADD-RANDOM-EDGE
// ---------------

template <class G, class R>
void addRandomEdge(G& a, R& rnd, int span) {
  uniform_real_distribution<> dis(0.0, 1.0);
  int u = int(dis(rnd) * span);
  int v = int(dis(rnd) * span);
  a.addEdge(u, v);
}


template <class G, class R>
void addRandomEdgeByDegree(G& a, R& rnd, int span) {
  uniform_real_distribution<> dis(0.0, 1.0);
  double deg = a.size() / a.span();
  int un = int(dis(rnd) * deg * span);
  int vn = int(dis(rnd) * deg * span);
  int u = -1, v = -1, n = 0;
  for (int w : a.vertices()) {
    if (un<0 && un > n+a.degree(w)) u = w;
    if (vn<0 && vn > n+a.degree(w)) v = w;
    if (un>0 && vn>=0) break;
    n += a.degree(w);
  }
  if (u<0) u = int(un/deg);
  if (v<0) v = int(vn/deg);
  a.addEdge(u, v);
}




// REMOVE-RANDOM-EDGE
// ------------------

template <class G, class R>
bool removeRandomEdge(G& a, R& rnd, int u) {
  uniform_real_distribution<> dis(0.0, 1.0);
  if (a.degree(u) == 0) return false;
  int vi = int(dis(rnd) * a.degree(u)), i = 0;
  for (int v : a.edges(u))
    if (i++ == vi) { a.removeEdge(u, v); return true; }
  return false;
}


template <class G, class R>
bool removeRandomEdge(G& a, R& rnd) {
  uniform_real_distribution<> dis(0.0, 1.0);
  int u = int(dis(rnd) * a.span());
  return removeRandomEdge(a, rnd, u);
}


template <class G, class R>
bool removeRandomEdgeByDegree(G& a, R& rnd) {
  uniform_real_distribution<> dis(0.0, 1.0);
  int v = int(dis(rnd) * a.size()), n = 0;
  for (int u : a.vertices()) {
    if (v > n+a.degree(u)) n += a.degree(u);
    else return removeRandomEdge(a, rnd, u);
  }
  return false;
}
