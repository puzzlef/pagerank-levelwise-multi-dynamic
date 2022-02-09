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

template <class G>
inline bool addEdge(G& a, const pair<int, int>& uv) {
  auto [u, v] = uv;
  if (u < 0 || v < 0) return false;
  a.addEdge(u, v);
  return true;
}


template <class G, class R>
auto suggestAddRandomEdge(const G& x, R& rnd, int span) {
  uniform_real_distribution<> dis(0.0, 1.0);
  int u = int(dis(rnd) * span);
  int v = int(dis(rnd) * span);
  return make_pair(u, v);
}
template <class G, class R>
inline bool addRandomEdge(G& a, R& rnd, int span) {
  return addEdge(a, suggestAddRandomEdge(a, rnd, span));
}


template <class G, class R>
auto suggestAddRandomEdgeByDegree(const G& x, R& rnd, int span) {
  uniform_real_distribution<> dis(0.0, 1.0);
  double deg = x.size() / x.span();
  int un = int(dis(rnd) * deg * span);
  int vn = int(dis(rnd) * deg * span);
  int u = -1, v = -1, n = 0;
  for (int w : x.vertices()) {
    if (un<0 && un > n+x.degree(w)) u = w;
    if (vn<0 && vn > n+x.degree(w)) v = w;
    if (un>0 && vn>=0) break;
    n += x.degree(w);
  }
  if (u<0) u = int(un/deg);
  if (v<0) v = int(vn/deg);
  return make_pair(u, v);
}
template <class G, class R>
inline bool addRandomEdgeByDegree(G& a, R& rnd, int span) {
  return addEdge(a, suggestAddRandomEdgeByDegree(a, rnd, span));
}




// REMOVE-RANDOM-EDGE
// ------------------

template <class G>
inline bool removeEdge(G& a, const pair<int, int>& uv) {
  auto [u, v] = uv;
  if (u < 0 || v < 0) return false;
  a.removeEdge(u, v);
  return true;
}


template <class G, class R>
auto suggestRemoveRandomEdge(const G& x, R& rnd, int u) {
  uniform_real_distribution<> dis(0.0, 1.0);
  if (a.degree(u) == 0) return false;
  int vi = int(dis(rnd) * a.degree(u)), i = 0;
  for (int v : a.edges(u))
    if (i++ == vi) return make_pair(u, v);
  return make_pair(-1, -1);
}
template <class G, class R>
inline bool removeRandomEdge(G& a, R& rnd, int u) {
  return removeEdge(a, suggestRemoveRandomEdge(a, rnd, u));
}


template <class G, class R>
auto suggestRemoveRandomEdge(const G& x, R& rnd) {
  uniform_real_distribution<> dis(0.0, 1.0);
  int u = int(dis(rnd) * x.span());
  return suggestRemoveRandomEdge(x, rnd, u);
}
template <class G, class R>
inline bool removeRandomEdge(G& a, R& rnd) {
  return removeEdge(a, suggestRemoveRandomEdge(a, rnd));
}


template <class G, class R>
auto suggestRemoveRandomEdgeByDegree(const G& x, R& rnd) {
  uniform_real_distribution<> dis(0.0, 1.0);
  int v = int(dis(rnd) * x.size()), n = 0;
  for (int u : x.vertices()) {
    if (v > n+x.degree(u)) n += x.degree(u);
    else return suggestRemoveRandomEdge(x, rnd, u);
  }
  return make_pair(-1, -1);
}
template <class G, class R>
inline bool removeRandomEdgeByDegree(G& a, R& rnd) {
  return removeEdge(a, suggestRemoveRandomEdgeByDegree(a, rnd));
}
