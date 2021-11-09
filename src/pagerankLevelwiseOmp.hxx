#pragma once
#include <vector>
#include <algorithm>
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "transpose.hxx"
#include "components.hxx"
#include "dynamic.hxx"
#include "pagerank.hxx"
#include "pagerankOmp.hxx"
#include "pagerankMonolithicOmp.hxx"

using std::vector;
using std::swap;





template <class T>
inline T pagerankLevelwiseError(T E, int n, int N, int EF) {
  return EF<=2? E*n/N : E;
}


template <class G, class H, class C>
auto pagerankWaves(const G& w, const H& wt, const C& wcs, const G& x, const H& xt, const C& xcs) {
  int W = wcs.size();
  int X = xcs.size();
  auto b = blockgraph(x, xcs);
  vector<bool> dirty(X);
  for (int u : b.vertices()) {
    if (dirty[u]) continue;
    if (findIndex(wcs, xcs[u])>=0 && componentsEqual(w, wt, xcs[u], x, xt, xcs[u])) continue;
    dfsDo(b, u, [&](int v) { dirty[v] = true; });
  }
  vector<int> a(X);
  for (int i=0; i<X; i++) {
    int n = xcs[i].size();
    a[i] = dirty[i]? n:-n;
  }
  return a;
}


template <class C>
auto pagerankGroupComponents(const C& cs, const vector<int>& ws) {
  vector<int> is, js;
  for (int i=0; i<cs.size(); i++) {
    if (ws[i]>=0) is.push_back(i);
    else js.push_back(i);
  }
  auto a = joinAtUntilSize(cs, is, MIN_COMPUTE_SIZE_PR);
  a.push_back(joinAt(cs, js));
  return a;
}


template <class C>
auto pagerankGroupWaves(const C& cs) {
  vector<int> a;
  for (int i=0; i<cs.size()-1; i++)
    a.push_back(cs[i].size());
  a.push_back(-cs.back().size());
  return a;
}


template <class T, class J>
int pagerankLevelwiseOmpLoop(vector<T>& a, vector<T>& r, vector<T>& c, const vector<T>& f, const vector<int>& vfrom, const vector<int>& efrom, int i, J&& ns, int N, T p, T E, int L, int EF) {
  float l = 0;
  for (int n : ns) {
    if (n<=0) { i += -n; continue; }
    T np = T(n)/N, En = pagerankLevelwiseError(E, n, N, EF);
    l += pagerankMonolithicOmpLoop(a, r, c, f, vfrom, efrom, i, n, N, p, En, L, EF)*np;
    swap(a, r);
    i += n;
  }
  swap(a, r);
  return int(l);
}




// PAGERANK (STATIC / INCREMENTAL)
// -------------------------------

// Find pagerank using a single thread (pull, CSR).
// @param x  original graph
// @param xt transpose graph (with vertex-data=out-degree)
// @param q  initial ranks (optional)
// @param o  options {damping=0.85, tolerance=1e-6, maxIterations=500}
// @returns {ranks, iterations, time}
template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseOmp(const G& x, const H& xt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  int  N  = xt.order();
  auto cs = joinUntilSize(sortedComponents(x, xt), MIN_COMPUTE_PR());
  auto ns = pagerankLevelwiseWaves(cs);
  auto ks = join(cs);
  return pagerankOmp(xt, ks, 0, ns, pagerankLevelwiseOmpLoop<T>, q, o);
}
template <class G, class T=float>
PagerankResult<T> pagerankLevelwiseOmp(const G& x, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto xt = transposeWithDegree(x);
  return pagerankLevelwiseOmp(x, xt, q, o);
}




// PAGERANK (DYNAMIC)
// ------------------

template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseOmpDynamic(const G& x, const H& xt, const G& y, const H& yt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto cs = sortedComponents(y, yt);
  auto b  = blockgraph(x, cs);
  auto [ics, nc] = dynamicComponentIndices(x, y, cs, b);
  if (nc==0) return PagerankResult<T>::initial(xt, q);
  auto cs = pagerankLevelwiseGroupComponents();  // PROGRAM DEBT!
  auto ns = pagerankLevelwiseGroupWaves();       // PROGRAM DEBT!
  auto ks = join(cs);
  return pagerankOmp(xt, ks, 0, ns, pagerankLevelwiseOmpLoop<T>, q, o);
}
template <class G, class T=float>
PagerankResult<T> pagerankLevelwiseOmpDynamic(const G& x, const G& y, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto xt = transposeWithDegree(x);
  auto yt = transposeWithDegree(y);
  return pagerankLevelwiseOmpDynamic(x, xt, y, yt, q, o);
}
