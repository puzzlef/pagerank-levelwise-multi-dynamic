#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "transpose.hxx"
#include "dynamic.hxx"
#include "pagerank.hxx"
#include "pagerankOmp.hxx"

using std::vector;
using std::swap;




// PAGERANK-LOOP
// -------------

template <class T>
int pagerankMonolithicOmpLoop(vector<T>& a, vector<T>& r, vector<T>& c, const vector<T>& f, const vector<int>& vfrom, const vector<int>& efrom, int i, int n, int N, T p, T E, int L, int EF) {
  T  c0 = (1-p)/N;
  int l = 0;
  while (l<L) {
    pagerankCalculateOmp(a, c, vfrom, efrom, i, n, c0);  // assume contribtions (c) is precalculated
    T el = pagerankErrorOmp(a, r, i, n, EF); ++l;        // one iteration complete
    if (el<E || l>=L) break;                             // check tolerance, iteration limit
    multiplyOmp(c, a, f, i, n);                          // update partial contributions (c)
    swap(a, r);                                          // final ranks always in (a)
  }
  return l;
}




// PAGERANK (STATIC / INCREMENTAL)
// -------------------------------

// Find pagerank accelerated with OpenMP (pull, CSR).
// @param x  original graph
// @param xt transpose graph (with vertex-data=out-degree)
// @param q  initial ranks (optional)
// @param o  options {damping=0.85, tolerance=1e-6, maxIterations=500}
// @returns {ranks, iterations, time}
template <class G, class H, class T=float>
PagerankResult<T> pagerankMonolithicOmp(const G& x, const H& xt, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *D=nullptr) {
  int  N  = xt.order();  if (N==0) return PagerankResult<T>::initial(xt, q);
  auto ks = pagerankVertices(x, xt, o, D);
  return pagerankOmp(xt, ks, 0, N, pagerankMonolithicOmpLoop<T>, q, o);
}

template <class G, class T=float>
PagerankResult<T> pagerankMonolithicOmp(const G& x, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *D=nullptr) {
  auto xt = transposeWithDegree(x);
  return pagerankMonolithicOmp(x, xt, q, o);
}




// PAGERANK (DYNAMIC)
// ------------------

template <class G, class H, class T=float>
PagerankResult<T> pagerankMonolithicOmpDynamic(const G& x, const H& xt, const G& y, const H& yt, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *D=nullptr) {
  int  N = yt.order();                                         if (N==0) return PagerankResult<T>::initial(yt, q);
  auto [ks, n] = pagerankDynamicVertices(x, xt, y, yt, o, D);  if (n==0) return PagerankResult<T>::initial(yt, q);
  return pagerankOmp(yt, ks, 0, n, pagerankMonolithicOmpLoop<T>, q, o);
}

template <class G, class T=float>
PagerankResult<T> pagerankMonolithicOmpDynamic(const G& x, const G& y, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *D=nullptr) {
  auto xt = transposeWithDegree(x);
  auto yt = transposeWithDegree(y);
  return pagerankMonolithicOmpDynamic(x, xt, y, yt, q, o, D);
}
