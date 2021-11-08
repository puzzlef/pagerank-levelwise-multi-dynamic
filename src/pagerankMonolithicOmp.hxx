#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "transpose.hxx"
#include "dynamic.hxx"
#include "pagerank.hxx"

using std::vector;
using std::swap;




template <class T>
void pagerankFactorOmp(vector<T>& a, const vector<int>& vdata, int i, int n, T p) {
  #pragma omp parallel for num_threads(32) schedule(auto)
  for (int u=i; u<i+n; u++) {
    int d = vdata[u];
    a[u] = d>0? p/d : 0;
  }
}


template <class T>
void pagerankCalculateOmp(vector<T>& a, const vector<T>& c, const vector<int>& vfrom, const vector<int>& efrom, int i, int n, T c0) {
  #pragma omp parallel for num_threads(32) schedule(auto)
  for (int v=i; v<i+n; v++)
    a[v] = c0 + sumAt(c, sliceIter(efrom, vfrom[v], vfrom[v+1]));
}




// PAGERANK-ERROR
// --------------

template <class T>
T pagerankErrorOmp(const vector<T>& x, const vector<T>& y, int i, int N, int EF) {
  switch (EF) {
    case 1:  return l1NormOmp(x, y, i, N);
    case 2:  return l2NormOmp(x, y, i, N);
    default: return liNormOmp(x, y, i, N);
  }
}




// PAGERANK-CORE
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


template <class H, class J, class FL, class T=float>
PagerankResult<T> pagerankMonolithicOmpCore(const H& xt, const J&& ks, int i, int n, FL fl, const vector<T> *q, PagerankOptions<T> o) {
  int  N  = xt.order();
  T    p  = o.damping;
  T    E  = o.tolerance;
  int  L  = o.maxIterations, l = 0;
  int  EF = o.toleranceNorm;
  auto vfrom = sourceOffsets(xt, ks);
  auto efrom = destinationIndices(xt, ks);
  auto vdata = vertexData(xt, ks);
  vector<T> a(N), r(N), c(N), f(N), qc;
  if (q) qc = compressContainer(xt, *q, ks);
  float t = measureDurationMarked([&](auto mark) {
    if (N==0 || n==0) return;  // skip if nothing to do!
    if (q) copyOmp(r, qc);     // copy old ranks (qc), if given
    else fillOmp(r, T(1)/N);
    copyOmp(a, r);
    mark([&] { pagerankFactorOmp(f, vdata, 0, N, p); multiplyOmp(c, a, f, 0, N); });  // calculate factors (f) and contributions (c)
    mark([&] { l = fl(a, r, c, f, vfrom, efrom, i, n, N, p, E, L, EF); });            // calculate ranks of vertices
  }, o.repeat);
  return {decompressContainer(xt, a, ks), l, t};
}




// PAGERANK (STATIC / INCREMENTAL)
// -------------------------------

// Find pagerank accelerated with OpenMP (pull, CSR).
// @param x original graph
// @param q initial ranks (optional)
// @param o options {damping=0.85, tolerance=1e-6, maxIterations=500}
// @returns {ranks, iterations, time}
template <class G, class FL, class T=float>
PagerankResult<T> pagerankMonolithicOmp(const G& x, FL fl, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  int  N  = x.order();
  auto xt = transposeWithDegree(x);
  return pagerankMonolithicOmpCore(xt, xt.vertices(), 0, N, fl, q, o);
}

template <class G, class T=float>
PagerankResult<T> pagerankMonolithicOmp(const G& x, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  return pagerankMonolithicOmp(x, pagerankMonolithicOmpLoop<T>, q, o);
}




// PAGERANK (DYNAMIC)
// ------------------

template <class G, class FD, class FL, class T=float>
PagerankResult<T> pagerankMonolithicOmpDynamic(const G& x, const G& y, FD fd, FL fl, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto [ks, n] = fd(x, y);
  auto yt = transposeWithDegree(y);
  return pagerankMonolithicOmpCore(yt, ks, 0, n, fl, q, o);
}

template <class G, class T=float>
PagerankResult<T> pagerankMonolithicOmpDynamic(const G& x, const G& y, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto xt = transposeWithDegree(x);
  auto yt = transposeWithDegree(y);
  auto [ks, n] = dynamicVertices(x, xt, y, yt);
  return pagerankMonolithicOmpCore(yt, ks, 0, n, pagerankMonolithicOmpLoop<T>, q, o);
}
