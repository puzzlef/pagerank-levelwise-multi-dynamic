#pragma once
#include <array>
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "transpose.hxx"
#include "components.hxx"
#include "dynamic.hxx"
#include "pagerank.hxx"
#include "pagerankCuda.hxx"
#include "pagerankLevelwiseSeq.hxx"

using std::array;
using std::vector;
using std::swap;




// PAGERANK-LOOP
// -------------

template <class T, class J>
int pagerankLevelwiseCudaLoop(T *e, T *r0, T *eD, T *r0D, T *&aD, T *&rD, T *cD, const T *fD, const int *vfromD, const int *efromD, int i, const J& ws, int N, T p, T E, int L, int EF) {
  float l = 0;
  for (const auto& w : ws) {
    const auto& [nt, nb] = w; int n = nt+nb;
    if (n<=0) { i += -n; continue; }
    T np = T(n)/N, En = EF<=2? E*n/N : E;
    l += pagerankMonolithicCudaLoop(e, r0, eD, r0D, aD, rD, cD, fD, vfromD, efromD, i, w, N, p, En, L, EF)*np;
    swap(aD, rD);
    i += n;
  }
  swap(aD, rD);
  return int(l);
}




// PAGERANK (STATIC / INCREMENTAL)
// -------------------------------

// Find pagerank accelerated with CUDA (pull, CSR).
// @param x  original graph
// @param xt transpose graph (with vertex-data=out-degree)
// @param q  initial ranks (optional)
// @param o  options {damping=0.85, tolerance=1e-6, maxIterations=500}
// @returns {ranks, iterations, time}
template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseCuda(const G& x, const H& xt, const vector<T> *q, const PagerankOptions<T>& o, const PagerankData<G>& D) {
  int  N  = xt.order();  if (N==0) return PagerankResult<T>::initial(xt, q);
  auto cs = joinUntilSize(D.sortedComponents, MIN_COMPUTE_PRC());
  forEach(cs, [&](auto& ks) { pagerankPartition(xt, ks); });
  auto ns = pagerankPairWave(xt, cs);
  auto ks = join(cs);
  return pagerankCuda(xt, ks, 0, ns, pagerankLevelwiseCudaLoop<T, decltype(ns)>, q, o);
}
template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseCuda(const G& x, const H& xt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto cs = sortedComponents(x, xt);
  return pagerankLevelwiseCuda(x, xt, q, o, PagerankData<G>(move(cs)));
}
template <class G, class T=float>
PagerankResult<T> pagerankLevelwiseCuda(const G& x, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto xt = transposeWithDegree(x);
  return pagerankLevelwiseCuda(x, xt, q, o);
}




// PAGERANK (DYNAMIC)
// ------------------

template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseCudaDynamic(const G& x, const H& xt, const G& y, const H& yt, const vector<T> *q, const PagerankOptions<T>& o, const PagerankData<G>& D) {
  int  N  = yt.order();                                 if (N==0) return PagerankResult<T>::initial(yt, q);
  const auto& cs = D.sortedComponents;
  const auto& b  = D.blockgraph;
  auto [is, n] = dynamicComponentIndices(x, y, cs, b);  if (n==0) return PagerankResult<T>::initial(yt, q);
  auto ds = joinAtUntilSize(cs, sliceIter(is, 0, n), MIN_COMPUTE_PRC());
  forEach(ds, [&](auto& ks) { pagerankPartition(yt, ks); });
  auto ns = pagerankPairWave(yt, ds);
  auto ks = join(ds); joinAt(ks, cs, sliceIter(is, n));
  return pagerankCuda(yt, ks, 0, ns, pagerankLevelwiseCudaLoop<T, decltype(ns)>, q, o);
}
template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseCudaDynamic(const G& x, const H& xt, const G& y, const H& yt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto cs = components(y, yt);
  auto b  = blockgraph(y, cs);
  sortComponents(cs, b);
  return pagerankLevelwiseCudaDynamic(x, xt, y, yt, q, o, PagerankData<G>(move(cs), move(b)));
}
template <class G, class T=float>
PagerankResult<T> pagerankLevelwiseCudaDynamic(const G& x, const G& y, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto xt = transposeWithDegree(x);
  auto yt = transposeWithDegree(y);
  return pagerankLevelwiseCudaDynamic(x, xt, y, yt, q, o);
}
