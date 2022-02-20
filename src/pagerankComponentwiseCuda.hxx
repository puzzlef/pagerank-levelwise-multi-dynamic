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
#include "sort.hxx"
#include "dynamic.hxx"
#include "pagerank.hxx"
#include "pagerankCuda.hxx"
#include "pagerankMonolithicCuda.hxx"

using std::array;
using std::vector;
using std::swap;




// PAGERANK-LOOP
// -------------

template <class T, class J>
float pagerankComponentwiseCudaLoop(T *e, T *r0, T *eD, T *r0D, T *&aD, T *&rD, T *cD, const T *fD, const int *vfromD, const int *efromD, int i, const J& ws, int N, T p, T E, int L, int EF) {
  float t = 0;
  for (const auto& w : ws) {
    const auto& [nt, nb] = w;
    int n = -nt + nb;  // thread no. is -ve
    if (n<=0) { i += -n; continue; }
    T np = T(n)/N, En = EF<=2? E*n/N : E;
    t += pagerankMonolithicCudaLoop(e, r0, eD, r0D, aD, rD, cD, fD, vfromD, efromD, i, w, N, p, En, L, EF);
    swap(aD, rD);
    i += n;
  }
  swap(aD, rD);
  return t;
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
PagerankResult<T> pagerankComponentwiseCuda(const G& x, const H& xt, const vector<T> *q, const PagerankOptions<T>& o, const PagerankData<G>& D) {
  const auto& cs = D.components;
  const auto& b  = D.blockgraph;
  int  N  = xt.order();  if (N==0) return PagerankResult<T>::initial(xt, q);
  auto ds = topologicalComponentsFrom(cs, b);
  auto gs = joinUntilSize<int>(ds, o.minCompute);
  forEach(gs, [&](auto& g) { pagerankPartition(xt, g); });
  auto ns = pagerankPairWave(xt, gs);
  auto ks = join<int>(gs);
  return pagerankCuda(xt, ks, 0, ns, pagerankComponentwiseCudaLoop<T, decltype(ns)>, q, o);
}
template <class G, class H, class T=float>
PagerankResult<T> pagerankComponentwiseCuda(const G& x, const H& xt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto cs = components(x, xt);
  auto b  = blockgraph(x, cs);
  return pagerankComponentwiseCuda(x, xt, q, o, PagerankData<G>(cs, b));
}
template <class G, class T=float>
PagerankResult<T> pagerankComponentwiseCuda(const G& x, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto xt = transposeWithDegree(x);
  return pagerankComponentwiseCuda(x, xt, q, o);
}




// PAGERANK (DYNAMIC)
// ------------------

template <class G, class H, class T=float>
PagerankResult<T> pagerankComponentwiseCudaDynamic(const G& x, const H& xt, const G& y, const H& yt, const vector<T> *q, const PagerankOptions<T>& o, const PagerankData<G>& D) {
  const auto& cs = D.components;
  const auto& b  = D.blockgraph;
  int  N  = yt.order();                                 if (N==0) return PagerankResult<T>::initial(yt, q);
  auto ds = topologicalComponentsFrom(cs, b);
  auto [is, n] = dynamicComponentIndices(x, y, ds, b);  if (n==0) return PagerankResult<T>::initial(yt, q);
  auto gs = joinAtUntilSize<int>(ds, sliceIter(is, 0, n), o.minCompute);
  forEach(gs, [&](auto& g) { pagerankPartition(yt, g); });
  auto ns = pagerankPairWave(yt, gs);
  auto ks = join<int>(gs); joinAt(gs, ds, sliceIter(is, n));
  return pagerankCuda(yt, ks, 0, ns, pagerankComponentwiseCudaLoop<T, decltype(ns)>, q, o);
}
template <class G, class H, class T=float>
PagerankResult<T> pagerankComponentwiseCudaDynamic(const G& x, const H& xt, const G& y, const H& yt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto cs = components(y, yt);
  auto b  = blockgraph(y, cs);
  return pagerankComponentwiseCudaDynamic(x, xt, y, yt, q, o, PagerankData<G>(cs, b));
}
template <class G, class T=float>
PagerankResult<T> pagerankComponentwiseCudaDynamic(const G& x, const G& y, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto xt = transposeWithDegree(x);
  auto yt = transposeWithDegree(y);
  return pagerankComponentwiseCudaDynamic(x, xt, y, yt, q, o);
}
