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
#include "pagerankComponentwiseCuda.hxx"

using std::array;
using std::vector;
using std::swap;




// PAGERANK (STATIC / INCREMENTAL)
// -------------------------------

// Find pagerank accelerated with CUDA (pull, CSR).
// @param x  original graph
// @param xt transpose graph (with vertex-data=out-degree)
// @param q  initial ranks (optional)
// @param o  options {damping=0.85, tolerance=1e-6, maxIterations=500}
// @returns {ranks, iterations, time}
template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseCuda(const G& x, const H& xt, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *D=nullptr) {
  int  N  = xt.order();  if (N==0) return PagerankResult<T>::initial(xt, q);
  const auto& cs = componentsD(x, xt, D);
  const auto& b  = blockgraphD(x, cs, D);
  const auto& bt = blockgraphTransposeD(b, D);
  auto gs = levelwiseGroupedComponentsFrom(cs, bt);
  forEach(gs, [&](auto& g) { pagerankPartition(xt, g); });
  auto ns = pagerankPairWave(xt, gs);
  auto ks = join<int>(gs);
  return pagerankCuda(xt, ks, 0, ns, pagerankComponentwiseCudaLoop<T, decltype(ns)>, q, o);
}

template <class G, class T=float>
PagerankResult<T> pagerankLevelwiseCuda(const G& x, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *D=nullptr) {
  auto xt = transposeWithDegree(x);
  return pagerankLevelwiseCuda(x, xt, q, o, D);
}




// PAGERANK (DYNAMIC)
// ------------------

template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseCudaDynamic(const G& x, const H& xt, const G& y, const H& yt, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *D=nullptr) {
  int  N  = yt.order();                                 if (N==0) return PagerankResult<T>::initial(yt, q);
  const auto& cs = componentsD(y, yt, D);
  const auto& b  = blockgraphD(y, cs, D);
  const auto& bt = blockgraphTransposeD(b, D);
  auto gi = levelwiseGroupIndices(bt);
  auto [is, n] = dynamicComponentIndices(x, y, cs, b);  if (n==0) return PagerankResult<T>::initial(yt, q);
  auto ig = groupBy<int>(sliceIter(is, 0, n), [&](int i) { return gi[i]; });
  auto gs = joinAt2d(cs, ig);
  forEach(gs, [&](auto& g) { pagerankPartition(yt, g); });
  auto ns = pagerankPairWave(yt, gs);
  auto ks = join<int>(gs); joinAt(ks, cs, sliceIter(is, n));
  return pagerankCuda(yt, ks, 0, ns, pagerankComponentwiseCudaLoop<T, decltype(ns)>, q, o);
}

template <class G, class T=float>
PagerankResult<T> pagerankLevelwiseCudaDynamic(const G& x, const G& y, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *D=nullptr) {
  auto xt = transposeWithDegree(x);
  auto yt = transposeWithDegree(y);
  return pagerankLevelwiseCudaDynamic(x, xt, y, yt, q, o, D);
}
