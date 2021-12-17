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
PagerankResult<T> pagerankLevelwiseCuda(const G& x, const H& xt, const vector<T> *q, const PagerankOptions<T>& o, const PagerankData<G>& D) {
  const auto& cs = D.components;
  const auto& b  = D.blockgraph;
  int  N  = xt.order();  if (N==0) return PagerankResult<T>::initial(xt, q);
  auto bt = transpose(b);
  auto gs = levelwiseGroupedComponentsFrom(cs, bt);
  forEach(gs, [&](auto& g) { pagerankPartition(xt, g); });
  auto ns = pagerankPairWave(xt, gs);
  auto ks = join(gs);
  return pagerankCuda(xt, ks, 0, ns, pagerankComponentwiseCudaLoop<T, decltype(ns)>, q, o);
}
template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseCuda(const G& x, const H& xt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto cs = components(x, xt);
  auto b  = blockgraph(x, cs);
  return pagerankLevelwiseCuda(x, xt, q, o, PagerankData<G>(cs, b));
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
  const auto& cs = D.components;
  const auto& b  = D.blockgraph;
  int  N  = yt.order();                                 if (N==0) return PagerankResult<T>::initial(yt, q);
  auto bt = transpose(b);
  auto ds = levelwiseComponentsFrom(cs, bt):
  auto gi = levelwiseGroupIndices(bt);
  auto [is, n] = dynamicComponentIndices(x, y, ds, b);  if (n==0) return PagerankResult<T>::initial(yt, q);
  auto fn = [&](const auto& ig, int i) { return ig.empty() || ig.back()==i; };
  auto ig = joinIf(sliceIter(is, 0, n), fn);
  auto gs = joinAt2d(ds, ig);
  forEach(gs, [&](auto& g) { pagerankPartition(yt, g); });
  auto ns = pagerankPairWave(yt, gs);
  auto ks = join(gs); joinAt(ks, ds, sliceIter(is, n));
  return pagerankCuda(yt, ks, 0, ns, pagerankComponentwiseCudaLoop<T, decltype(ns)>, q, o);
}
template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseCudaDynamic(const G& x, const H& xt, const G& y, const H& yt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto cs = components(y, yt);
  auto b  = blockgraph(y, cs);
  return pagerankLevelwiseCudaDynamic(x, xt, y, yt, q, o, PagerankData<G>(cs, b));
}
template <class G, class T=float>
PagerankResult<T> pagerankLevelwiseCudaDynamic(const G& x, const G& y, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto xt = transposeWithDegree(x);
  auto yt = transposeWithDegree(y);
  return pagerankLevelwiseCudaDynamic(x, xt, y, yt, q, o);
}
