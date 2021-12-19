#pragma once
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
#include "pagerankOmp.hxx"
#include "pagerankMonolithicOmp.hxx"
#include "pagerankComponentwiseOmp.hxx"

using std::vector;
using std::swap;




// PAGERANK (STATIC / INCREMENTAL)
// -------------------------------

// Find pagerank using a single thread (pull, CSR).
// @param x  original graph
// @param xt transpose graph (with vertex-data=out-degree)
// @param q  initial ranks (optional)
// @param o  options {damping=0.85, tolerance=1e-6, maxIterations=500}
// @returns {ranks, iterations, time}
template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseOmp(const G& x, const H& xt, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *D=nullptr) {
  int  N  = xt.order();  if (N==0) return PagerankResult<T>::initial(xt, q);
  const auto& cs = componentsD(x, xt, D);
  const auto& b  = blockgraphD(x, cs, D);
  const auto& bt = blockgraphTransposeD(b, D);
  auto gs = levelwiseGroupedComponentsFrom(cs, bt);
  auto ns = transformIter(gs, [&](const auto& g) { return g.size(); });
  auto ks = join<int>(gs);
  return pagerankOmp(xt, ks, 0, ns, pagerankComponentwiseOmpLoop<T, decltype(ns)>, q, o);
}
template <class G, class T=float>
PagerankResult<T> pagerankLevelwiseOmp(const G& x, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *D=nullptr) {
  auto xt = transposeWithDegree(x);
  return pagerankLevelwiseOmp(x, xt, q, o, D);
}




// PAGERANK (DYNAMIC)
// ------------------

template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseOmpDynamic(const G& x, const H& xt, const G& y, const H& yt, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *D=nullptr) {
  int  N  = yt.order();                                 if (N==0) return PagerankResult<T>::initial(yt, q);
  const auto& cs = componentsD(y, yt, D);
  const auto& b  = blockgraphD(y, cs, D);
  const auto& bt = blockgraphTransposeD(b, D);
  auto ds = levelwiseComponentsFrom(cs, bt);
  auto gi = levelwiseGroupIndices(bt);
  auto [is, n] = dynamicComponentIndices(x, y, ds, b);  if (n==0) return PagerankResult<T>::initial(yt, q);
  auto fn = [&](const auto& b, int i) { return b.empty() || gi[b.back()]==gi[i]; };
  auto ig = groupIf<int>(sliceIter(is, 0, n), fn);
  auto gs = joinAt2d<int>(ds, ig);
  auto ns = transformIter(gs, [&](const auto& g) { return g.size(); });
  auto ks = join<int>(gs); joinAt(ks, ds, sliceIter(is, n));
  return pagerankOmp(yt, ks, 0, ns, pagerankComponentwiseOmpLoop<T, decltype(ns)>, q, o);
}

template <class G, class T=float>
PagerankResult<T> pagerankLevelwiseOmpDynamic(const G& x, const G& y, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *D=nullptr) {
  auto xt = transposeWithDegree(x);
  auto yt = transposeWithDegree(y);
  return pagerankLevelwiseOmpDynamic(x, xt, y, yt, q, o, D);
}
