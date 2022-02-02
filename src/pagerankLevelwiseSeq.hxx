#pragma once
#include <utility>
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
#include "pagerankSeq.hxx"
#include "pagerankMonolithicSeq.hxx"
#include "pagerankComponentwiseSeq.hxx"

using std::vector;
using std::swap;
using std::move;




// PAGERANK (STATIC / INCREMENTAL)
// -------------------------------

// Find pagerank using a single thread (pull, CSR).
// @param x  original graph
// @param xt transpose graph (with vertex-data=out-degree)
// @param q  initial ranks (optional)
// @param o  options {damping=0.85, tolerance=1e-6, maxIterations=500}
// @returns {ranks, iterations, time}
template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseSeq(const G& x, const H& xt, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *D=nullptr) {
  int  N  = xt.order();  if (N==0) return PagerankResult<T>::initial(xt, q);
  const auto& cs = componentsD(x, xt, D);
  const auto& b  = blockgraphD(x, cs, D);
  const auto& bt = blockgraphTransposeD(b, D);
  auto fs = levelwiseGroupedComponentsFrom(cs, bt);
  auto gs = joinUntilSize(fs, o.minCompute);
  auto ns = transformIter(gs, [&](const auto& g) { return g.size(); });
  auto ks = join<int>(gs);
  return pagerankSeq(xt, ks, 0, ns, pagerankComponentwiseSeqLoop<T, decltype(ns)>, q, o);
}
template <class G, class T=float>
PagerankResult<T> pagerankLevelwiseSeq(const G& x, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *D=nullptr) {
  auto xt = transposeWithDegree(x);
  return pagerankLevelwiseSeq(x, xt, q, o, D);
}




// PAGERANK (DYNAMIC)
// ------------------

template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseSeqDynamic(const G& x, const H& xt, const G& y, const H& yt, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *D=nullptr) {
  int  N  = yt.order();  if (N==0) return PagerankResult<T>::initial(yt, q);
  const auto& cs = componentsD(x, xt, D);
  const auto& b  = blockgraphD(x, cs, D);
  const auto& bt = blockgraphTransposeD(b, D);
  auto gi = levelwiseGroupIndices(bt);
  auto [is, n] = dynamicComponentIndices(x, y, cs, b);  if (n==0) return PagerankResult<T>::initial(yt, q);
  auto ig = groupBy<int>(sliceIter(is, 0, n), [&](int i) { return gi[i]; });
  auto fs = joinAt2d<int>(cs, ig);
  auto gs = joinUntilSize(fs, o.minCompute);
  auto ns = transformIter(gs, [&](const auto& g) { return g.size(); });
  auto ks = join<int>(gs); joinAt(ks, cs, sliceIter(is, n));
  return pagerankSeq(yt, ks, 0, ns, pagerankComponentwiseSeqLoop<T, decltype(ns)>, q, o);
}
template <class G, class T=float>
PagerankResult<T> pagerankLevelwiseSeqDynamic(const G& x, const G& y, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}, const PagerankData<G> *D=nullptr) {
  auto xt = transposeWithDegree(x);
  auto yt = transposeWithDegree(y);
  return pagerankLevelwiseSeqDynamic(x, xt, y, yt, q, o, D);
}
