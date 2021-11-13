#pragma once
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
#include "pagerankSeq.hxx"
#include "pagerankMonolithicSeq.hxx"

using std::vector;
using std::swap;




// PAGERANK-LOOP
// -------------

template <class T, class J>
int pagerankLevelwiseSeqLoop(vector<T>& a, vector<T>& r, vector<T>& c, const vector<T>& f, const vector<int>& vfrom, const vector<int>& efrom, int i, const J& ns, int N, T p, T E, int L, int EF) {
  float l = 0;
  for (int n : ns) {
    if (n<=0) { i += -n; continue; }
    T np = T(n)/N, En = EF<=2? E*n/N : E;
    l += pagerankMonolithicSeqLoop(a, r, c, f, vfrom, efrom, i, n, N, p, En, L, EF)*np;
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
PagerankResult<T> pagerankLevelwiseSeq(const G& x, const H& xt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  int  N  = xt.order();  if (N==0) return PagerankResult<T>::initial(xt, q);
  auto cs = joinUntilSize(sortedComponents(x, xt), MIN_COMPUTE_PR());
  auto ns = transformIter(cs, [&](const auto& c) { return c.size(); });
  auto ks = join(cs);
  return pagerankSeq(xt, ks, 0, ns, pagerankLevelwiseSeqLoop<T, decltype(ns)>, q, o);
}
template <class G, class T=float>
PagerankResult<T> pagerankLevelwiseSeq(const G& x, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto xt = transposeWithDegree(x);
  return pagerankLevelwiseSeq(x, xt, q, o);
}




// PAGERANK (DYNAMIC)
// ------------------

template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseSeqDynamic(const G& x, const H& xt, const G& y, const H& yt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  int  N  = yt.order();                                 if (N==0) return PagerankResult<T>::initial(yt, q);
  auto cs = sortedComponents(y, yt);
  auto b  = blockgraph(y, cs);
  auto [is, n] = dynamicComponentIndices(x, y, cs, b);  if (n==0) return PagerankResult<T>::initial(yt, q);
  auto ds = joinAtUntilSize(cs, sliceIter(is, 0, n), MIN_COMPUTE_PR());
  auto ns = transformIter(ds, [&](const auto& d) { return d.size(); });
  auto ks = join(ds); joinAt(ks, cs, sliceIter(is, n));
  return pagerankSeq(yt, ks, 0, ns, pagerankLevelwiseSeqLoop<T, decltype(ns)>, q, o);
}
template <class G, class T=float>
PagerankResult<T> pagerankLevelwiseSeqDynamic(const G& x, const G& y, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto xt = transposeWithDegree(x);
  auto yt = transposeWithDegree(y);
  return pagerankLevelwiseSeqDynamic(x, xt, y, yt, q, o);
}
