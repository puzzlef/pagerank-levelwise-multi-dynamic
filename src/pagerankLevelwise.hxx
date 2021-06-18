#pragma once
#include <vector>
#include <algorithm>
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "components.hxx"
#include "blockgraph.hxx"
#include "topologicalSort.hxx"
#include "pagerank.hxx"
#include "pagerankMonolithic.hxx"

using std::vector;
using std::swap;




template <class G, class H>
auto pagerankComponents(const G& x, const H& xt) {
  auto a = components(x, xt);
  auto b = blockgraph(x, a);
  auto bks = topologicalSort(b);
  reorder(a, bks);
  return a;
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
int pagerankLevelwiseLoop(vector<T>& a, vector<T>& r, vector<T>& c, const vector<T>& f, const vector<int>& vfrom, const vector<int>& efrom, int i, J&& ns, int N, T p, T E, int L) {
  float l = 0;
  for (int n : ns) {
    float nN = float(n)/N;
    if (n<=0) { i += -n; continue; }
    l += pagerankMonolithicLoop(a, r, c, f, vfrom, efrom, i, n, N, p, E*nN, L) * nN;
    swap(a, r);
    i += n;
  }
  swap(a, r);
  return int(l);
}


// Find pagerank of components in topological order (pull, CSR, skip-comp).
// @param w  previous graph
// @param wt previous transpose graph, with vertex-data=out-degree
// @param x  current graph
// @param xt current transpose graph, with vertex-data=out-degree
// @param q initial ranks (optional)
// @param o options {damping=0.85, tolerance=1e-6, maxIterations=500}
// @returns {ranks, iterations, time}
template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwise(const G& w, const H& wt, const G& x, const H& xt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  T    p = o.damping;
  T    E = o.tolerance;
  int  L = o.maxIterations, l;
  int  N = xt.order();
  auto wcs = pagerankComponents(w, wt);
  auto xcs = pagerankComponents(x, xt);
  auto ws = pagerankWaves(w, wt, wcs, x, xt, xcs);
  auto cs = pagerankGroupComponents(xcs, ws);
  auto ns = pagerankGroupWaves(cs);
  auto ks = join(cs);
  auto vfrom = sourceOffsets(xt, ks);
  auto efrom = destinationIndices(xt, ks);
  auto vdata = vertexData(xt, ks);
  vector<T> a(N), r(N), c(N), f(N), qc;
  if (q) qc = compressContainer(xt, *q, ks);
  float t = measureDurationMarked([&](auto mark) {
    fill(a, T());
    if (q) copy(r, qc);
    else fill(r, T(1)/N);
    mark([&] { pagerankFactor(f, vdata, 0, N, p); multiply(c, r, f); copy(a, r); });
    mark([&] { l = pagerankLevelwiseLoop(a, r, c, f, vfrom, efrom, 0, ns, N, p, E, L); });
  }, o.repeat);
  return {decompressContainer(xt, a, ks), l, t};
}


// Find pagerank of components in topological order (pull, CSR).
// @param x  current graph
// @param xt current transpose graph, with vertex-data=out-degree
// @param q initial ranks (optional)
// @param o options {damping=0.85, tolerance=1e-6, maxIterations=500}
// @returns {ranks, iterations, time}
template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwise(const G& x, const H& xt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  DiGraph<> w; DiGraph<int> wt;
  return pagerankLevelwise(w, wt, x, xt, q, o);
}
