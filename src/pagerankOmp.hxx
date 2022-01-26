#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "pagerank.hxx"
#include "pagerankSeq.hxx"

using std::vector;
using std::swap;




// PAGERANK-FACTOR
// ---------------
// For contribution factors of vertices (unchanging).

template <class T>
void pagerankFactorOmp(vector<T>& a, const vector<int>& vdata, int i, int n, T p) {
  #pragma omp parallel for num_threads(32) schedule(auto)
  for (int u=i; u<i+n; u++) {
    int d = vdata[u];
    a[u] = d>0? p/d : 0;
  }
}




// PAGERANK-CALCULATE
// ------------------
// For rank calculation from in-edges.

template <class T>
void pagerankCalculateOmp(vector<T>& a, const vector<T>& c, const vector<int>& vfrom, const vector<int>& efrom, int i, int n, T c0) {
  #pragma omp parallel for num_threads(32) schedule(auto)
  for (int v=i; v<i+n; v++)
    a[v] = c0 + sumAt(c, sliceIter(efrom, vfrom[v], vfrom[v+1]));
}




// PAGERANK-ERROR
// --------------
// For convergence check.

template <class T>
T pagerankErrorOmp(const vector<T>& x, const vector<T>& y, int i, int N, int EF) {
  switch (EF) {
    case 1:  return l1NormOmp(x, y, i, N);
    case 2:  return l2NormOmp(x, y, i, N);
    default: return liNormOmp(x, y, i, N);
  }
}




// PAGERANK
// --------
// For Monolithic / Componentwise PageRank.

template <class H, class J, class M, class FL, class T=float>
PagerankResult<T> pagerankOmp(const H& xt, const J& ks, int i, const M& ns, FL fl, const vector<T> *q, const PagerankOptions<T>& o) {
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
    if (q) copyOmp(r, qc);    // copy old ranks (qc), if given
    else fillOmp(r, T(1)/N);
    copyOmp(a, r);
    mark([&] { pagerankFactorOmp(f, vdata, 0, N, p); multiplyOmp(c, a, f, 0, N); });  // calculate factors (f) and contributions (c)
    mark([&] { l = fl(a, r, c, f, vfrom, efrom, i, ns, N, p, E, L, EF); });           // calculate ranks of vertices
  }, o.repeat);
  return {decompressContainer(xt, a, ks), l, t};
}
