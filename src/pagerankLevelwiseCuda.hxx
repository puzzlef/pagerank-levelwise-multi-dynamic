#pragma once
#include <vector>
#include <algorithm>
#include <utility>
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "components.hxx"
#include "blockgraph.hxx"
#include "topologicalSort.hxx"
#include "pagerank.hxx"
#include "pagerankMonolithicCuda.hxx"

using std::pair;
using std::vector;
using std::swap;




template <class G, class H, class T>
auto pagerankComponentsCuda(const G& x, const H& xt, const PagerankOptions<T>& o) {
  auto a  = joinUntilSize(components(x, xt), o.minComponentSize);
  auto fp = [&](int u) { return xt.degree(u) < PAGERANK_SWITCH_DEGREE; };
  for (auto& c : a) partition(c.begin(), c.end(), fp);
  auto b   = blockgraph(x, a);
  auto bks = topologicalSort(b);
  reorder(a, bks);
  return a;
}


template <class G, class H, class C>
auto pagerankComponentWaves(const G& w, const H& wt, const C& wcs, const G& x, const H& xt, const C& xcs) {
  int W = wcs.size();
  int X = xcs.size();
  auto b = blockgraph(x, xcs);
  vector<bool> dirty(X);
  for (int u : b.vertices()) {
    if (dirty[u]) continue;
    if (findIndex(wcs, xcs[u])>=0 && componentsEqual(w, wt, xcs[u], x, xt, xcs[u])) continue;
    dfsDo(b, u, [&](int v) { dirty[v] = true; });
  }
  vector<pair<int, vector<int>>> a;
  for (int i=0; i<X; i++) {
    int n = xcs[i].size();
    a.push_back({dirty[i]? n:n, pagerankWave(xt, xcs[i])});
  }
  return a;
}


template <class T, class J>
int pagerankLevelwiseCudaLoop(T *e, T *r0, T *eD, T *r0D, T *&aD, T *&rD, T *cD, const T *fD, const int *vfromD, const int *efromD, const int *vdataD, int i, J&& ws, int N, T p, T E, int L) {
  float l = 0;
  for (const auto& w : ws) {
    int n = w.first; const auto& ns = w.second;
    if (n<0) { i += -n; continue; }
    l += pagerankMonolithicCudaLoop(e, r0, eD, r0D, aD, rD, cD, fD, vfromD, efromD, vdataD, i, ns, N, p, E * (float(n)/N), L) * (float(n)/N);
    swap(aD, rD);
    i += n;
  }
  swap(aD, rD);
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
PagerankResult<T> pagerankLevelwiseCuda(const G& w, const H& wt, const G& x, const H& xt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  T    p = o.damping;
  T    E = o.tolerance;
  int  L = o.maxIterations, l;
  int  N = xt.order();
  int  R = reduceSizeCu(N);
  auto wcs = pagerankComponentsCuda(w, wt, o);
  auto xcs = pagerankComponentsCuda(x, xt, o);
  auto ws = pagerankComponentWaves(w, wt, wcs, x, xt, xcs);
  auto ks = join(xcs);
  auto vfrom = sourceOffsets(xt, ks);
  auto efrom = destinationIndices(xt, ks);
  auto vdata = vertexData(xt, ks);
  int VFROM1 = vfrom.size() * sizeof(int);
  int EFROM1 = efrom.size() * sizeof(int);
  int VDATA1 = vdata.size() * sizeof(int);
  int N1 = N * sizeof(T);
  int R1 = R * sizeof(T);
  vector<T> a(N), r(N);

  T *e,  *r0;
  T *eD, *r0D, *fD, *rD, *cD, *aD;
  int *vfromD, *efromD, *vdataD;
  // TRY( cudaProfilerStart() );
  TRY( cudaSetDeviceFlags(cudaDeviceMapHost) );
  TRY( cudaHostAlloc(&e,  R1, cudaHostAllocDefault) );
  TRY( cudaHostAlloc(&r0, R1, cudaHostAllocDefault) );
  TRY( cudaMalloc(&eD,  R1) );
  TRY( cudaMalloc(&r0D, R1) );
  TRY( cudaMalloc(&aD, N1) );
  TRY( cudaMalloc(&rD, N1) );
  TRY( cudaMalloc(&cD, N1) );
  TRY( cudaMalloc(&fD, N1) );
  TRY( cudaMalloc(&vfromD, VFROM1) );
  TRY( cudaMalloc(&efromD, EFROM1) );
  TRY( cudaMalloc(&vdataD, VDATA1) );
  TRY( cudaMemcpy(vfromD, vfrom.data(), VFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(efromD, efrom.data(), EFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(vdataD, vdata.data(), VDATA1, cudaMemcpyHostToDevice) );

  float t = measureDurationMarked([&](auto mark) {
    if (q) r = compressContainer(xt, *q, ks);
    else fill(r, T(1)/N);
    TRY( cudaMemcpy(aD, a.data(), N1, cudaMemcpyHostToDevice) );
    TRY( cudaMemcpy(rD, r.data(), N1, cudaMemcpyHostToDevice) );
    mark([&] { pagerankFactorCu(fD, vdataD, 0, N, p); });
    mark([&] { l = pagerankLevelwiseCudaLoop(e, r0, eD, r0D, aD, rD, cD, fD, vfromD, efromD, vdataD, 0, ws, N, p, E, L); });
  }, o.repeat);
  TRY( cudaMemcpy(a.data(), aD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFreeHost(e) );
  TRY( cudaFreeHost(r0) );
  TRY( cudaFree(eD) );
  TRY( cudaFree(r0D) );
  TRY( cudaFree(aD) );
  TRY( cudaFree(rD) );
  TRY( cudaFree(cD) );
  TRY( cudaFree(fD) );
  TRY( cudaFree(vfromD) );
  TRY( cudaFree(efromD) );
  TRY( cudaFree(vdataD) );
  // TRY( cudaProfilerStop() );
  return {decompressContainer(xt, a, ks), l, t};
}


// Find pagerank of components in topological order (pull, CSR).
// @param x  current graph
// @param xt current transpose graph, with vertex-data=out-degree
// @param q initial ranks (optional)
// @param o options {damping=0.85, tolerance=1e-6, maxIterations=500}
// @returns {ranks, iterations, time}
template <class G, class H, class T=float>
PagerankResult<T> pagerankLevelwiseCuda(const G& x, const H& xt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  DiGraph<> w; DiGraph<int> wt;
  return pagerankLevelwiseCuda(w, wt, x, xt, q, o);
}
