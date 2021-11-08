#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "transpose.hxx"
#include "dynamic.hxx"
#include "pagerank.hxx"
#include "pagerankCuda.hxx"

using std::vector;
using std::sqrt;
using std::partition;
using std::swap;
using std::min;
using std::max;




// PAGERANK-LOOP
// -------------

template <class T, class J>
int pagerankMonolithicCudaLoop(T *e, T *r0, T *eD, T *r0D, T *&aD, T *&rD, T *cD, const T *fD, const int *vfromD, const int *efromD, const int *vdataD, int i, const J&& ns, int N, T p, T E, int L, int EF) {
  int n = sumAbs(ns);
  int R = reduceSizeCu<T>(n);
  size_t R1 = R * sizeof(T);
  T  c0 = (1-p)/N;
  int l = 0;
  while (l<L) {
    pagerankSwitchedCu(aD, cD, vfromD, efromD, i, ns, c0);  // assume contribtions (cD) is precalculated
    pagerankErrorCu(eD, aD+i, rD+i, n, EF);
    TRY( cudaMemcpy(e, eD, R1, cudaMemcpyDeviceToHost) );
    T el = pagerankErrorReduce(e, R, EF); ++l;              // one iteration complete
    if (el<E || l>=L) break;                                // check tolerance, iteration limit
    multiplyCu(cD+i, aD+i, fD+i, n);                        // update partial contributions (cD)
    swap(aD, rD);                                           // final ranks always in (aD)
  }
  return l;
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
PagerankResult<T> pagerankMonolithicCuda(const G& x, const H& xt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto fm = [](int u) { return u; };
  auto fp = [&](auto ib, auto ie) {
    partition(ib, ie, [&](int u) { return xt.degree(u) < SWITCH_DEGREE_PRC(); });
  };
  auto ks = vertices(xt, fm, fp); vector2d<int> cs {ks};
  auto ns = pagerankWave(xt, cs);
  return pagerankCuda(xt, ks, 0, ns, pagerankMonolithicCudaLoop<T>, q, o);
}

template <class G, class T=float>
PagerankResult<T> pagerankMonolithicCuda(const G& x, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto xt = transposeWithDegree(x);
  return pagerankMonolithicCuda(x, xt, q, o);
}
