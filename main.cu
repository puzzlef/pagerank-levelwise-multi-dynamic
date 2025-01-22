#include <cmath>
#include <vector>
#include <cstdio>
#include <iostream>
#include <utility>
#include <random>
#include <omp.h>
#include "src/main.hxx"

using namespace std;




#define MAX_THREADS 64
#define MIN_COMPUTE_CUDA 10000000




template <class G, class T>
void printRow(const G& x, const PagerankResult<T>& a, const PagerankResult<T>& b, const char *tec) {
  auto e = l1NormOmp(b.ranks, a.ranks);
  print(x); printf(" [%09.3f ms; %03d iters.] [%.4e err.] %s\n", b.time, b.iterations, e, tec);
}


template <class G>
void runPagerankBatch(const G& xo, int repeat, int steps, float batchFraction) {
  using T = float;
  using K = typename G::key_type;
  using E = typename G::edge_value_type;
  enum NormFunction { L0=0, L1=1, L2=2, Li=3 };
  int span = int(1 * xo.span());
  vector<T> r0, s0, r1, s1;
  vector<T> *init = nullptr;
  random_device dev;
  default_random_engine rnd(dev());

  for (int i=0; i<steps; i++) {
    auto x  = addSelfLoopsOmp(xo, E(), [&](int u) { return isDeadEnd(xo, u); });
    auto xt = transposeWithDegreeOmp(x);
    auto ksOld = vertices(x);
    auto a0 = pagerankMonolithicOmp(x, xt, init, {repeat});
    auto r0 = a0.ranks;

    // Add random edges for this batch.
    int batch = int(ceil(batchFraction * x.size()));
    auto yo = duplicate(xo);
    for (int i=0; i<int(ceil(0.2*batch)); i++)
      removeRandomEdge(yo, rnd);
    for (int i=0; i<int(ceil(0.8*batch)); i++)
      addRandomEdge(yo, rnd, span);
    // for (int i=0; i<batch; i++)
    //   addRandomEdgeByDegree(yo, rnd, span);
    updateOmpU(yo);
    auto y  = addSelfLoopsOmp(yo, E(), [&](int u) { return isDeadEnd(yo, u); });
    auto yt = transposeWithDegreeOmp(y);
    auto ks = vertices(y);
    vector<T> s0(y.span());
    int X = ksOld.size();
    int Y = ks.size();

    // INSERTIONS + DELETIONS:
    // Adjust ranks for insertions + deletions.
    adjustRanks(s0, r0, ksOld, ks, 0.0f, float(X)/(Y+1), 1.0f/(Y+1));

    // Find Pagerank data.
    auto cs  = components(y, yt);
    auto b   = blockgraph(y, cs);
    auto bt  = transposeOmp(b);
    auto gs  = levelwiseGroupedComponentsFrom(cs, bt);
    auto [yks, yn] = dynamicVertices(x, xt, y, yt);
    auto [ycs, ym] = dynamicComponentIndices(x, xt, y, yt, cs, b);
    PagerankData<G> D {move(b), move(bt), move(cs)};
    printf("- batch update size: %d\n", batch);
    printf("- components: %d\n", b.order());
    printf("- blockgraph-levels: %zu\n", gs.size());
    printf("- affected-vertices: %zu\n", yn);
    printf("- affected-components: %zu\n", ym);

    // Find nvGraph-based pagerank.
    auto b0 = pagerankMonolithicOmp(y, yt, init, {repeat});
    // printRow(y, b0, b0, "pagerankNvgraph (static)");
    // auto c0 = pagerankNvgraph(y, yt, &s0, {repeat});
    // printRow(y, b0, c0, "pagerankNvgraph (incremental)");

    // Find OpenMP-based Monolithic pagerank.
    // auto b2 = pagerankMonolithicOmp(y, yt, init, {repeat, Li}, &D);
    // printRow(y, b0, b2, "pagerankMonolithicOmp (static)");
    // auto c2 = pagerankMonolithicOmp(y, yt, &s0, {repeat, Li}, &D);
    // printRow(y, b0, c2, "pagerankMonolithicOmp (incremental)");
    // auto d2 = pagerankMonolithicOmpDynamic(x, xt, y, yt, &s0, {repeat, Li}, &D);
    // printRow(y, b0, d2, "pagerankMonolithicOmp (dynamic)");

    // Find OpenMP-based Monolithic pagerank (split).
    auto h2 = pagerankMonolithicOmp(y, yt, init, {repeat, Li, 1, true}, &D);
    printRow(y, b0, h2, "pagerankMonolithicOmpSplit (static)");
    auto i2 = pagerankMonolithicOmp(y, yt, &s0, {repeat, Li, 1, true}, &D);
    printRow(y, b0, i2, "pagerankMonolithicOmpSplit (incremental)");
    auto j2 = pagerankMonolithicOmpDynamic(x, xt, y, yt, &s0, {repeat, Li, 1, true}, &D);
    printRow(y, b0, j2, "pagerankMonolithicOmpSplit (dynamic)");

    // Find CUDA-based Monolithic pagerank.
    // auto b3 = pagerankMonolithicCuda(y, yt, init, {repeat, Li, MIN_COMPUTE_CUDA}, &D);
    // printRow(y, b0, b3, "pagerankMonolithicCuda (static)");
    // auto c3 = pagerankMonolithicCuda(y, yt, &s0, {repeat, Li, MIN_COMPUTE_CUDA}, &D);
    // printRow(y, b0, c3, "pagerankMonolithicCuda (incremental)");
    // auto d3 = pagerankMonolithicCudaDynamic(x, xt, y, yt, &s0, {repeat, Li, MIN_COMPUTE_CUDA}, &D);
    // printRow(y, b0, d3, "pagerankMonolithicCuda (dynamic)");

    // Find CUDA-based Monolithic pagerank (split).
    // auto h3 = pagerankMonolithicCuda(y, yt, init, {repeat, Li, MIN_COMPUTE_CUDA, true}, &D);
    // printRow(y, b0, h3, "pagerankMonolithicCudaSplit (static)");
    // auto i3 = pagerankMonolithicCuda(y, yt, &s0, {repeat, Li, MIN_COMPUTE_CUDA, true}, &D);
    // printRow(y, b0, i3, "pagerankMonolithicCudaSplit (incremental)");
    // auto j3 = pagerankMonolithicCudaDynamic(x, xt, y, yt, &s0, {repeat, Li, MIN_COMPUTE_CUDA, true}, &D);
    // printRow(y, b0, j3, "pagerankMonolithicCudaSplit (dynamic)");

    // Find OpenMP-based Levelwise pagerank.
    auto b5 = pagerankLevelwiseOmp(y, yt, init, {repeat, Li}, &D);
    printRow(y, b0, b5, "pagerankLevelwiseOmp (static)");
    auto c5 = pagerankLevelwiseOmp(y, yt, &s0, {repeat, Li}, &D);
    printRow(y, b0, c5, "pagerankLevelwiseOmp (incremental)");
    auto d5 = pagerankLevelwiseOmpDynamic(x, xt, y, yt, &s0, {repeat, Li}, &D);
    printRow(y, b0, d5, "pagerankLevelwiseOmp (dynamic)");

    // Find CUDA-based Levelwise pagerank.
    // auto b6 = pagerankLevelwiseCuda(y, yt, init, {repeat, Li}, &D);
    // printRow(y, b0, b6, "pagerankLevelwiseCuda (static)");
    // auto c6 = pagerankLevelwiseCuda(y, yt, &s0, {repeat, Li}, &D);
    // printRow(y, b0, c6, "pagerankLevelwiseCuda (incremental)");
    // auto d6 = pagerankLevelwiseCudaDynamic(x, xt, y, yt, &s0, {repeat, Li}, &D);
    // printRow(y, b0, d6, "pagerankLevelwiseCuda (dynamic)");
  }
}


template <class G>
void runPagerank(const G& x, int repeat) {
  vector<float> batches {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
  int M = x.size(), steps = 5;
  for (float batch : batches) {
    printf("\n# Batch size %.0e\n", (double) batch);
    runPagerankBatch(x, repeat, steps, batch);
  }
}


int main(int argc, char **argv) {
  char *file = argv[1];
  int repeat = argc>2? stoi(argv[2]) : 5;
  omp_set_num_threads(MAX_THREADS);
  printf("Loading graph %s ...\n", file);
  DiGraph<> x;
  readMtxOmpW(x, file); println(x);
  runPagerank(x, repeat);
  printf("\n");
  return 0;
}
