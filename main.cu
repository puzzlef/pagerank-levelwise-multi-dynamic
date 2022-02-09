#include <cmath>
#include <vector>
#include <cstdio>
#include <iostream>
#include <utility>
#include <random>
#include "src/main.hxx"

using namespace std;




#define MIN_COMPUTE_CUDA 10000000


struct GraphDelta {
  vector<pair<int, int>> deletions;
  vector<pair<int, int>> insertions;
};


template <class G, class T>
void printRow(const G& x, const PagerankResult<T>& a, const PagerankResult<T>& b, const char *tec) {
  auto e = l1Norm(b.ranks, a.ranks);
  print(x); printf(" [%09.3f ms; %03d iters.] [%.4e err.] %s\n", b.time, b.iterations, e, tec);
}

template <class G>
void runPagerankBatch(const G& xr, const GraphDelta& delta, int repeat, int batch) {
  using T = float;
  enum NormFunction { L0=0, L1=1, L2=2, Li=3 };
  vector<T> r0, s0, r1, s1;
  vector<T> *init = nullptr;
  int DD = delta.deletions.size();
  int DI = delta.insertions.size();
  auto xo = copy(xr);

  for (int di=0, ii=0; di<DD || ii<DI;) {
    auto x  = selfLoop(xo, [&](int u) { return isDeadEnd(xo, u); });
    auto xt = transposeWithDegree(x);
    auto ksOld = vertices(x);
    auto a0 = pagerankNvgraph(x, xt, init, {repeat});
    auto r0 = a0.ranks;

    // Add random edges for this batch.
    auto yo = copy(xo);
    int  db = di<=ii? min(ceilDiv(batch, 2), DD-di) : 0;
    int  ib = min(batch-db, DI-ii);
    for (int j=0; j<db; j++)
      removeEdge(yo, delta.deletions[di++]);
    for (int j=0; j<ib; j++)
      addEdge(yo, delta.insertions[ii++]);
    yo.correct();
    auto y  = selfLoop(yo, [&](int u) { return isDeadEnd(yo, u); });
    auto yt = transposeWithDegree(y);
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
    auto bt  = transpose(b);
    auto gs  = levelwiseGroupedComponentsFrom(cs, bt);
    auto [yks, yn] = dynamicVertices(x, xt, y, yt);
    auto [ycs, ym] = dynamicComponentIndices(x, xt, y, yt, cs, b);
    PagerankData<G> D {move(b), move(bt), move(cs)};
    printf("- components: %d\n", b.order());
    printf("- blockgraph-levels: %d\n", gs.size());
    printf("- affected-vertices: %d\n", yn);
    printf("- affected-components: %d\n", ym);

    // Find nvGraph-based pagerank.
    auto b0 = pagerankNvgraph(y, yt, init, {repeat});
    printRow(y, b0, b0, "pagerankNvgraph (static)");
    auto c0 = pagerankNvgraph(y, yt, &s0, {repeat});
    printRow(y, b0, c0, "pagerankNvgraph (incremental)");

    // Find sequential Monolithic pagerank.
    // auto b1 = pagerankMonolithicSeq(y, yt, init, {repeat, Li}, &D);
    // printRow(y, b0, b1, "pagerankMonolithicSeq (static)");
    // auto c1 = pagerankMonolithicSeq(y, yt, &s0, {repeat, Li}, &D);
    // printRow(y, b0, c1, "pagerankMonolithicSeq (incremental)");
    // auto d1 = pagerankMonolithicSeqDynamic(x, xt, y, yt, &s0, {repeat, Li}, &D);
    // printRow(y, b0, d1, "pagerankMonolithicSeq (dynamic)");

    // Find sequential Monolithic pagerank (split).
    // auto h1 = pagerankMonolithicSeq(y, yt, init, {repeat, Li, 1, true}, &D);
    // printRow(y, b0, h1, "pagerankMonolithicSeqSplit (static)");
    // auto i1 = pagerankMonolithicSeq(y, yt, &s0, {repeat, Li, 1, true}, &D);
    // printRow(y, b0, i1, "pagerankMonolithicSeqSplit (incremental)");
    // auto j1 = pagerankMonolithicSeqDynamic(x, xt, y, yt, &s0, {repeat, Li, 1, true}, &D);
    // printRow(y, b0, j1, "pagerankMonolithicSeqSplit (dynamic)");

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
    auto h3 = pagerankMonolithicCuda(y, yt, init, {repeat, Li, MIN_COMPUTE_CUDA, true}, &D);
    printRow(y, b0, h3, "pagerankMonolithicCudaSplit (static)");
    auto i3 = pagerankMonolithicCuda(y, yt, &s0, {repeat, Li, MIN_COMPUTE_CUDA, true}, &D);
    printRow(y, b0, i3, "pagerankMonolithicCudaSplit (incremental)");
    auto j3 = pagerankMonolithicCudaDynamic(x, xt, y, yt, &s0, {repeat, Li, MIN_COMPUTE_CUDA, true}, &D);
    printRow(y, b0, j3, "pagerankMonolithicCudaSplit (dynamic)");

    // Find sequential Levelwise pagerank.
    // auto b4 = pagerankLevelwiseSeq(y, yt, init, {repeat, Li}, &D);
    // printRow(y, b0, b4, "pagerankLevelwiseSeq (static)");
    // auto c4 = pagerankLevelwiseSeq(y, yt, &s0, {repeat, Li}, &D);
    // printRow(y, b0, c4, "pagerankLevelwiseSeq (incremental)");
    // auto d4 = pagerankLevelwiseSeqDynamic(x, xt, y, yt, &s0, {repeat, Li}, &D);
    // printRow(y, b0, d4, "pagerankLevelwiseSeq (dynamic)");

    // Find OpenMP-based Levelwise pagerank.
    auto b5 = pagerankLevelwiseOmp(y, yt, init, {repeat, Li}, &D);
    printRow(y, b0, b5, "pagerankLevelwiseOmp (static)");
    auto c5 = pagerankLevelwiseOmp(y, yt, &s0, {repeat, Li}, &D);
    printRow(y, b0, c5, "pagerankLevelwiseOmp (incremental)");
    auto d5 = pagerankLevelwiseOmpDynamic(x, xt, y, yt, &s0, {repeat, Li}, &D);
    printRow(y, b0, d5, "pagerankLevelwiseOmp (dynamic)");

    // Find CUDA-based Levelwise pagerank.
    auto b6 = pagerankLevelwiseCuda(y, yt, init, {repeat, Li}, &D);
    printRow(y, b0, b6, "pagerankLevelwiseCuda (static)");
    auto c6 = pagerankLevelwiseCuda(y, yt, &s0, {repeat, Li}, &D);
    printRow(y, b0, c6, "pagerankLevelwiseCuda (incremental)");
    auto d6 = pagerankLevelwiseCudaDynamic(x, xt, y, yt, &s0, {repeat, Li}, &D);
    printRow(y, b0, d6, "pagerankLevelwiseCuda (dynamic)");

    // Move ahead.
    xo = move(yo);
  }
}


template <class G>
auto createMixedGraphDelta(const G& x, int del, int ins) {
  GraphDelta a;
  random_device dev;
  default_random_engine rnd(dev());
  for (int i=0; i<del; ++i)
    a.deletions.push_back(suggestRemoveRandomEdgeByDegree(x, rnd));
  for (int i=0; i<ins; ++i)
    a.insertions.push_back(suggestAddRandomEdgeByDegree(x, rnd, x.span()));
  return a;
}

template <class G>
void runPagerank(const G& x, int repeat) {
  vector<int> batches {1, 500, 1000, 2000};
  int steps = 1; int B = batches.back();
  for (int step=0; step<steps; ++step) {
    // printf("\n# Step %d\n", step);
    GraphDelta delta = createMixedGraphDelta(x, B/2, B/2);
    for (int batch : batches) {
      printf("\n# Batch size %.0e\n", (double) batch);
      runPagerankBatch(x, delta, repeat, batch);
    }
  }
}


int main(int argc, char **argv) {
  char *file = argv[1];
  int repeat = argc>2? stoi(argv[2]) : 5;
  printf("Loading graph %s ...\n", file);
  auto x = readMtx(file); println(x);
  runPagerank(x, repeat);
  printf("\n");
  return 0;
}
