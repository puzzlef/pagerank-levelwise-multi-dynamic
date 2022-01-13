#include <cmath>
#include <vector>
#include <cstdio>
#include <iostream>
#include <utility>
#include <random>
#include "src/main.hxx"

using namespace std;




#define MIN_COMPUTE_CUDA 10000000

template <class G, class T>
void printRow(const G& x, const PagerankResult<T>& a, const PagerankResult<T>& b, const char *tec) {
  auto e = l1Norm(b.ranks, a.ranks);
  print(x); printf(" [%09.3f ms; %03d iters.] [%.4e err.] %s\n", b.time, b.iterations, e, tec);
}

template <class G>
void runPagerankBatch(const G& xo, int repeat, int steps, int batch) {
  using T = float;
  enum NormFunction { L0=0, L1=1, L2=2, Li=3 };
  int span = int(1.1 * xo.span());
  vector<T> r0, s0, r1, s1;
  vector<T> *init = nullptr;
  random_device dev;
  default_random_engine rnd(dev());

  for (int i=0; i<steps; i++) {
    auto x  = selfLoop(xo, [&](int u) { return isDeadEnd(xo, u); });
    auto xt = transposeWithDegree(x);
    auto ksOld = vertices(x);
    auto a0 = pagerankNvgraph(x, xt, init, {repeat});
    auto r0 = a0.ranks;

    // Add random edges for this batch.
    auto yo = copy(xo);
    for (int i=0; i<batch; i++)
      addRandomEdgeByDegree(yo, rnd, span);
    auto y  = selfLoop(yo, [&](int u) { return isDeadEnd(yo, u); });
    auto yt = transposeWithDegree(y);
    auto ks = vertices(y);
    vector<T> s0(y.span());
    int X = ksOld.size();
    int Y = ks.size();

    // INSERTIONS:
    // Adjust ranks for insertions.
    adjustRanks(s0, r0, ksOld, ks, 0.0f, float(X)/(Y+1), 1.0f/(Y+1));

    // Find Pagerank data.
    auto cs  = components(y, yt);
    auto b   = blockgraph(y, cs);
    auto bt  = transpose(b);
    PagerankData<G> D {move(b), move(bt), move(cs)};

    // Find nvGraph-based pagerank.
    auto b0 = pagerankNvgraph(y, yt, init, {repeat});
    printRow(y, b0, b0, "I:pagerankNvgraph (static)");
    auto c0 = pagerankNvgraph(y, yt, &s0, {repeat});
    printRow(y, b0, c0, "I:pagerankNvgraph (incremental)");

    // Find sequential Monolithic pagerank.
    // auto b1 = pagerankMonolithicSeq(y, yt, init, {repeat, Li}, &D);
    // printRow(y, b0, b1, "I:pagerankMonolithicSeq (static)");
    // auto c1 = pagerankMonolithicSeq(y, yt, &s0, {repeat, Li}, &D);
    // printRow(y, b0, c1, "I:pagerankMonolithicSeq (incremental)");
    // auto d1 = pagerankMonolithicSeqDynamic(x, xt, y, yt, &s0, {repeat, Li}, &D);
    // printRow(y, b0, d1, "I:pagerankMonolithicSeq (dynamic)");

    // Find sequential Monolithic pagerank (split).
    auto h1 = pagerankMonolithicSeq(y, yt, init, {repeat, Li, 1, true}, &D);
    printRow(y, b0, h1, "I:pagerankMonolithicSeqSplit (static)");
    auto i1 = pagerankMonolithicSeq(y, yt, &s0, {repeat, Li, 1, true}, &D);
    printRow(y, b0, i1, "I:pagerankMonolithicSeqSplit (incremental)");
    auto j1 = pagerankMonolithicSeqDynamic(x, xt, y, yt, &s0, {repeat, Li, 1, true}, &D);
    printRow(y, b0, j1, "I:pagerankMonolithicSeqSplit (dynamic)");

    // Find OpenMP-based Monolithic pagerank.
    // auto b2 = pagerankMonolithicOmp(y, yt, init, {repeat, Li}, &D);
    // printRow(y, b0, b2, "I:pagerankMonolithicOmp (static)");
    // auto c2 = pagerankMonolithicOmp(y, yt, &s0, {repeat, Li}, &D);
    // printRow(y, b0, c2, "I:pagerankMonolithicOmp (incremental)");
    // auto d2 = pagerankMonolithicOmpDynamic(x, xt, y, yt, &s0, {repeat, Li}, &D);
    // printRow(y, b0, d2, "I:pagerankMonolithicOmp (dynamic)");

    // Find OpenMP-based Monolithic pagerank (split).
    // auto h2 = pagerankMonolithicOmp(y, yt, init, {repeat, Li, 1, true}, &D);
    // printRow(y, b0, h2, "I:pagerankMonolithicOmpSplit (static)");
    // auto i2 = pagerankMonolithicOmp(y, yt, &s0, {repeat, Li, 1, true}, &D);
    // printRow(y, b0, i2, "I:pagerankMonolithicOmpSplit (incremental)");
    // auto j2 = pagerankMonolithicOmpDynamic(x, xt, y, yt, &s0, {repeat, Li, 1, true}, &D);
    // printRow(y, b0, j2, "I:pagerankMonolithicOmpSplit (dynamic)");

    // Find CUDA-based Monolithic pagerank.
    // auto b3 = pagerankMonolithicCuda(y, yt, init, {repeat, Li, MIN_COMPUTE_CUDA}, &D);
    // printRow(y, b0, b3, "I:pagerankMonolithicCuda (static)");
    // auto c3 = pagerankMonolithicCuda(y, yt, &s0, {repeat, Li, MIN_COMPUTE_CUDA}, &D);
    // printRow(y, b0, c3, "I:pagerankMonolithicCuda (incremental)");
    // auto d3 = pagerankMonolithicCudaDynamic(x, xt, y, yt, &s0, {repeat, Li, MIN_COMPUTE_CUDA}, &D);
    // printRow(y, b0, d3, "I:pagerankMonolithicCuda (dynamic)");

    // Find CUDA-based Monolithic pagerank (split).
    auto h3 = pagerankMonolithicCuda(y, yt, init, {repeat, Li, MIN_COMPUTE_CUDA, true}, &D);
    printRow(y, b0, h3, "I:pagerankMonolithicCudaSplit (static)");
    auto i3 = pagerankMonolithicCuda(y, yt, &s0, {repeat, Li, MIN_COMPUTE_CUDA, true}, &D);
    printRow(y, b0, i3, "I:pagerankMonolithicCudaSplit (incremental)");
    auto j3 = pagerankMonolithicCudaDynamic(x, xt, y, yt, &s0, {repeat, Li, MIN_COMPUTE_CUDA, true}, &D);
    printRow(y, b0, j3, "I:pagerankMonolithicCudaSplit (dynamic)");

    // Find sequential Levelwise pagerank.
    auto b4 = pagerankLevelwiseSeq(y, yt, init, {repeat, Li}, &D);
    printRow(y, b0, b4, "I:pagerankLevelwiseSeq (static)");
    auto c4 = pagerankLevelwiseSeq(y, yt, &s0, {repeat, Li}, &D);
    printRow(y, b0, c4, "I:pagerankLevelwiseSeq (incremental)");
    auto d4 = pagerankLevelwiseSeqDynamic(x, xt, y, yt, &s0, {repeat, Li}, &D);
    printRow(y, b0, d4, "I:pagerankLevelwiseSeq (dynamic)");

    // Find OpenMP-based Levelwise pagerank.
    // auto b5 = pagerankLevelwiseOmp(y, yt, init, {repeat, Li}, &D);
    // printRow(y, b0, b5, "I:pagerankLevelwiseOmp (static)");
    // auto c5 = pagerankLevelwiseOmp(y, yt, &s0, {repeat, Li}, &D);
    // printRow(y, b0, c5, "I:pagerankLevelwiseOmp (incremental)");
    // auto d5 = pagerankLevelwiseOmpDynamic(x, xt, y, yt, &s0, {repeat, Li}, &D);
    // printRow(y, b0, d5, "I:pagerankLevelwiseOmp (dynamic)");

    // Find CUDA-based Levelwise pagerank.
    auto b6 = pagerankLevelwiseCuda(y, yt, init, {repeat, Li}, &D);
    printRow(y, b0, b6, "I:pagerankLevelwiseCuda (static)");
    auto c6 = pagerankLevelwiseCuda(y, yt, &s0, {repeat, Li}, &D);
    printRow(y, b0, c6, "I:pagerankLevelwiseCuda (incremental)");
    auto d6 = pagerankLevelwiseCudaDynamic(x, xt, y, yt, &s0, {repeat, Li}, &D);
    printRow(y, b0, d6, "I:pagerankLevelwiseCuda (dynamic)");

    // DELETIONS:
    // Adjust ranks for deletions.
    auto s1 = b0.ranks;
    vector<T> r1(x.span());
    adjustRanks(r1, s1, ks, ksOld, 0.0f, float(Y)/(X+1), 1.0f/(X+1));

    // Find Pagerank data.
    auto ds = components(x, xt);
    auto c  = blockgraph(x, ds);
    auto ct = transpose(c);
    PagerankData<G> E {move(c), move(ct), move(ds)};

    // Find nvGraph-based pagerank.
    auto e0 = pagerankNvgraph(x, xt, init, {repeat, Li});
    printRow(y, e0, e0, "D:pagerankNvgraph (static)");
    auto f0 = pagerankNvgraph(x, xt, &r1, {repeat, Li});
    printRow(y, e0, f0, "D:pagerankNvgraph (incremental)");

    // Find sequential Monolithic pagerank.
    // auto e1 = pagerankMonolithicSeq(x, xt, init, {repeat, Li}, &E);
    // printRow(y, e0, e1, "D:pagerankMonolithicSeq (static)");
    // auto f1 = pagerankMonolithicSeq(x, xt, &r1, {repeat, Li}, &E);
    // printRow(y, e0, f1, "D:pagerankMonolithicSeq (incremental)");
    // auto g1 = pagerankMonolithicSeqDynamic(y, yt, x, xt, &r1, {repeat, Li}, &E);
    // printRow(y, e0, g1, "D:pagerankMonolithicSeq (dynamic)");

    // Find sequential Monolithic pagerank (split).
    auto k1 = pagerankMonolithicSeq(x, xt, init, {repeat, Li, 1, true}, &E);
    printRow(y, e0, k1, "D:pagerankMonolithicSeqSplit (static)");
    auto l1 = pagerankMonolithicSeq(x, xt, &r1, {repeat, Li, 1, true}, &E);
    printRow(y, e0, l1, "D:pagerankMonolithicSeqSplit (incremental)");
    auto m1 = pagerankMonolithicSeqDynamic(y, yt, x, xt, &r1, {repeat, Li, 1, true}, &E);
    printRow(y, e0, m1, "D:pagerankMonolithicSeqSplit (dynamic)");

    // Find OpenMP-based Monolithic pagerank.
    // auto e2 = pagerankMonolithicOmp(x, xt, init, {repeat, Li}, &E);
    // printRow(y, e0, e2, "D:pagerankMonolithicOmp (static)");
    // auto f2 = pagerankMonolithicOmp(x, xt, &r1, {repeat, Li}, &E);
    // printRow(y, e0, f2, "D:pagerankMonolithicOmp (incremental)");
    // auto g2 = pagerankMonolithicOmpDynamic(y, yt, x, xt, &r1, {repeat, Li}, &E);
    // printRow(y, e0, g2, "D:pagerankMonolithicOmp (dynamic)");

    // Find OpenMP-based Monolithic pagerank.
    // auto k2 = pagerankMonolithicOmp(x, xt, init, {repeat, Li, 1, true}, &E);
    // printRow(y, e0, k2, "D:pagerankMonolithicOmpSplit (static)");
    // auto l2 = pagerankMonolithicOmp(x, xt, &r1, {repeat, Li, 1, true}, &E);
    // printRow(y, e0, l2, "D:pagerankMonolithicOmpSplit (incremental)");
    // auto m2 = pagerankMonolithicOmpDynamic(y, yt, x, xt, &r1, {repeat, Li, 1, true}, &E);
    // printRow(y, e0, m2, "D:pagerankMonolithicOmpSplit (dynamic)");

    // Find CUDA-based Monolithic pagerank.
    // auto e3 = pagerankMonolithicCuda(x, xt, init, {repeat, Li, MIN_COMPUTE_CUDA}, &E);
    // printRow(y, e0, e3, "D:pagerankMonolithicCuda (static)");
    // auto f3 = pagerankMonolithicCuda(x, xt, &r1, {repeat, Li, MIN_COMPUTE_CUDA}, &E);
    // printRow(y, e0, f3, "D:pagerankMonolithicCuda (incremental)");
    // auto g3 = pagerankMonolithicCudaDynamic(y, yt, x, xt, &r1, {repeat, Li, MIN_COMPUTE_CUDA}, &E);
    // printRow(y, e0, g3, "D:pagerankMonolithicCuda (dynamic)");

    // Find CUDA-based Monolithic pagerank (split).
    auto k3 = pagerankMonolithicCuda(x, xt, init, {repeat, Li, MIN_COMPUTE_CUDA, true}, &E);
    printRow(y, e0, k3, "D:pagerankMonolithicCudaSplit (static)");
    auto l3 = pagerankMonolithicCuda(x, xt, &r1, {repeat, Li, MIN_COMPUTE_CUDA, true}, &E);
    printRow(y, e0, l3, "D:pagerankMonolithicCudaSplit (incremental)");
    auto m3 = pagerankMonolithicCudaDynamic(y, yt, x, xt, &r1, {repeat, Li, MIN_COMPUTE_CUDA, true}, &E);
    printRow(y, e0, m3, "D:pagerankMonolithicCudaSplit (dynamic)");

    // Find sequential Levelwise pagerank.
    auto e4 = pagerankLevelwiseSeq(x, xt, init, {repeat, Li}, &E);
    printRow(y, e0, e4, "D:pagerankLevelwiseSeq (static)");
    auto f4 = pagerankLevelwiseSeq(x, xt, &r1, {repeat, Li}, &E);
    printRow(y, e0, f4, "D:pagerankLevelwiseSeq (incremental)");
    auto g4 = pagerankLevelwiseSeqDynamic(y, yt, x, xt, &r1, {repeat, Li}, &E);
    printRow(y, e0, g4, "D:pagerankLevelwiseSeq (dynamic)");

    // Find OpenMP-based Levelwise pagerank.
    // auto e5 = pagerankLevelwiseOmp(x, xt, init, {repeat, Li}, &E);
    // printRow(y, e0, e5, "D:pagerankLevelwiseOmp (static)");
    // auto f5 = pagerankLevelwiseOmp(x, xt, &r1, {repeat, Li}, &E);
    // printRow(y, e0, f5, "D:pagerankLevelwiseOmp (incremental)");
    // auto g5 = pagerankLevelwiseOmpDynamic(y, yt, x, xt, &r1, {repeat, Li}, &E);
    // printRow(y, e0, g5, "D:pagerankLevelwiseOmp (dynamic)");

    // Find CUDA-based Levelwise pagerank.
    auto e6 = pagerankLevelwiseCuda(x, xt, init, {repeat, Li}, &E);
    printRow(y, e0, e6, "D:pagerankLevelwiseCuda (static)");
    auto f6 = pagerankLevelwiseCuda(x, xt, &r1, {repeat, Li}, &E);
    printRow(y, e0, f6, "D:pagerankLevelwiseCuda (incremental)");
    auto g6 = pagerankLevelwiseCudaDynamic(y, yt, x, xt, &r1, {repeat, Li}, &E);
    printRow(y, e0, g6, "D:pagerankLevelwiseCuda (dynamic)");
  }
}


template <class G>
void runPagerank(const G& x, int repeat) {
  int M = x.size(), steps = 10;
  for (int batch=10, i=0; batch<=10000; batch*=i&1? 2:5, i++) {
    printf("\n# Batch size %.0e\n", (double) batch);
    runPagerankBatch(x, repeat, steps, batch);
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
