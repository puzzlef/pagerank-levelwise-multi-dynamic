#include <cmath>
#include <string>
#include <sstream>
#include <cstdio>
#include <iostream>
#include <utility>
#include "src/main.hxx"

using namespace std;




#define REPEAT 1

void runPagerankBatch(const string& data, bool show, int skip, int batch) {
  vector<float>  ranksAdj;
  vector<float> *initStatic  = nullptr;
  vector<float> *initDynamic = &ranksAdj;

  DiGraph<> x;
  stringstream s(data);
  while (true) {
    // Skip some edges (to speed up execution)
    if (!readSnapTemporal(x, s, skip)) break;
    loopDeadEnds(x);
    auto xt = transposeWithDegree(x);
    auto a1 = pagerankNvgraph(xt);
    auto ksOld    = vertices(x);
    auto ranksOld = move(a1.ranks);

    // Read edges for this batch.
    auto y = copy(x);
    if (!readSnapTemporal(y, s, batch)) break;
    loopDeadEnds(y);
    auto yt = transposeWithDegree(y);
    auto ks = vertices(y);

    // Adjust ranks using scaled-fill.
    ranksAdj.resize(y.span());
    adjustRanks(ranksAdj, ranksOld, ksOld, ks, 0.0f, float(ksOld.size())/ks.size(), 1.0f/ks.size());

    // Find static pagerank using nvGraph.
    auto a2 = pagerankNvgraph(yt, initStatic, {REPEAT});
    auto e2 = l1Norm(a2.ranks, a2.ranks);
    print(yt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankNvgraph [static]\n", a2.time, a2.iterations, e2);

    // Find dynamic pagerank using nvGraph.
    auto a3 = pagerankNvgraph(yt, initDynamic, {REPEAT});
    auto e3 = l1Norm(a3.ranks, a2.ranks);
    print(yt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankNvgraph [dynamic]\n", a3.time, a3.iterations, e3);

    // Find static pagerank (monolithic).
    auto a4 = pagerankMonolithic(yt, initStatic, {REPEAT});
    auto e4 = l1Norm(a4.ranks, a2.ranks);
    print(yt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankMonolithic [static]\n", a4.time, a4.iterations, e4);

    // Find dynamic pagerank (monolithic).
    auto a5 = pagerankMonolithic(yt, initDynamic, {REPEAT});
    auto e5 = l1Norm(a5.ranks, a2.ranks);
    print(yt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankMonolithic [dynamic]\n", a5.time, a5.iterations, e5);

    // Find static levelwise pagerank.
    auto a6 = pagerankLevelwise(y, yt, initStatic, {REPEAT});
    auto e6 = l1Norm(a6.ranks, a2.ranks);
    print(yt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankLevelwise [static]\n", a6.time, a6.iterations, e6);

    // Find dynamic levelwise pagerank.
    auto a7 = pagerankLevelwise(x, xt, y, yt, initDynamic, {REPEAT});
    auto e7 = l1Norm(a7.ranks, a2.ranks);
    print(yt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankLevelwise [dynamic]\n", a7.time, a7.iterations, e7);

    // Find CUDA based static pagerank (monolithic).
    auto a8 = pagerankMonolithicCuda(yt, initStatic, {REPEAT});
    auto e8 = l1Norm(a8.ranks, a2.ranks);
    print(yt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankMonolithicCuda [static]\n", a8.time, a8.iterations, e8);

    // Find CUDA based dynamic pagerank (monolithic).
    auto a9 = pagerankMonolithicCuda(yt, initDynamic, {REPEAT});
    auto e9 = l1Norm(a9.ranks, a2.ranks);
    print(yt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankMonolithicCuda [dynamic]\n", a9.time, a9.iterations, e9);

    // Find CUDA based static levelwise pagerank.
    auto a10 = pagerankLevelwiseCuda(y, yt, initStatic, {REPEAT});
    auto e10 = l1Norm(a10.ranks, a2.ranks);
    print(yt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankLevelwiseCuda [static]\n", a10.time, a10.iterations, e10);

    // Find CUDA based dynamic levelwise pagerank.
    auto a11 = pagerankLevelwiseCuda(x, xt, y, yt, initDynamic, {REPEAT});
    auto e11 = l1Norm(a11.ranks, a2.ranks);
    print(yt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankLevelwiseCuda [dynamic]\n", a11.time, a11.iterations, e11);

    x = move(y);
  }
}


void runPagerank(const string& data, bool show) {
  int M = countLines(data), steps = 100;
  printf("Temporal edges: %d\n", M);
  for (int batch=100, i=0; batch<M; batch*=i&1? 2:5, i++) {
    int skip = max(M/steps - batch, 0);
    printf("\n# Batch size %.0e\n", (double) batch);
    runPagerankBatch(data, show, skip, batch);
  }
}


int main(int argc, char **argv) {
  char *file = argv[1];
  bool  show = argc > 2;
  printf("Using graph %s ...\n", file);
  string d = readFile(file);
  runPagerank(d, show);
    printf("\n");
  return 0;
}
