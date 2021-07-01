#pragma once
#include <vector>
#include <utility>
#include "_main.hxx"

using std::vector;
using std::move;




// LAUNCH CONFIG
// -------------

// For pagerank cuda block-per-vertex
#define BLOCK_DIM_PRB 64
#define GRID_DIM_PRB  GRID_LIMIT

// For pagerank cuda thread-per-vertex (default)
#define BLOCK_DIM_PRT 128
#define GRID_DIM_PRT  8192

// For pagerank cuda thread-per-vertex (low avg. density)
#define BLOCK_DIM_PRT_LOWDENSITY 512
#define GRID_DIM_PRT_LOWDENSITY  8192

// For pagerank cuda thread-per-vertex (high avg. degree)
#define BLOCK_DIM_PRT_HIGHDEGREE 32
#define GRID_DIM_PRT_HIGHDEGREE  8192

// For pagerank cuda switched (block approach)
#define BLOCK_DIM_PRSB 256
#define GRID_DIM_PRSB  GRID_LIMIT

// For pagerank cuda switched (thread approach)
#define BLOCK_DIM_PRST 512
#define GRID_DIM_PRST  GRID_LIMIT

// For pagerank cuda switched switch-point
#define SWITCH_DEGREE_PR 64
#define SWITCH_LIMIT_PR  32

// For levelwise pagerank
#define MIN_COMPUTE_SIZE_PR 5000000




// PAGERANK-OPTIONS
// ----------------

template <class T>
struct PagerankOptions {
  int repeat;
  T   damping;
  T   tolerance;
  int maxIterations;

  PagerankOptions(int repeat=1, T damping=0.85, T tolerance=1e-6, int maxIterations=500) :
  repeat(repeat), damping(damping), tolerance(tolerance), maxIterations(maxIterations) {}
};




// PAGERANK-RESULT
// ---------------

template <class T>
struct PagerankResult {
  vector<T> ranks;
  int   iterations;
  float time;

  PagerankResult(vector<T>&& ranks, int iterations=0, float time=0) :
  ranks(ranks), iterations(iterations), time(time) {}

  PagerankResult(vector<T>& ranks, int iterations=0, float time=0) :
  ranks(move(ranks)), iterations(iterations), time(time) {}
};
