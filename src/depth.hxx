#pragma once
#include <tuple>
#include "dfs_old.hxx"
#include "deadEnds.hxx"

using std::make_tuple;




// DEPTH
// -----

template <class G>
int depth(const G& x, int u) {
  int a = 0; if (x.order()==0) return 0;
  dfsDepthDo(x, u, 0, [&](int v, int d) { if (d>a) a=d; });
  return ++a;
}




// MIN/MAX/AVG
// -----------

template <class G, class H>
int minDepth(const G& x, const H& xt) {
  int dmin = x.order(), D = 0;
  deadEndsForEach(xt, [&](int u) {
    int d = depth(x, u); ++D;
    if (d<dmin) dmin = d;
  });
  return D>0? dmin : 0;
}

template <class G, class H>
int maxDepth(const G& x, const H& xt) {
  int dmax = 0;
  deadEndsForEach(xt, [&](int u) {
    int d = depth(x, u);
    if (d>dmax) dmax = d;
  });
  return dmax;
}

template <class G, class H>
float avgDepth(const G& x, const H& xt) {
  int ds = 0, D = 0;
  deadEndsForEach(xt, [&](int u) {
    int d = depth(x, u); ++D;
    ds += d;
  });
  return D>0? ds/float(D) : 0;
}


template <class G, class H>
auto minMaxAvgDepth(const G& x, const H& xt) {
  int dmin = x.order();
  int dmax = 0, ds = 0, D = 0;
  deadEndsForEach(xt, [&](int u) {
    int d = depth(x, u); ++D;
    if (d<dmin) dmin = d;
    if (d>dmax) dmax = d;
    ds += d;
  });
  if (D==0) dmin = 0;
  float davg = D>0? ds/float(D) : 0;
  return make_tuple(dmin, dmax, davg);
}
