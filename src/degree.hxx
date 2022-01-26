#pragma once
#include <tuple>

using std::make_tuple;



// MIN/MAX/AVG
// -----------

template <class G>
int minDegree(const G& x) {
  int dmin = x.order();
  for (int u : x.vertices()) {
    int d = x.degree(u);
    if (d<dmin) dmin = d;
  }
  return dmin;
}

template <class G>
int maxDegree(const G& x) {
  int dmax = 0;
  for (int u : x.vertices()) {
    int d = x.degree(u);
    if (d>dmax) dmax = d;
  }
  return dmax;
}

template <class G>
float avgDegree(const G& x) {
  int N = x.order();
  return N>0? x.size()/float(N) : 0;
}


template <class G>
auto minMaxAvgDegree(const G& x) {
  int dmin = x.order();
  int dmax = 0;
  for (int u : x.vertices()) {
    int d = x.degree(u);
    if (d<dmin) dmin = d;
    if (d>dmax) dmax = d;
  }
  return make_tuple(dmin, dmax, avgDegree(x));
}
