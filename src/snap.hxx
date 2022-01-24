#pragma once
#include <string>
#include <vector>
#include <istream>
#include <sstream>

using std::string;
using std::vector;
using std::istream;
using std::stringstream;
using std::getline;




// READ-SNAP-TEMPORAL
// ------------------

template <class G>
bool readSnapTemporalLine(G& a, const string& ln, bool sym=false) {
  int u, v, t;
  stringstream ls(ln);
  if (!(ls >> u >> v >> t)) return false;
  a.addEdge(u, v);
  if (sym) a.addEdge(v, u);
  return true;
}


template <class G>
bool readSnapTemporal(G& a, istream& s, size_t N, bool sym=false) {
  size_t i = 0;
  for (; i<N; i++) {
    string ln; getline(s, ln);
    if (!readSnapTemporalLine(a, ln, sym)) break;
  }
  return i>0;
}
