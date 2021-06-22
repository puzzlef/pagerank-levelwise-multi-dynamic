#include <cmath>
#include <string>
#include <sstream>
#include <cstdio>
#include <iostream>
#include <utility>
#include "src/main.hxx"

using namespace std;




void runExpt(const string& data, bool show) {
  auto ix = range(10);
  auto iy = transform(ix, [](int n) { return n*2; });
  auto iz = show? ix : iy;
  for (int v : iy)
    printf("%d\n", v);
}


int main(int argc, char **argv) {
  char *file = argv[1];
  bool  show = argc > 2;
  printf("Using graph %s ...\n", file);
  string d = readFile(file);
  runExpt(d, show);
  printf("\n");
  return 0;
}
