Performance of [levelwise] based static vs dynamic PageRank ([pull], [CSR],
[comp-50], [scaled-fill]).

This experiment was for comparing performance between:
1. Find static pagerank using [standard algorithm] *(monolithic)*.
2. Find static pagerank using [levelwise algorithm].
3. Find dynamic pagerank using [levelwise algorithm].

Each approach was attempted on a number of graphs, running each with multiple
batch sizes (`1`, `5`, `10`, `50`, ...). Each pagerank computation was run 5
times for both approaches to get a good time measure. **Levelwise** pagerank
is the [STIC-D algorithm], without **ICD** optimizations (using single-thread).
Clearly, **dynamic** *levelwise* pagerank faster is than the **static**
approach for many batch sizes.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets]. The input
data used for this experiment is available at the
[Stanford Large Network Dataset Collection].

<br>

```bash
$ g++ -O3 main.cxx
$ ./a.out ~/data/email-Eu-core-temporal.txt
$ ./a.out ~/data/CollegeMsg.txt
$ ...

# (SHORTENED)
# Using graph sx-stackoverflow ...
# Temporal edges: 63497051
# order: 2601975 size: 36951021 {}
#
# # Batch size 1e+0
# [03778.890 ms; 044 iters.] [0.0000e+0 err.] pagerankMonolithic [static]
# [02090.958 ms; 049 iters.] [3.9878e-6 err.] pagerankLevelwise [static]
# [00051.059 ms; 000 iters.] [3.9724e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 5e+0
# [03790.509 ms; 044 iters.] [0.0000e+0 err.] pagerankMonolithic [static]
# [02074.838 ms; 049 iters.] [3.9779e-6 err.] pagerankLevelwise [static]
# [00101.090 ms; 001 iters.] [4.0244e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 1e+1
# [03828.967 ms; 044 iters.] [0.0000e+0 err.] pagerankMonolithic [static]
# [02087.992 ms; 049 iters.] [3.9770e-6 err.] pagerankLevelwise [static]
# [00133.614 ms; 001 iters.] [4.1197e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 5e+1
# [03784.429 ms; 044 iters.] [0.0000e+0 err.] pagerankMonolithic [static]
# [02089.587 ms; 049 iters.] [3.9787e-6 err.] pagerankLevelwise [static]
# [00220.933 ms; 002 iters.] [4.4152e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 1e+2
# [03844.255 ms; 044 iters.] [0.0000e+0 err.] pagerankMonolithic [static]
# [02107.710 ms; 049 iters.] [3.9819e-6 err.] pagerankLevelwise [static]
# [00262.562 ms; 003 iters.] [4.5857e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 5e+2
# [03774.800 ms; 044 iters.] [0.0000e+0 err.] pagerankMonolithic [static]
# [02070.990 ms; 049 iters.] [3.9792e-6 err.] pagerankLevelwise [static]
# [00379.282 ms; 005 iters.] [4.9178e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 1e+3
# [04266.784 ms; 044 iters.] [0.0000e+0 err.] pagerankMonolithic [static]
# [02268.871 ms; 049 iters.] [3.9950e-6 err.] pagerankLevelwise [static]
# [00486.638 ms; 006 iters.] [5.0327e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 5e+3
# [04689.506 ms; 044 iters.] [0.0000e+0 err.] pagerankMonolithic [static]
# [02436.644 ms; 049 iters.] [3.9832e-6 err.] pagerankLevelwise [static]
# [00806.018 ms; 011 iters.] [4.9621e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 1e+4
# [04274.704 ms; 044 iters.] [0.0000e+0 err.] pagerankMonolithic [static]
# [02297.683 ms; 049 iters.] [3.9969e-6 err.] pagerankLevelwise [static]
# [00916.021 ms; 014 iters.] [4.9902e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 5e+4
# [04687.937 ms; 044 iters.] [0.0000e+0 err.] pagerankMonolithic [static]
# [02440.569 ms; 049 iters.] [3.9902e-6 err.] pagerankLevelwise [static]
# [01407.113 ms; 021 iters.] [4.9240e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 1e+5
# [04539.768 ms; 044 iters.] [0.0000e+0 err.] pagerankMonolithic [static]
# [02403.685 ms; 049 iters.] [3.9363e-6 err.] pagerankLevelwise [static]
# [01566.698 ms; 024 iters.] [4.8300e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 5e+5
# [04467.654 ms; 044 iters.] [0.0000e+0 err.] pagerankMonolithic [static]
# [02360.843 ms; 049 iters.] [3.9984e-6 err.] pagerankLevelwise [static]
# [01955.227 ms; 031 iters.] [4.6541e-6 err.] pagerankLevelwise [dynamic]
```

[![](https://i.imgur.com/J7EJ7g4.gif)][sheets]
[![](https://i.imgur.com/cMSMbSZ.gif)][sheets]
[![](https://i.imgur.com/RQVbFi9.gif)][sheets]
[![](https://i.imgur.com/UENUt6J.gif)][sheets]
[![](https://i.imgur.com/JPt5IZQ.gif)][sheets]
[![](https://i.imgur.com/ODME87P.gif)][sheets]
[![](https://i.imgur.com/7rYF21F.gif)][sheets]
[![](https://i.imgur.com/N6rAgzg.gif)][sheets]
[![](https://i.imgur.com/aKwa3Iy.gif)][sheets]
[![](https://i.imgur.com/kpCf9il.gif)][sheets]
[![](https://i.imgur.com/3s999ST.gif)][sheets]
[![](https://i.imgur.com/P737MqX.gif)][sheets]
[![](https://i.imgur.com/89GNw7a.gif)][sheets]
[![](https://i.imgur.com/LyYRrLu.gif)][sheets]

<br>
<br>


## References

- [STIC-D: algorithmic techniques for efficient parallel pagerank computation on real-world graphs][STIC-D algorithm]
- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [Stanford Large Network Dataset Collection]

<br>
<br>

[![](https://i.imgur.com/cL6ZNtU.jpg)](https://www.youtube.com/watch?v=xEfsE8H6sok)

[levelwise]: https://github.com/puzzlef/pagerank-monolithic-vs-levelwise
[levelwise algorithm]: https://github.com/puzzlef/pagerank-monolithic-vs-levelwise
[standard algorithm]: https://github.com/puzzlef/pagerank-monolithic-vs-levelwise
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[CSR]: https://github.com/puzzlef/pagerank-class-vs-csr
[comp-50]: https://github.com/puzzlef/pagerank-levelwise-openmp-adjust-component-size
[scaled-fill]: https://github.com/puzzlef/pagerank-dynamic-adjust-ranks
[STIC-D algorithm]: https://www.slideshare.net/SubhajitSahu/sticd-algorithmic-techniques-for-efficient-parallel-pagerank-computation-on-realworld-graphs
[charts]: https://photos.app.goo.gl/1XqKzvtL73xN8Tro6
[sheets]: https://docs.google.com/spreadsheets/d/1azuAqSPU2RP8Z8wVxNSbW5AJuq1jQmZVo9fA90rf-_s/edit?usp=sharing
[Stanford Large Network Dataset Collection]: http://snap.stanford.edu/data/index.html
