Performance of **static** vs **dynamic** [levelwise CUDA] based PageRank ([pull], [CSR],
[skip-teleport], [compute-5M], [skip-comp], [scaled-fill]).

This experiment was for comparing performance between:
1. Find **static** pagerank of updated graph using [nvGraph][pr-nvgraph].
2. Find **dynamic** pagerank of updated graph using [nvGraph][pr-nvgraph].
3. Find **static** [monolithic CUDA] based pagerank of updated graph.
4. Find **dynamic** [monolithic CUDA] based pagerank of updated graph.
5. Find **static** [levelwise CUDA] based pagerank of updated graph.
6. Find **dynamic** [levelwise CUDA] based pagerank of updated graph.

Each approach was attempted on a number of graphs, running each with multiple
batch sizes (`1`, `5`, `10`, `50`, ...). Each batch size was run with 5
different updates to graph, and each specific update was run 5 times for each
approach to get a good time measure. **Levelwise** pagerank is the
[STIC-D algorithm], without **ICD** optimizations. Indeed, **dynamic levelwise**
pagerank is **faster** than the *static* approach for many batch sizes. In
order to measure error, [nvGraph] pagerank is taken as a reference.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets]. The input
data used for this experiment is available at ["graphs"] (for small ones), and
the [SuiteSparse Matrix Collection].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -lnvgraph -O3 main.cu
$ ./a.out ~/data/min-1DeadEnd.mtx
$ ./a.out ~/data/min-2SCC.mtx
$ ...

# (SHORTENED)
# ...
# Loading graph soc-LiveJournal1.mtx ...
# order: 4847571 size: 69532893 {}
# [00166.145 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
#
# # Batch size 1e+0
# order: 4847571.4 size: 69532893.2 {} [00166.207 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4847571.4 size: 69532893.2 {} [00023.409 ms; 000 iters.] [4.8099e-7 err.] pagerankNvgraph [dynamic]
# order: 4847571.4 size: 69532893.2 {} [00175.986 ms; 058 iters.] [2.6337e-6 err.] pagerankMonolithic [static]
# order: 4847571.4 size: 69532893.2 {} [00003.770 ms; 001 iters.] [1.9923e-6 err.] pagerankMonolithic [dynamic]
# order: 4847571.4 size: 69532893.2 {} [00169.749 ms; 058 iters.] [2.6337e-6 err.] pagerankLevelwise [static]
# order: 4847571.4 size: 69532893.2 {} [00004.392 ms; 000 iters.] [1.8941e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 5e+0
# order: 4847571.4 size: 69532897.4 {} [00166.272 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4847571.4 size: 69532897.4 {} [00041.124 ms; 000 iters.] [9.1002e-7 err.] pagerankNvgraph [dynamic]
# order: 4847571.4 size: 69532897.4 {} [00176.012 ms; 058 iters.] [2.6201e-6 err.] pagerankMonolithic [static]
# order: 4847571.4 size: 69532897.4 {} [00006.193 ms; 002 iters.] [1.8311e-6 err.] pagerankMonolithic [dynamic]
# order: 4847571.4 size: 69532897.4 {} [00169.698 ms; 058 iters.] [2.6201e-6 err.] pagerankLevelwise [static]
# order: 4847571.4 size: 69532897.4 {} [00006.166 ms; 001 iters.] [1.8307e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 1e+1
# order: 4847572 size: 69532902.4 {} [00166.361 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4847572 size: 69532902.4 {} [00034.358 ms; 000 iters.] [9.1024e-7 err.] pagerankNvgraph [dynamic]
# order: 4847572 size: 69532902.4 {} [00175.984 ms; 058 iters.] [2.6557e-6 err.] pagerankMonolithic [static]
# order: 4847572 size: 69532902.4 {} [00010.442 ms; 003 iters.] [1.3797e-6 err.] pagerankMonolithic [dynamic]
# order: 4847572 size: 69532902.4 {} [00169.930 ms; 058 iters.] [2.6557e-6 err.] pagerankLevelwise [static]
# order: 4847572 size: 69532902.4 {} [00011.490 ms; 003 iters.] [1.1553e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 5e+1
# order: 4847579 size: 69532944.6 {} [00166.308 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4847579 size: 69532944.6 {} [00056.475 ms; 000 iters.] [1.2656e-6 err.] pagerankNvgraph [dynamic]
# order: 4847579 size: 69532944.6 {} [00175.903 ms; 058 iters.] [2.6704e-6 err.] pagerankMonolithic [static]
# order: 4847579 size: 69532944.6 {} [00018.498 ms; 006 iters.] [1.7267e-6 err.] pagerankMonolithic [dynamic]
# order: 4847579 size: 69532944.6 {} [00170.171 ms; 058 iters.] [2.6704e-6 err.] pagerankLevelwise [static]
# order: 4847579 size: 69532944.6 {} [00019.031 ms; 005 iters.] [1.5730e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 1e+2
# order: 4847590.6 size: 69533000.2 {} [00166.239 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4847590.6 size: 69533000.2 {} [00061.095 ms; 000 iters.] [1.3170e-6 err.] pagerankNvgraph [dynamic]
# order: 4847590.6 size: 69533000.2 {} [00175.901 ms; 058 iters.] [2.6394e-6 err.] pagerankMonolithic [static]
# order: 4847590.6 size: 69533000.2 {} [00022.109 ms; 007 iters.] [2.3196e-6 err.] pagerankMonolithic [dynamic]
# order: 4847590.6 size: 69533000.2 {} [00170.020 ms; 058 iters.] [2.6394e-6 err.] pagerankLevelwise [static]
# order: 4847590.6 size: 69533000.2 {} [00029.485 ms; 007 iters.] [2.0674e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 5e+2
# order: 4847662.2 size: 69533434.2 {} [00183.298 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4847662.2 size: 69533434.2 {} [00083.204 ms; 000 iters.] [1.4059e-6 err.] pagerankNvgraph [dynamic]
# order: 4847662.2 size: 69533434.2 {} [00193.346 ms; 058 iters.] [2.6631e-6 err.] pagerankMonolithic [static]
# order: 4847662.2 size: 69533434.2 {} [00041.759 ms; 012 iters.] [3.6946e-6 err.] pagerankMonolithic [dynamic]
# order: 4847662.2 size: 69533434.2 {} [00170.103 ms; 058 iters.] [2.6631e-6 err.] pagerankLevelwise [static]
# order: 4847662.2 size: 69533434.2 {} [00037.236 ms; 011 iters.] [3.4490e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 1e+3
# order: 4847749 size: 69533980.6 {} [00166.107 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4847749 size: 69533980.6 {} [00082.776 ms; 000 iters.] [1.4555e-6 err.] pagerankNvgraph [dynamic]
# order: 4847749 size: 69533980.6 {} [00175.992 ms; 058 iters.] [2.6475e-6 err.] pagerankMonolithic [static]
# order: 4847749 size: 69533980.6 {} [00045.088 ms; 015 iters.] [4.3855e-6 err.] pagerankMonolithic [dynamic]
# order: 4847749 size: 69533980.6 {} [00170.114 ms; 058 iters.] [2.6475e-6 err.] pagerankLevelwise [static]
# order: 4847749 size: 69533980.6 {} [00044.585 ms; 013 iters.] [4.1042e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 5e+3
# order: 4848475.8 size: 69538336.4 {} [00166.161 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4848475.8 size: 69538336.4 {} [00099.868 ms; 000 iters.] [1.5585e-6 err.] pagerankNvgraph [dynamic]
# order: 4848475.8 size: 69538336.4 {} [00176.119 ms; 058 iters.] [2.6079e-6 err.] pagerankMonolithic [static]
# order: 4848475.8 size: 69538336.4 {} [00070.810 ms; 023 iters.] [4.9181e-6 err.] pagerankMonolithic [dynamic]
# order: 4848475.8 size: 69538336.4 {} [00170.212 ms; 058 iters.] [2.6079e-6 err.] pagerankLevelwise [static]
# order: 4848475.8 size: 69538336.4 {} [00070.265 ms; 021 iters.] [4.3191e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 1e+4
# order: 4849394.8 size: 69543808.4 {} [00166.294 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4849394.8 size: 69543808.4 {} [00105.567 ms; 000 iters.] [1.6712e-6 err.] pagerankNvgraph [dynamic]
# order: 4849394.8 size: 69543808.4 {} [00176.222 ms; 058 iters.] [2.6247e-6 err.] pagerankMonolithic [static]
# order: 4849394.8 size: 69543808.4 {} [00082.712 ms; 027 iters.] [4.9692e-6 err.] pagerankMonolithic [dynamic]
# order: 4849394.8 size: 69543808.4 {} [00169.999 ms; 058 iters.] [2.6247e-6 err.] pagerankLevelwise [static]
# order: 4849394.8 size: 69543808.4 {} [00081.970 ms; 025 iters.] [4.3628e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 5e+4
# order: 4856532.8 size: 69587368.6 {} [00406.951 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4856532.8 size: 69587368.6 {} [00290.850 ms; 000 iters.] [2.0169e-6 err.] pagerankNvgraph [dynamic]
# order: 4856532.8 size: 69587368.6 {} [00290.077 ms; 058 iters.] [2.6554e-6 err.] pagerankMonolithic [static]
# order: 4856532.8 size: 69587368.6 {} [00178.957 ms; 036 iters.] [5.2206e-6 err.] pagerankMonolithic [dynamic]
# order: 4856532.8 size: 69587368.6 {} [00340.038 ms; 058 iters.] [2.6554e-6 err.] pagerankLevelwise [static]
# order: 4856532.8 size: 69587368.6 {} [00214.932 ms; 033 iters.] [4.4503e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 1e+5
# order: 4865427.8 size: 69641739 {} [00769.731 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4865427.8 size: 69641739 {} [00588.862 ms; 000 iters.] [2.1626e-6 err.] pagerankNvgraph [dynamic]
# order: 4865427.8 size: 69641739 {} [00461.525 ms; 058 iters.] [2.6499e-6 err.] pagerankMonolithic [static]
# order: 4865427.8 size: 69641739 {} [00319.360 ms; 040 iters.] [5.0220e-6 err.] pagerankMonolithic [dynamic]
# order: 4865427.8 size: 69641739 {} [00454.465 ms; 058 iters.] [2.6499e-6 err.] pagerankLevelwise [static]
# order: 4865427.8 size: 69641739 {} [00319.607 ms; 037 iters.] [4.4145e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 5e+5
# order: 4930397.8 size: 70072338 {} [00771.812 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4930397.8 size: 70072338 {} [00652.143 ms; 000 iters.] [2.4444e-6 err.] pagerankNvgraph [dynamic]
# order: 4930397.8 size: 70072338 {} [00460.249 ms; 058 iters.] [2.7571e-6 err.] pagerankMonolithic [static]
# order: 4930397.8 size: 70072338 {} [00356.116 ms; 049 iters.] [4.6516e-6 err.] pagerankMonolithic [dynamic]
# order: 4930397.8 size: 70072338 {} [00437.763 ms; 058 iters.] [2.7571e-6 err.] pagerankLevelwise [static]
# order: 4930397.8 size: 70072338 {} [00386.386 ms; 045 iters.] [4.6498e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 1e+6
# order: 4999189.4 size: 70601370 {} [00786.062 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4999189.4 size: 70601370 {} [00729.155 ms; 000 iters.] [2.0311e-6 err.] pagerankNvgraph [dynamic]
# order: 4999189.4 size: 70601370 {} [00466.436 ms; 058 iters.] [2.8056e-6 err.] pagerankMonolithic [static]
# order: 4999189.4 size: 70601370 {} [00416.211 ms; 052 iters.] [4.5759e-6 err.] pagerankMonolithic [dynamic]
# order: 4999189.4 size: 70601370 {} [00463.187 ms; 058 iters.] [2.8056e-6 err.] pagerankLevelwise [static]
# order: 4999189.4 size: 70601370 {} [00414.512 ms; 049 iters.] [4.5753e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 5e+6
# order: 5258189.6 size: 74648558.8 {} [00851.027 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 5258189.6 size: 74648558.8 {} [00821.523 ms; 000 iters.] [1.0021e-6 err.] pagerankNvgraph [dynamic]
# order: 5258189.6 size: 74648558.8 {} [00497.804 ms; 058 iters.] [3.5681e-6 err.] pagerankMonolithic [static]
# order: 5258189.6 size: 74648558.8 {} [00480.474 ms; 056 iters.] [4.3080e-6 err.] pagerankMonolithic [dynamic]
# order: 5258189.6 size: 74648558.8 {} [00650.291 ms; 056 iters.] [5.0597e-6 err.] pagerankLevelwise [static]
# order: 5258189.6 size: 74648558.8 {} [00612.824 ms; 053 iters.] [4.7137e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 1e+7
# order: 5321147.6 size: 79595675.4 {} [00899.173 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 5321147.6 size: 79595675.4 {} [00836.380 ms; 000 iters.] [9.2568e-7 err.] pagerankNvgraph [dynamic]
# order: 5321147.6 size: 79595675.4 {} [00674.229 ms; 057 iters.] [3.7252e-6 err.] pagerankMonolithic [static]
# order: 5321147.6 size: 79595675.4 {} [00650.231 ms; 055 iters.] [4.3098e-6 err.] pagerankMonolithic [dynamic]
# order: 5321147.6 size: 79595675.4 {} [00807.394 ms; 053 iters.] [4.4987e-6 err.] pagerankLevelwise [static]
# order: 5321147.6 size: 79595675.4 {} [00784.281 ms; 051 iters.] [5.2691e-6 err.] pagerankLevelwise [dynamic]
#
# # Batch size 5e+7
# order: 5332329 size: 119532813.6 {} [01049.286 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 5332329 size: 119532813.6 {} [01034.612 ms; 000 iters.] [8.4161e-8 err.] pagerankNvgraph [dynamic]
# order: 5332329 size: 119532813.6 {} [00699.930 ms; 032 iters.] [2.6754e-6 err.] pagerankMonolithic [static]
# order: 5332329 size: 119532813.6 {} [00681.976 ms; 031 iters.] [2.8770e-6 err.] pagerankMonolithic [dynamic]
# order: 5332329 size: 119532813.6 {} [00864.484 ms; 032 iters.] [2.4783e-6 err.] pagerankLevelwise [static]
# order: 5332329 size: 119532813.6 {} [00868.942 ms; 031 iters.] [3.7332e-6 err.] pagerankLevelwise [dynamic]
#
# ...
```

[![](https://i.imgur.com/jSWK5AH.gif)][sheets]
[![](https://i.imgur.com/CYFzTqr.gif)][sheets]

<br>
<br>


## References

- [STIC-D: algorithmic techniques for efficient parallel pagerank computation on real-world graphs][STIC-D algorithm]
- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/pH5CTr2.jpg)](https://www.youtube.com/watch?v=rskLxOHNF3k)

[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
[STIC-D algorithm]: https://www.slideshare.net/SubhajitSahu/sticd-algorithmic-techniques-for-efficient-parallel-pagerank-computation-on-realworld-graphs
[nvGraph]: https://github.com/rapidsai/nvgraph
["graphs"]: https://github.com/puzzlef/graphs
[pr-nvgraph]: https://github.com/puzzlef/pagerank-nvgraph-static-vs-dynamic
[monolithic CUDA]: https://github.com/puzzlef/pagerank-cuda-monolithic-vs-levelwise
[levelwise CUDA]: https://github.com/puzzlef/pagerank-cuda-monolithic-vs-levelwise
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[CSR]: https://github.com/puzzlef/pagerank-class-vs-csr
[skip-teleport]: https://github.com/puzzlef/pagerank-levelwise-skip-teleport
[compute-5M]: https://github.com/puzzlef/pagerank-levelwise-cuda-adjust-compute-size
[skip-comp]: https://github.com/puzzlef/pagerank-levelwise-dynamic-validate-skip-unchanged-components
[scaled-fill]: https://github.com/puzzlef/pagerank-dynamic-adjust-ranks
[charts]: https://photos.app.goo.gl/mu2v8BrMsxkWKEM17
[sheets]: https://docs.google.com/spreadsheets/d/17-W0dGD6wjBvVaXVILlmUGFpBnPfaMIRPaqRf7u9mUk/edit?usp=sharing
