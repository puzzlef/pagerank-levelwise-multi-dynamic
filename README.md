Performance benefit of **skipping converged vertices** for **CUDA** based
PageRank ([pull], [CSR]).

`TODO!`

This experiment was for comparing performance between:
1. Find PageRank **without optimization**.
2. Find PageRank *skipping converged vertices* **with re-check** (in `2`-`16` turns).
3. Find PageRank *skipping converged vertices* **after several turns** (in `2`-`64` turns).

Each approach was attempted on a number of graphs, running each approach 5 times
to get a good time measure. **Skip with re-check** (`skip-check`) is done every
`2`-`16` turns. **Skip after turns** (`skip-after`) is done after `2`-`64`
turns.

Results indicate that the optimizations provide an improvement on only a
few graphs (without introducing too much error):
- For `web-Stanford`, a `skip-check` of `11`-`14` appears to work best.
- For `web-BerkStan`, a `skip-check` of `8`-`14` appears to work best.
- For other graphs, there is no improvement.

On average however, *neither* `skip-check`, nor `skip-after` gives **better
speed** than the **default (unoptimized) approach** (considering the error
introduced due to skipping). This could be due to the unnecessary iterations
added by `skip-check` (mistakenly skipped), and increased memory accesses
performed by `skip-after` (tracking converged count).

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are to be included below, generated from [sheets]. The input
data used for this experiment is available at ["graphs"] (for small ones), and
the [SuiteSparse Matrix Collection]. This experiment was done with guidance
from [Prof. Dip Sankar Banerjee] and [Prof. Kishore Kothapalli].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -lnvgraph -O3 main.cu
$ ./a.out ~/data/min-1DeadEnd.mtx
$ ./a.out ~/data/min-2SCC.mtx
$ ...

# ...
#
# Using graph /home/subhajit/data/sx-superuser.txt ...
# Temporal edges: 1443340
#
# # Batch size 1e+01
# order: 37574 size: 148561 {} [00003.049 ms; 000 iters.] [0.0000e+00 err.] I:pagerankNvgraph (static)
# order: 37574 size: 148561 {} [00002.077 ms; 000 iters.] [5.2493e-07 err.] I:pagerankNvgraph (incremental)
# order: 37574 size: 148561 {} [00012.016 ms; 020 iters.] [1.9376e-05 err.] I:pagerankMonolithicSeq (static)
# order: 37574 size: 148561 {} [00001.250 ms; 002 iters.] [2.1160e-05 err.] I:pagerankMonolithicSeq (incremental)
# order: 37574 size: 148561 {} [00000.720 ms; 002 iters.] [2.4707e-05 err.] I:pagerankMonolithicSeq (dynamic)
# order: 37574 size: 148561 {} [00009.600 ms; 020 iters.] [1.9376e-05 err.] I:pagerankMonolithicSeqSplit (static)
# order: 37574 size: 148561 {} [00001.016 ms; 002 iters.] [2.1160e-05 err.] I:pagerankMonolithicSeqSplit (incremental)
# order: 37574 size: 148561 {} [00000.728 ms; 002 iters.] [2.4707e-05 err.] I:pagerankMonolithicSeqSplit (dynamic)
# order: 37574 size: 148561 {} [00005.641 ms; 020 iters.] [1.9376e-05 err.] I:pagerankMonolithicOmp (static)
# order: 37574 size: 148561 {} [00000.550 ms; 002 iters.] [2.1160e-05 err.] I:pagerankMonolithicOmp (incremental)
# order: 37574 size: 148561 {} [00000.370 ms; 002 iters.] [2.4707e-05 err.] I:pagerankMonolithicOmp (dynamic)
# order: 37574 size: 148561 {} [00005.139 ms; 020 iters.] [1.9376e-05 err.] I:pagerankMonolithicOmpSplit (static)
# order: 37574 size: 148561 {} [00000.566 ms; 002 iters.] [2.1160e-05 err.] I:pagerankMonolithicOmpSplit (incremental)
# order: 37574 size: 148561 {} [00000.342 ms; 002 iters.] [2.4707e-05 err.] I:pagerankMonolithicOmpSplit (dynamic)
# order: 37574 size: 148561 {} [00000.590 ms; 020 iters.] [1.9378e-05 err.] I:pagerankMonolithicCuda (static)
# order: 37574 size: 148561 {} [00000.077 ms; 002 iters.] [2.1160e-05 err.] I:pagerankMonolithicCuda (incremental)
# order: 37574 size: 148561 {} [00000.086 ms; 002 iters.] [2.4705e-05 err.] I:pagerankMonolithicCuda (dynamic)
# order: 37574 size: 148561 {} [00000.701 ms; 020 iters.] [1.9378e-05 err.] I:pagerankMonolithicCudaSplit (static)
# order: 37574 size: 148561 {} [00000.085 ms; 002 iters.] [2.1160e-05 err.] I:pagerankMonolithicCudaSplit (incremental)
# order: 37574 size: 148561 {} [00000.087 ms; 002 iters.] [2.4705e-05 err.] I:pagerankMonolithicCudaSplit (dynamic)
# order: 37574 size: 148561 {} [00009.635 ms; 020 iters.] [1.9376e-05 err.] I:pagerankLevelwiseSeq (static)
# order: 37574 size: 148561 {} [00001.034 ms; 002 iters.] [2.1160e-05 err.] I:pagerankLevelwiseSeq (incremental)
# order: 37574 size: 148561 {} [00000.723 ms; 001 iters.] [2.4707e-05 err.] I:pagerankLevelwiseSeq (dynamic)
# order: 37574 size: 148561 {} [00005.948 ms; 020 iters.] [1.9376e-05 err.] I:pagerankLevelwiseOmp (static)
# order: 37574 size: 148561 {} [00000.819 ms; 002 iters.] [2.1160e-05 err.] I:pagerankLevelwiseOmp (incremental)
# order: 37574 size: 148561 {} [00000.346 ms; 001 iters.] [2.4707e-05 err.] I:pagerankLevelwiseOmp (dynamic)
# order: 37574 size: 148561 {} [00003.213 ms; 020 iters.] [1.9379e-05 err.] I:pagerankLevelwiseCuda (static)
# order: 37574 size: 148561 {} [00000.429 ms; 002 iters.] [2.1164e-05 err.] I:pagerankLevelwiseCuda (incremental)
# order: 37574 size: 148561 {} [00000.206 ms; 001 iters.] [2.4715e-05 err.] I:pagerankLevelwiseCuda (dynamic)
# order: 37574 size: 148561 {} [00003.092 ms; 000 iters.] [0.0000e+00 err.] D:pagerankNvgraph (static)
# ...
```

[![](https://i.imgur.com/ZqjTC2X.png)][sheetp]
[![](https://i.imgur.com/DfLlLf1.png)][sheetp]
[![](https://i.imgur.com/t9XY6Rf.png)][sheetp]
[![](https://i.imgur.com/qCKfpWZ.png)][sheetp]
[![](https://i.imgur.com/GjDK1Y7.png)][sheetp]
[![](https://i.imgur.com/SaCKdfR.png)][sheetp]

[![](https://i.imgur.com/lrAcQsh.png)][sheetp]
[![](https://i.imgur.com/0qIMCWA.png)][sheetp]
[![](https://i.imgur.com/0rU6vgM.png)][sheetp]
[![](https://i.imgur.com/L4T5lsT.png)][sheetp]
[![](https://i.imgur.com/dqnGE8q.png)][sheetp]
[![](https://i.imgur.com/FXraCAt.png)][sheetp]

<br>
<br>


## References

- [STIC-D: Algorithmic Techniques For Efficient Parallel Pagerank Computation on Real-World Graphs](https://gist.github.com/wolfram77/bb09968cc0e592583c4b180243697d5a)
- [Adjusting PageRank parameters and Comparing results](https://arxiv.org/abs/2108.02997)
- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](https://www.youtube.com/watch?v=ke9g8hB0MEo)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/KExwVG1.jpg)](https://www.youtube.com/watch?v=A7TKQKAFIi4)
[![DOI](https://zenodo.org/badge/381913855.svg)](https://zenodo.org/badge/latestdoi/381913855)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://www.iiit.ac.in/people/faculty/kkishore/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
["graphs"]: https://github.com/puzzlef/graphs
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[CSR]: https://github.com/puzzlef/pagerank-class-vs-csr
[charts]: https://photos.app.goo.gl/Fq6dQn2DVR61JLpN7
[sheets]: https://docs.google.com/spreadsheets/d/1Ci4zSJqs2dK_TFtkKfP90sp0srRKeo0f8t2g_bSywFM/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vSUn7LILIgMLqtUvRvNbiaB022SN9z9GRVzuJqVA2mRFDXMo1jOSLfdALveLsW1gaXO6FFMa_wVY0S3/pubhtml
