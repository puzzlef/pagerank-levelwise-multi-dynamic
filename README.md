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
$ ./a.out ~/data/CollegeMsg.txt
$ ./a.out ~/data/email-Eu-core-temporal.txt
$ ...

# ...
#
# Using graph /home/subhajit/data/sx-stackoverflow.txt ...
# Temporal edges: 63497051
#
# # Batch size 1e+01
# order: 747591 size: 6102013 {} [00014.756 ms; 000 iters.] [0.0000e+00 err.] I:pagerankNvgraph (static)
# order: 747591 size: 6102013 {} [00003.421 ms; 000 iters.] [2.4457e-07 err.] I:pagerankNvgraph (incremental)
# order: 747591 size: 6102013 {} [00225.349 ms; 012 iters.] [1.9275e-03 err.] I:pagerankMonolithicSeq (static)
# order: 747591 size: 6102013 {} [00019.601 ms; 001 iters.] [1.4345e-06 err.] I:pagerankMonolithicSeq (incremental)
# order: 747591 size: 6102013 {} [00016.551 ms; 001 iters.] [1.4948e-06 err.] I:pagerankMonolithicSeq (dynamic)
# order: 747591 size: 6102013 {} [00187.553 ms; 012 iters.] [1.9275e-03 err.] I:pagerankMonolithicSeqSplit (static)
# order: 747591 size: 6102013 {} [00016.696 ms; 001 iters.] [1.4345e-06 err.] I:pagerankMonolithicSeqSplit (incremental)
# order: 747591 size: 6102013 {} [00014.893 ms; 001 iters.] [1.4948e-06 err.] I:pagerankMonolithicSeqSplit (dynamic)
# order: 747591 size: 6102013 {} [00077.452 ms; 021 iters.] [3.1954e-05 err.] I:pagerankMonolithicOmp (static)
# order: 747591 size: 6102013 {} [00076.233 ms; 001 iters.] [1.4345e-06 err.] I:pagerankMonolithicOmp (incremental)
# order: 747591 size: 6102013 {} [00076.056 ms; 001 iters.] [1.4948e-06 err.] I:pagerankMonolithicOmp (dynamic)
# order: 747591 size: 6102013 {} [01011.759 ms; 019 iters.] [7.9036e-05 err.] I:pagerankMonolithicOmpSplit (static)
# order: 747591 size: 6102013 {} [00008.771 ms; 001 iters.] [1.4345e-06 err.] I:pagerankMonolithicOmpSplit (incremental)
# order: 747591 size: 6102013 {} [00064.084 ms; 001 iters.] [1.4948e-06 err.] I:pagerankMonolithicOmpSplit (dynamic)
# order: 747591 size: 6102013 {} [00002.537 ms; 012 iters.] [1.9275e-03 err.] I:pagerankMonolithicCuda (static)
# order: 747591 size: 6102013 {} [00000.294 ms; 001 iters.] [1.4467e-06 err.] I:pagerankMonolithicCuda (incremental)
# order: 747591 size: 6102013 {} [00000.287 ms; 001 iters.] [1.5070e-06 err.] I:pagerankMonolithicCuda (dynamic)
# order: 747591 size: 6102013 {} [00002.505 ms; 012 iters.] [1.9275e-03 err.] I:pagerankMonolithicCudaSplit (static)
# order: 747591 size: 6102013 {} [00000.291 ms; 001 iters.] [1.4467e-06 err.] I:pagerankMonolithicCudaSplit (incremental)
# order: 747591 size: 6102013 {} [00000.284 ms; 001 iters.] [1.5070e-06 err.] I:pagerankMonolithicCudaSplit (dynamic)
# order: 747591 size: 6102013 {} [00183.146 ms; 011 iters.] [1.9068e-02 err.] I:pagerankLevelwiseSeq (static)
# order: 747591 size: 6102013 {} [00016.420 ms; 000 iters.] [1.5560e-06 err.] I:pagerankLevelwiseSeq (incremental)
# order: 747591 size: 6102013 {} [00015.229 ms; 000 iters.] [1.6053e-06 err.] I:pagerankLevelwiseSeq (dynamic)
# order: 747591 size: 6102013 {} [05064.182 ms; 019 iters.] [1.1208e-02 err.] I:pagerankLevelwiseOmp (static)
# order: 747591 size: 6102013 {} [00003.913 ms; 000 iters.] [1.5560e-06 err.] I:pagerankLevelwiseOmp (incremental)
# order: 747591 size: 6102013 {} [00331.369 ms; 000 iters.] [1.6053e-06 err.] I:pagerankLevelwiseOmp (dynamic)
# order: 747591 size: 6102013 {} [00032.053 ms; 011 iters.] [1.9068e-02 err.] I:pagerankLevelwiseCuda (static)
# order: 747591 size: 6102013 {} [00002.755 ms; 000 iters.] [1.5658e-06 err.] I:pagerankLevelwiseCuda (incremental)
# order: 747591 size: 6102013 {} [00001.944 ms; 000 iters.] [1.6151e-06 err.] I:pagerankLevelwiseCuda (dynamic)
# order: 747591 size: 6102013 {} [00014.706 ms; 000 iters.] [0.0000e+00 err.] D:pagerankNvgraph (static)
# ...
```

[![](https://imgur.com/MIe18NP.png)][sheetp]
[![](https://imgur.com/lfHLA6K.png)][sheetp]
[![](https://imgur.com/cT68OJ3.png)][sheetp]
[![](https://imgur.com/ShuDCF8.png)][sheetp]
[![](https://imgur.com/eSBDfFQ.png)][sheetp]
[![](https://imgur.com/P1yU11s.png)][sheetp]

[![](https://imgur.com/0PHEeVw.png)][sheetp]
[![](https://imgur.com/yi4vn2s.png)][sheetp]
[![](https://imgur.com/ZIqaxYD.png)][sheetp]
[![](https://imgur.com/4VqFxXw.png)][sheetp]
[![](https://imgur.com/8a9rLga.png)][sheetp]
[![](https://imgur.com/zbblUNg.png)][sheetp]

[![](https://imgur.com/B3Dp1uy.gif)][sheetp]
[![](https://imgur.com/BSkaJHK.gif)][sheetp]
[![](https://imgur.com/W0ifRM7.gif)][sheetp]
[![](https://imgur.com/L7bF83U.gif)][sheetp]

[![](https://imgur.com/IPfcLIM.gif)][sheetp]
[![](https://imgur.com/R3Rh618.gif)][sheetp]
[![](https://imgur.com/mo1LX5r.gif)][sheetp]
[![](https://imgur.com/70wAFEE.gif)][sheetp]

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
[charts]: https://photos.app.goo.gl/3P8QAqK8oXvbeAVNA
[sheets]: https://docs.google.com/spreadsheets/d/1Ci4zSJqs2dK_TFtkKfP90sp0srRKeo0f8t2g_bSywFM/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vSUn7LILIgMLqtUvRvNbiaB022SN9z9GRVzuJqVA2mRFDXMo1jOSLfdALveLsW1gaXO6FFMa_wVY0S3/pubhtml
