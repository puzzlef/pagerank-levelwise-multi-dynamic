# set termoption linewidth 3
# set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 5 ps 5.5   # blue
# set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 7 ps 1.5   # red
# do for [i=1:100] {
#   set style line i linewidth 30
# }
# set key autotitle columnhead
# plot [500:10000] 'data/arabic-2005.csv'
set term pdf
set terminal pdf size 5in,3in
set output 'output.pdf'

set termoption dashed
set datafile separator ','
set style fill solid border lt -1
set style textbox opaque noborder
set boxwidth 0.20 abs
set xtics rotate by 45 right
set grid y
set key above font ",8"
set logscale y 10
set ylabel 'Runtime (ms)'

# set title 'AM time for all batch sizes (omp)' offset 0,-0.8
plot 'data.csv' \
      using ($1-0.20):7  title 'Dynamic Monolithic' with boxes fill pattern 1, \
  ''  using 1:13:xtic(2) title 'Dynamic Levelwise'  with boxes fill pattern 2, \
  ''  using ($1+0.20):19 title 'Dynamic HyPR (CPU)' with boxes fill pattern 3, \
  ''  using ($1+0.40):11 title 'Static Levelwise'   with boxes fill pattern 3, \
  ''  using ($1+0.20):(0.9*$19):21 title '' with labels font ",7" rotate right, \
  ''  using ($1+0.40):(0.9*$11):20 title '' with labels font ",7" rotate right
