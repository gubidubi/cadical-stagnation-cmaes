#PBS -N benchmark2019_single
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -o /home/gubidubi/cadical/results/benchmark2019_single_res

FILENAME=$(sed -n "${FILE_INDEX}p" cadical/file_list.txt)
echo -n "$FILENAME \t"
./cadical/cadical-stagnation/build/cadical -n -q --restartbasefast=4 --restartbaseslow=500 -t 50 --seed=2026 "cadical/$FILENAME"

