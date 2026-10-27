#PBS -N optmize2019
#PBS -l select=1:ncpus=56
#PBS -l walltime=335:30:00
#PBS -j oe
#PBS -o /home/gubidubi/cadical/results/optmize2019_res

module load python/3.12.8
source cadical/meu_env/bin/activate

python3 cadical/CMAES/fulloptmization.py

