#PBS -N distoptmize2019
#PBS -l select=4:ncpus=40:mpiprocs=20
#PBS -l walltime=335:30:00
#PBS -j oe
#PBS -o /home/gubidubi/cadical/results/distoptmize2019_res

module load python/3.12.8
module load gcc/6.4.0
module load openmpi/4.0.3
source cadical/meu_env/bin/activate

# Notice we replaced 'python3 -m mpi4py.futures' with './run_mpi_task.sh'
mpiexec -n 80 \
        --hostfile $PBS_NODEFILE \
        --map-by ppr:20:node \
        --bind-to core \
        --report-bindings \
        ./run_mpi_task.sh cadical/CMAES/distributedoptmizationv2.py
