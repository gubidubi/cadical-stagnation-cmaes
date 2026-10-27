from mpi4py.futures import MPIPoolExecutor
import time
import numpy as np
import cma
import subprocess
import random
import os

DEBUGGING = False                 # True to print a bunch of stuff

CUTOFF = 30                       # CPU time for IND result
INSTANCES_PER_IT = 16
REPS_PER_INSTANCE = 5
POPULATION = 50                   # recomended is 4 + floor(3ln(N))
MAX_ITERATIONS = 400
INSTANCE_LIST = "easy_list2.txt"

# CUTOFF = 2
# INSTANCES_PER_IT = 2
# REPS_PER_INSTANCE = 2
# POPULATION = 10
# MAX_ITERATIONS = 10
# INSTANCE_LIST = "easy_list.txt"

def load_instances(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the instance file at: {file_path}")

    with open(file_path, "r") as f:
        instances = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not instances:
        raise ValueError(f"The instance file {file_path} is empty!")

    print(f"Successfully loaded the list of {len(instances)} instances for tuning.")
    return instances

def run_instance(task_id, pc, pl, filename):
    if (pc >= pl):
        return 3*CUTOFF
    binary_path = "cadical/cadical-stagnation/build/cadical"
    # binary_path = "/home/gubidubi/Desktop/Outros/TG1/CaDiCal_Ballock/cadical-stagnation/cadical-stagnation/build/cadical"
    
    arguments = ["-n", "-q", f"--restartbasefast={pc}", f"--restartbaseslow={pl}", "-t", f"{CUTOFF}", f"--seed={task_id}", f"cadical/{filename}"]
    # arguments = ["-n", "-q", f"--restartbasefast={pc}", f"--restartbaseslow={pl}", "-t", f"{CUTOFF}", f"--seed={task_id}", f"/home/gubidubi/Desktop/Outros/TG1/Benchmarks/{filename}"]
    try:
        result = subprocess.run(
            [binary_path] + arguments,
            capture_output=True,
            text=True,
            timeout=3*CUTOFF,
        )
        output = result.stdout.split('\t')
    except Exception as e:
        if DEBUGGING: print(f"Error during {task_id}: {e}")
        return 3*CUTOFF
    if (output == ['']):
        if DEBUGGING: print(f"failed {task_id}: ({pc}, {pl}) - {filename} -- {result}")
        return 3*CUTOFF
    try:
        cputime = float(output[-1].strip())
        is_sat = output[-2]
        if (is_sat == "IND"):
            return 2*CUTOFF
        return cputime
    except Exception as e:
        if DEBUGGING: print(f"Error reading CaDiCal output in task {task_id}: {e}")
        return 3*CUTOFF

def task_run_instance(task_local):
    start = time.time()

    task_id, x, instance = task_local
    pc, pl = x
    if DEBUGGING: print(f"task {task_id}: ({pc}, {pl}) - {instance}")
    res = float(run_instance(task_id, pc, pl, instance))
    if DEBUGGING: print(f"result {task_id}: {res}")

    end = time.time()
    if DEBUGGING: print(f"Task {task_id} on instance finished in {end - start:.2f} seconds", flush=True)
    return res


def main():
    working_dir = os.getcwd()
    if DEBUGGING: print(working_dir)

    instances = load_instances(f"cadical/{INSTANCE_LIST}")
    # instances = load_instances(f"{INSTANCE_LIST}")

    initial_guess = [3.0, 640.0]
    initial_sigma = 1.0
    my_stds = [3.0, 640.0]
    
    integer_indices = list(range(2))

    opts = {
        'popsize': POPULATION,
        'bounds': [1, np.inf],
        'maxiter': MAX_ITERATIONS,
        'CMA_stds': my_stds,
        'integer_variables': integer_indices,
    }
    es = cma.CMAEvolutionStrategy(initial_guess, initial_sigma, opts)

    with MPIPoolExecutor() as executor:
        while not es.stop():
            X = es.ask()
            X_int = np.round(X).astype(int)
            current_generation_batch = random.sample(instances, INSTANCES_PER_IT)

            all_tasks = []
            task_id = 0
            for config in X_int:
                for instance in current_generation_batch:
                    for rep in range(REPS_PER_INSTANCE):
                        all_tasks.append((task_id, config, instance))
                        task_id += 1

            results = list(executor.map(task_run_instance, all_tasks))
            
            if DEBUGGING: print(results)

            fitness_values = [float(np.mean(results[i : i + INSTANCES_PER_IT*REPS_PER_INSTANCE])) for i in range(0, len(results), INSTANCES_PER_IT*REPS_PER_INSTANCE)]
            
            if DEBUGGING: print(X)
            if DEBUGGING: print(fitness_values)

            es.tell(X, fitness_values)
            es.disp()
            es.logger.add()
        print("Optimization Completed!", es.result.xfavorite)

if __name__ == '__main__':
    main()
