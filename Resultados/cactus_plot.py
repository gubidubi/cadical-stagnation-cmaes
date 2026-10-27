import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_from_csv(csv_path, timeout=5000):
    """
    Reads solver execution times from a CSV and plots a cumulative solved instances graph.
    
    :param csv_path: Path to the CSV file.
    :param timeout: Maximum runtime threshold (seconds). Runtimes >= timeout are treated as unsolved.
    """
    # 1. Load the CSV data using pandas
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file at '{csv_path}' was not found.")
        return

    # 2. Identify solver columns 
    # Assumes the first column is the instance name/ID; all other columns are solvers.
    # Adjust this if your CSV has a different structure!
    solver_columns = df.columns[1:] 
    
    # Initialize the plot
    plt.figure(figsize=(8, 6))
    
    # 3. Process each solver column
    for solver_name in solver_columns:
        # Extract runtimes and convert to a numpy array, dropping any empty missing values
        runtimes = df[solver_name].dropna().to_numpy()
        
        # Filter out timeouts (only keep instances solved strictly under the timeout limit)
        solved_runtimes = runtimes[runtimes < timeout]
        
        # Sort runtimes in ascending order
        sorted_runtimes = np.sort(solved_runtimes)
        
        # Create cumulative counts (1 to N)
        solved_counts = np.arange(1, len(sorted_runtimes) + 1)
        
        # Force the curve to start precisely at the origin (0,0)
        X = np.insert(sorted_runtimes, 0, 0.0)
        Y = np.insert(solved_counts, 0, 0)
        
        # Use LaTeX formatting for MAB labels if they match your original legend
        display_label = solver_name
        if "MAB_eps_0" in solver_name:
            display_label = r"MAB $\epsilon = 0$"
        elif "MAB_eps_50" in solver_name:
            display_label = r"MAB $\epsilon = 50$"
        
        # Plot continuous line matching the reference image style
        plt.plot(X, Y, linewidth=1.8, label=display_label)

    # 4. Styling adjustments
    plt.xlabel("Tempo (s)", fontsize=11)
    plt.ylabel("Instâncias resolvidas", fontsize=11)
    plt.xlim(-100, timeout + 100) # Give a small margin past 0 and the timeout
    
    # Clean, subtle grid lines
    plt.grid(True, which="both", linestyle="-", color="#EAEAEA", linewidth=0.8)
    
    # Legend appearance matching academic style
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#D3D3D3", fontsize=10)
    plt.tight_layout()
    
    # Save a high-quality copy for your paper and show it
    plt.savefig("comp2026.png", dpi=300)
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # Replace 'solver_results.csv' with the actual path to your data file
    plot_from_csv("TG resultados - Comp2026.csv", timeout=5000)

