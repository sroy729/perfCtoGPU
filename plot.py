import subprocess
import matplotlib.pyplot as plt
import os
import sys
import csv
import pandas as pd
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np

# --- Settings ---
time_out = 100 #seconds
cpu_dir = './GPU' #replace GPU for CPU
out_dir = f'{cpu_dir}/out'
sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

# --- Parse executable name ---
if len(sys.argv) < 2:
    print("Usage: python run_and_plot.py <executable_name>")
    sys.exit(1)

exe_name = sys.argv[1]
exe_path = f"{cpu_dir}/{exe_name}"
csv_path = f"{out_dir}/{exe_name}.csv"

# Ensure out directory exists
os.makedirs(out_dir, exist_ok=True)

# --- Run for each size ---
results = []

clean_dir = ["make", "-C", cpu_dir, "clean"]
result = subprocess.run(clean_dir, capture_output=True, text=True)
for size in sizes: 
    print(f"\n=== Building and Running for SIZE={size} ===")

    # Build the code
    build_cmd = ["make", "-C", cpu_dir, f"SIZE={size}"]
    result = subprocess.run(build_cmd, capture_output=True, text=True)

    if result.returncode != 0: 
        print(f"❌ Build failed for SIZE={size}")
        print(result.stderr)
        results.append((size, "Build Failed"))
        continue

    # Run the executable
    try:
        run_result = subprocess.run([exe_path], capture_output=True, text=True, timeout=time_out)
    except subprocess.TimeoutExpired:
        print(f"⏰ Execution timed out for SIZE={size}")
        results.append((size, "Timeout"))
        continue

    output = run_result.stdout.strip()
    print(output)

    # Clean after run
    subprocess.run(["make", "-C", cpu_dir, "clean"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Parse execution time
    exec_time_ms = None
    for line in output.splitlines():
        if "Execution Time:" in line:
            try:
                exec_time_ms = float(line.split(":")[1].strip().split()[0])
                break
            except:
                pass

    if exec_time_ms is not None:
        results.append((size, exec_time_ms))
    else:
        results.append((size, "Execution Time not found"))

# --- Write results to CSV ---
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["SIZE", "ExecutionTime_ms"])
    for size, time in results:
        writer.writerow([size, time])

print(f"\n✅ Results written to: {csv_path}")

# --- Plotting from all CSVs ---
def plot_all_csvs(out_dir, timeout):
    plt.figure(figsize=(9, 6))
    for file in os.listdir(out_dir):
        if file.endswith(".csv"):
            filepath = os.path.join(out_dir, file)
            x_vals, y_vals = [], []
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        size = int(row["SIZE"])
                        time_str = row["ExecutionTime_ms"]
                        if time_str.strip().lower() == "timeout":
                            time = timeout*1000
                        else:
                            time = float(time_str)
                        x_vals.append(size)
                        y_vals.append(time)
                    except:
                        continue
            label = os.path.splitext(file)[0]
            plt.plot(x_vals, y_vals, marker='o', label=label)

    #add a line at a the timeout value
    plt.axhline(y=timeout*1000, color='red', linestyle='--', linewidth=1.5, label=f"{timeout} sec timeout")

    plt.title("Execution Time vs Matrix SIZE")
    plt.xlabel("Matrix SIZE (N x N)")
    plt.ylabel("Execution Time (ms)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/execution_time_plot.png")
    plt.show()
    print("\n📈 Plot saved as: figures/execution_time_plot.png")

# --- Plotting from all CSVs ---
def plot_all_csvs2(out_dir, timeout=0.41):
    def safe_log2(x):
        x = np.asarray(x)
        return np.where(x > 0, np.log2(x), 0)

    def get_label_from_filename(fname):
        name = os.path.splitext(os.path.basename(fname))[0]
        return name.replace("_", " ")

    # Load all .csv files in the directory
    files = [f for f in os.listdir(out_dir) if f.endswith('.csv')]
    if len(files) == 0:
        print("No CSV files found in:", out_dir)
        return

    plt.figure(figsize=(10, 6))

    for file in sorted(files):
        path = os.path.join(out_dir, file)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        if df.empty or 'x' not in df.columns or 'y' not in df.columns:
            print(f"Skipping {file} due to missing or empty columns.")
            continue

        x = df['x'].values
        y = df['y'].values

        # Convert "Timeout" or non-numeric entries to timeout value * 1000
        y = np.array([
            float(val) * 1000 if isinstance(val, str) and "Timeout" in val else float(val)
            for val in y
        ])

        label = get_label_from_filename(file)
        plt.plot(x, y, label=label, marker='o')

    plt.xlabel("Matrix Size (log₂)")
    plt.ylabel("Execution Time (ms)")
    plt.title("Matrix Multiplication Performance")
    plt.yscale("log")  # Set y-axis to log scale if needed
    plt.xticks(x, labels=[f"{int(v)}" for v in x])  # Optional: clean x labels
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot.png"))
    plt.show()

# Call plotting
plot_all_csvs(out_dir, timeout=time_out)

