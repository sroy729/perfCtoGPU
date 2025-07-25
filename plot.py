import subprocess
import matplotlib.pyplot as plt
import os
import sys
import csv

# --- Settings ---
time_out = 100 #seconds
cpu_dir = './CPU'
out_dir = f'{cpu_dir}/out'
sizes = [64, 128, 256, 512, 1024, 2048, 4096]

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

# Call plotting
plot_all_csvs(out_dir, timeout=time_out)

