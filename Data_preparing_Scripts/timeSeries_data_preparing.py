import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# ✅ STEP 1: Load Original Data
# ---------------------------
input_file = "10066852034034-timeseries.csv"

try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"❌ ERROR: File {input_file} not found!")
    exit()

print(f"✅ Loaded dataset with {len(df)} rows")

# ---------------------------
# ✅ STEP 2: Ensure ElapsedTime exists
# ---------------------------
if 'ElapsedTime' not in df.columns:
    print("⚠️ 'ElapsedTime' column not found → generating synthetic index")
    df['ElapsedTime'] = np.arange(len(df))

# ---------------------------
# ✅ STEP 3: Generate Synthetic CPU Frequency
# ---------------------------
total_rows = len(df)

# Distribution: 30% idle, 40% normal, 30% busy
idle_size = int(total_rows * 0.3)
normal_size = int(total_rows * 0.4)
busy_size = total_rows - idle_size - normal_size

# Generate values
idle_vals = np.random.normal(10, 3, idle_size)    # mean=10, std=3
normal_vals = np.random.normal(50, 10, normal_size) # mean=50, std=10
busy_vals = np.random.normal(90, 5, busy_size)     # mean=90, std=5

# Concatenate all
cpu_values = np.concatenate([idle_vals, normal_vals, busy_vals])
cpu_values = np.clip(cpu_values, 0, 100)  # ensure [0,100]
np.random.shuffle(cpu_values)  # shuffle to mix states

# Assign to dataframe
df['CPUFrequency'] = cpu_values

# ---------------------------
# ✅ STEP 4: Normalize 0-100
# ---------------------------
df['CPU_norm'] = (df['CPUFrequency'] - df['CPUFrequency'].min()) / (df['CPUFrequency'].max() - df['CPUFrequency'].min()) * 100
df['CPU_norm'] = df['CPU_norm'].round(2)

# ---------------------------
# ✅ STEP 5: Discretize into Observed States
# ---------------------------
def discretize(val):
    if val < 30:
        return 0  # idle
    elif val < 70:
        return 1  # normal
    else:
        return 2  # busy

df['ObsState'] = df['CPU_norm'].apply(discretize)

# ---------------------------
# ✅ STEP 6: Preview
# ---------------------------
print("\n✅ First rows of preprocessed data:")
print(df[['ElapsedTime', 'CPUFrequency', 'CPU_norm', 'ObsState']].head())

print("\n✅ Class distribution:")
print(df['ObsState'].value_counts())

# ---------------------------
# ✅ STEP 7: Save to CSV
# ---------------------------
output_file = "preprocessed_timeseries.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Saved preprocessed data → {output_file}")

