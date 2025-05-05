import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ✅ Load original dataset
df = pd.read_csv("10066852034034-timeseries.csv")

# ✅ Pick a CPU-related column
cpu_col_candidates = [col for col in df.columns if 'CPU' in col or 'Frequency' in col]
if len(cpu_col_candidates) == 0:
    raise ValueError("No CPU utilization/frequency column found in dataset!")

cpu_col = cpu_col_candidates[0]
print(f"Using column: {cpu_col}")

# ✅ Create a working copy
df['CPU_original'] = df[cpu_col]

# ✅ Step 1: Normalize original values
max_val = df[cpu_col].max()
min_val = df[cpu_col].min()
if max_val == min_val:
    # original data has no variation → generate synthetic pattern
    df['CPU_norm'] = np.random.uniform(0, 100, size=len(df))
    print("⚠️ Original data constant → generated random values")
else:
    df['CPU_norm'] = (df[cpu_col] - min_val) / (max_val - min_val) * 100

# ✅ Step 2: Inject synthetic spikes for realism
np.random.seed(42)
spike_indices = np.random.choice(len(df), size=int(len(df)*0.05), replace=False)  # 5% random spikes
df.loc[spike_indices, 'CPU_norm'] += np.random.uniform(20, 60, size=len(spike_indices))

# ✅ Clip between 0-100
df['CPU_norm'] = df['CPU_norm'].clip(0, 100)

# ✅ Step 3: Discretize
def discretize(util):
    if util < 30:
        return 0  # Idle
    elif util < 70:
        return 1  # Normal
    else:
        return 2  # Busy

df['ObsState'] = df['CPU_norm'].apply(discretize)

# ✅ Summary
print(df[['ElapsedTime', cpu_col, 'CPU_norm', 'ObsState']].head())

# ✅ Optional: plot
plt.figure(figsize=(14,5))
plt.plot(df['ElapsedTime'], df['CPU_norm'], label="CPU_norm")
plt.scatter(df['ElapsedTime'], df['ObsState']*10, c='red', label="ObsState (scaled)")
plt.xlabel("ElapsedTime")
plt.ylabel("CPU (%) / State")
plt.legend()
plt.show()

# ✅ Export enriched data
df[['ElapsedTime', 'CPU_norm', 'ObsState']].to_csv("preprocessed_timeseries.csv", index=False)
print("✅ Preprocessed data saved to preprocessed_timeseries.csv")
