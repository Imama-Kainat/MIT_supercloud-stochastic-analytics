import pandas as pd
import numpy as np

# 1️⃣ Load dataset
input_file = 'slurm-log.csv'
output_file = 'slurm-log-queueing-clean.csv'

df = pd.read_csv(input_file)

# 2️⃣ Convert UNIX timestamps to datetime
for col in ['time_submit', 'time_eligible', 'time_start', 'time_end']:
    df[col + '_dt'] = pd.to_datetime(df[col], unit='s', errors='coerce')

# 3️⃣ Filter invalid/missing timestamps
valid_rows = (
    (df['time_start'] > 0) &
    (df['time_end'] > 0) &
    (df['time_end'] >= df['time_start']) &
    (df['time_start'] >= df['time_submit'])
)

df_clean = df[valid_rows].copy()

# 4️⃣ Compute service_time in seconds
df_clean['service_time'] = (df_clean['time_end_dt'] - df_clean['time_start_dt']).dt.total_seconds()

# 5️⃣ Compute interarrival_time (seconds between consecutive submits)
df_clean = df_clean.sort_values('time_submit')
df_clean['interarrival_time'] = df_clean['time_submit_dt'].diff().dt.total_seconds()

# 6️⃣ Remove rows with non-positive service/interarrival times
df_clean = df_clean[
    (df_clean['service_time'] > 0) &
    (df_clean['interarrival_time'] > 0)
]

# 7️⃣ REMOVE EXTREME OUTLIERS (beyond 99th percentile for both)
service_threshold = df_clean['service_time'].quantile(0.99)
interarrival_threshold = df_clean['interarrival_time'].quantile(0.99)

df_clean = df_clean[
    (df_clean['service_time'] <= service_threshold) &
    (df_clean['interarrival_time'] <= interarrival_threshold)
]

print(f"Outlier thresholds → service_time ≤ {service_threshold:.2f}s, interarrival_time ≤ {interarrival_threshold:.2f}s")

# 8️⃣ Normalize columns (optional → scaled 0-1 range)
df_clean['service_time_norm'] = (df_clean['service_time'] - df_clean['service_time'].min()) / (df_clean['service_time'].max() - df_clean['service_time'].min())
df_clean['interarrival_time_norm'] = (df_clean['interarrival_time'] - df_clean['interarrival_time'].min()) / (df_clean['interarrival_time'].max() - df_clean['interarrival_time'].min())

# 9️⃣ Handle timelimit sentinel
df_clean['timelimit_clean'] = df_clean['timelimit'].replace(4294967295, np.nan)

# 🔟 Save cleaned dataset
df_clean.to_csv(output_file, index=False)

print(f"✅ Preprocessing complete. Cleaned data saved to: {output_file}")
print(f"Original rows: {len(df)}, Valid rows: {len(df_clean)} after filtering & outlier removal")
