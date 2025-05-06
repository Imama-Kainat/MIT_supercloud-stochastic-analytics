import pandas as pd

# === CONFIGURATION ===
input_file = 'slurm-log.csv'
output_file = 'cleaned_slurm_log.csv'

# === STATE MAPPING ===
state_mapping = {
    3: "COMPLETED",
    5: "FAILED",
    6: "FAILED",       # TIMEOUT → FAILED
    4: "CANCELLED",
    7: "FAILED",       # NODE_FAIL → FAILED
    11: "FAILED"       # OUT_OF_MEMORY → FAILED
}

# === HELPERS ===
def collapse_duplicates(seq):
    """Collapse consecutive duplicate states in a sequence."""
    collapsed = []
    for state in seq:
        if not collapsed or collapsed[-1] != state:
            collapsed.append(state)
    return collapsed

def build_transitions(seq):
    """Build transitions from a state sequence."""
    return [(seq[i], seq[i+1]) for i in range(len(seq)-1)]

# === LOAD DATA ===
print(f"Loading data from {input_file}...")
df = pd.read_csv(input_file)

# Ensure numeric
for col in ['time_submit', 'time_eligible', 'time_start']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate delays
df['queue_delay'] = (df['time_eligible'] - df['time_submit']).clip(lower=0).fillna(0)
df['start_delay'] = (df['time_start'] - df['time_eligible']).clip(lower=0).fillna(0)

# Threshold for adding loop (e.g., 75th percentile)
start_delay_threshold = df['start_delay'].quantile(0.75)
print(f"Start delay threshold (75th percentile): {start_delay_threshold} seconds")

# === PROCESS EACH JOB ===
state_sequences = []
transitions_list = []

for i, row in enumerate(df.itertuples(index=False)):
    if i % 10000 == 0:
        print(f"Processing row {i} of {len(df)}...")

    job_sequence = ['PENDING']
    
    # Add RUNNING if started
    if getattr(row, 'time_start', 0) > 0:
        job_sequence.append('RUNNING')
    
    # Inject loop if start_delay is high
    if getattr(row, 'start_delay', 0) >= start_delay_threshold:
        job_sequence.extend(['PENDING', 'RUNNING'])  # simulate loop
    
    # Add final state
    final_state = state_mapping.get(getattr(row, 'state', None), 'UNKNOWN')
    job_sequence.append(final_state)
    
    # Collapse duplicate consecutive states
    collapsed_seq = collapse_duplicates(job_sequence)
    
    # Build transitions
    transitions = build_transitions(collapsed_seq)
    
    state_sequences.append(collapsed_seq)
    transitions_list.append(transitions)

# Add columns
df['state_sequence'] = state_sequences
df['transitions'] = transitions_list
df['transitions_str'] = [' -> '.join(f"{a}->{b}" for a,b in t) for t in transitions_list]

# SAVE OUTPUT
df.to_csv(output_file, index=False)
print(f"\n✅ Cleaned data with loops saved to: {output_file}")

# Print sample
print("\nSample output:")
print(df[['id_job', 'state_sequence', 'transitions_str']].head(10))
