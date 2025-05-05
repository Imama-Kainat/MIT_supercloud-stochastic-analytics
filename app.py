import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from hmmlearn import hmm

# ──────────────────────────────────────────────────────────────────────────
# 1  Load cleaned data
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_slurm_log.csv")   # adjust path if needed

df = load_data()

# ──────────────────────────────────────────────────────────────────────────
# 2  Sidebar navigation
# ──────────────────────────────────────────────────────────────────────────
page = st.sidebar.radio("Navigation", ["Home", "Markov", "Hidden Markov", "Queueing"])

# ──────────────────────────────────────────────────────────────────────────
# 3  Home tab
# ──────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------
# Home page
# ------------------------------------------------------------
if page == "Home":
    import matplotlib.pyplot as plt

    # ──────────────────────────────────────────────────────────
    # Title & Subtitle
    # ──────────────────────────────────────────────────────────
    st.title("Stochastic Processes Project: Modeling GPU Resource Utilization")
    st.caption("Markov Chains · Hidden Markov Models · Queueing Theory")

    # ──────────────────────────────────────────────────────────
    # Welcome Narrative (README‑style)
    # ──────────────────────────────────────────────────────────
    st.markdown(
        """
### 👋 Welcome
This app documents our journey applying **stochastic‑process models** to a **2 TB GPU workload dataset** released by the [MIT SuperCloud Datacenter Challenge](https://supercloud.mit.edu/).  
We focus on three lenses:

* **Markov Chains** — state transitions of GPU utilisation  
* **Hidden Markov Models (HMMs)** — latent workload regimes  
* **Queueing Theory** — arrival / service dynamics of SLURM jobs  

---

### 📂 Dataset at a Glance
| Layer | Details |
|-------|---------|
| Root  | `cpu/` and `gpu/` |
| Sub‑folders | `0000/ … 0099/` (job shards) |
| Per‑job files | `*-summary.csv` + `*-timeseries.csv` |
| Total size | **≈ 2 TB** (public AWS S3 bucket) |

We analysed a **representative GPU sample** to keep local storage humane.

*Dataset link →* **`s3://mit-ll-supercloud-dc/data/`**

---

### 🔑 Extracted Features
* **Resource metrics**: `CPUUtilization`, `RSS`, `VMSize`, `IORead`, `IOWrite`, `Threads`, `ElapsedTime`  
* **Job metadata**: `time_submit`, `time_start`, `time_end`, `state`, `cpus_req`, `mem_req`, `partition`

---

### 🏗️ Challenges & Solutions
| Pain‑point | What we did |
|------------|-------------|
| **2 TB size** | `aws s3 cp --no-sign-request --recursive …` on only **gpu/** shards |
| **Undocumented states** | Combined SLURM codes + utilisation thresholds ➜ Idle / Normal / Busy |
| **Sparse symbols** | Emission matrices handle missing observations |
| **Validation** | Cross‑checked steady‑state vectors with domain intuition |

---

### 🚀 Quick‑Start

```bash
git clone https://github.com/yourusername/stocastiq.git
cd stocastiq
pip install -r requirements.txt
streamlit run app.py         # loads a bundled sample file
```
## 🔗 Live Repository

Browse the complete project source on GitHub:  
**<https://github.com/ZainabEman/MIT_supercloud-stochastic-analytics.git>**

---

## 📦 Dataset Access

- **Public AWS S3 bucket** (raw files, ≈ 2 TB):  
  `s3://mit-ll-supercloud-dc/`

- **Dataset landing page / paper** (overview & citation):  
  <https://arxiv.org/abs/2106.09701>

---

## 🎓 Acknowledgments

- **MIT SuperCloud Datacenter Challenge** team for releasing the large‑scale workload dataset  
- **SLURM** scheduler documentation and community contributors for elucidating job‑state codes  
- Open‑source maintainers of **hmmlearn**, **streamlit**, and **matplotlib** for the libraries powering our analyses  
- Everyone who reviewed the codebase, filed issues, or submitted pull requests to improve this project  

    """)


# ──────────────────────────────────────────────────────────────────────────
# 4  Markov‑chain analysis
# ──────────────────────────────────────────────────────────────────────────
elif page == "Markov":
    st.title("Markov Chain Analysis (with Pending ↔ Running loops)")

    st.subheader("Overview")
    st.write(
        """
        States: **PENDING → RUNNING → [COMPLETED | FAILED | CANCELLED]**.  
        Some jobs retry, yielding **RUNNING → PENDING → RUNNING** loops before absorption.
        """
    )

    # 4‑A  Build empirical transition matrix ───────────────────────────────
    states            = ["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]
    absorbing_set     = {"COMPLETED", "FAILED", "CANCELLED"}
    absorbing_list    = [s for s in states if s in absorbing_set]   # fixed order

    counts = {s: {t: 0 for t in states} for s in states}
    for trans_list in df["transitions"]:
        for a, b in eval(trans_list):
            if a in states and b in states:
                counts[a][b] += 1

    probs = {}
    for s in states:
        tot = sum(counts[s].values())
        if s in absorbing_set:
            probs[s] = {t: 1.0 if t == s else 0.0 for t in states}
        else:
            probs[s] = {t: counts[s][t] / tot if tot else 0.0 for t in states}

    matrix_df = pd.DataFrame(probs).T
    P         = matrix_df.to_numpy()

    st.subheader("Transition‑Probability Matrix")
    st.table(matrix_df)

    # 4‑B  Heatmap ─────────────────────────────────────────────────────────
    st.subheader("Heatmap")
    fig_hm, ax = plt.subplots()
    sns.heatmap(P, annot=True, fmt=".4f", cmap="Blues",
                xticklabels=states, yticklabels=states, ax=ax)
    st.pyplot(fig_hm)

    # 4‑C  State diagram ───────────────────────────────────────────────────
    st.subheader("Empirical State Diagram")
    G = nx.DiGraph()
    for a in states:
        for b in states:
            if matrix_df.loc[a, b] > 0:
                G.add_edge(a, b, weight=matrix_df.loc[a, b])

    pos = nx.circular_layout(G)
    node_colors = ["gold" if s == "PENDING"
                   else "skyblue" if s == "RUNNING"
                   else "lightcoral" for s in states]

    node_trace = go.Scatter(
        x=[pos[s][0] for s in states],
        y=[pos[s][1] for s in states],
        text=states,
        mode="markers+text",
        textposition="middle center",
        marker=dict(size=55, color=node_colors, line=dict(width=2, color="black")),
        hoverinfo="text"
    )

    def arrow(x0, y0, x1, y1, shift=0.0):
        return dict(ax=x0+shift, ay=y0-shift, x=x1+shift, y=y1-shift,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor="black")

    arrows = []
    for a, b in G.edges():
        x0, y0 = pos[a]; x1, y1 = pos[b]
        if {a, b} == {"PENDING", "RUNNING"}:
            s = 0.04 if a == "PENDING" else -0.04
            arrows.append(arrow(x0, y0, x1, y1, s))
        else:
            arrows.append(arrow(x0, y0, x1, y1))

    st.plotly_chart(
        go.Figure(
            data=[node_trace],
            layout=go.Layout(
                annotations=arrows, showlegend=False, hovermode="closest",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                height=640, margin=dict(t=40, l=20, r=20, b=20),
                title="State‑Transition Diagram"
            )
        )
    )
    st.markdown("🟡 PENDING  🔵 RUNNING  🔴 Absorbing")

    # 4‑D  Fundamental matrix & metrics ────────────────────────────────────
    idx            = {s: i for i, s in enumerate(states)}
    absorbing_idx  = [idx[s] for s in absorbing_list]
    transient_idx  = [i for i in range(len(states)) if i not in absorbing_idx]

    if transient_idx:
        Q = P[np.ix_(transient_idx, transient_idx)]
        try:
            N = np.linalg.inv(np.eye(len(Q)) - Q)                    # fundamental
        except np.linalg.LinAlgError:
            N = None
    else:
        N = None

    # Pre‑compute B = N R and U = N² R  (needed for per‑absorber times) ----
    if N is not None:
        R = P[np.ix_(transient_idx, absorbing_idx)]
        B = N @ R                       # hitting probabilities to each absorber
        U = (N @ N) @ R                 # unconditional time‑totals
    else:
        R = B = U = None

    st.subheader("⏱️ Markov Calculations")

    # (i) Recurrence & per‑absorber absorption -----------------------------
    sel_state = st.selectbox("Choose a state", states, key="rec_sel")
    i_sel     = idx[sel_state]

    colA, colB = st.columns(2)

    with colA:
        st.markdown(" Mean recurrence time")
        if i_sel in absorbing_idx:
            st.success("∞ (absorbing)")
        else:
            if P[i_sel, i_sel] < 1:
                st.success(f"{1/(1-P[i_sel,i_sel]):.4f} steps")
            else:
                st.info("Self‑loop = 1 → ∞")

    with colB:
        st.markdown(" Mean absorption time\n*(to each absorbing state)*")
        if i_sel in absorbing_idx:
            st.success("0 steps (already absorbing)")
        elif N is None:
            st.warning("Cannot compute (singular matrix).")
        else:
            row = transient_idx.index(i_sel)
            for a_idx, a_name in zip(absorbing_idx, absorbing_list):
                prob = B[row, absorbing_idx.index(a_idx)]
                if prob == 0:
                    st.write(f"• **{a_name}**: unreachable (prob 0)")
                else:
                    mean_k = U[row, absorbing_idx.index(a_idx)] / prob
                    st.write(f"• **{a_name}**: {mean_k:.4f} steps")

    # (ii) Mean passage time src → tgt ------------------------------------
    st.markdown(" Mean passage time between two states")
    col1, col2 = st.columns(2)
    with col1:
        src_state = st.selectbox("From", states, key="pass_src")
    with col2:
        tgt_state = st.selectbox("To",   states, key="pass_tgt")

    if src_state == tgt_state:
        st.info("Source = target → 0")
    else:
        P_mod           = P.copy()
        j_tgt           = idx[tgt_state]
        P_mod[j_tgt, :] = 0.0
        P_mod[j_tgt, j_tgt] = 1.0
        new_abs         = absorbing_idx + ([] if j_tgt in absorbing_idx else [j_tgt])
        new_trans       = [k for k in range(len(states)) if k not in new_abs]
        if new_trans:
            Qm = P_mod[np.ix_(new_trans, new_trans)]
            try:
                Nm = np.linalg.inv(np.eye(len(Qm)) - Qm)
                i_from   = new_trans.index(idx[src_state])
                meanpass = Nm[i_from, :].sum()
                st.success(f"{meanpass:.4f} steps")
            except np.linalg.LinAlgError:
                st.warning("Singular (I−Q) – cannot compute.")
        else:
            st.info("All states absorbing under this modification.")

    # 4‑E  Hitting probability --------------------------------------------
    st.markdown("---")
    st.subheader("🎯 Long‑run hitting probability")

    col3, col4 = st.columns(2)
    with col3:
        from_state = st.selectbox("From state", states, key="hit_from")
    with col4:
        to_state   = st.selectbox("To state", states, key="hit_to")

    if from_state == to_state:
        st.success("1")
    elif idx[from_state] in absorbing_idx:
        st.info("0 (already in a different absorber)")
    else:
        if idx[to_state] in absorbing_idx and N is not None:
            i = transient_idx.index(idx[from_state])
            j = absorbing_idx.index(idx[to_state])
            st.success(f"Probability: **{B[i,j]:.4f}**")
        else:
            P_tmp              = P.copy()
            j_tgt              = idx[to_state]
            P_tmp[j_tgt, :]    = 0.0
            P_tmp[j_tgt,j_tgt] = 1.0
            abs_tmp            = absorbing_idx + [j_tgt]
            trans_tmp          = [k for k in range(len(states)) if k not in abs_tmp]
            Qh = P_tmp[np.ix_(trans_tmp, trans_tmp)]
            Rh = P_tmp[np.ix_(trans_tmp, abs_tmp)]
            try:
                Nh  = np.linalg.inv(np.eye(len(Qh)) - Qh)
                Bh  = Nh @ Rh
                i   = trans_tmp.index(idx[from_state])
                j   = abs_tmp.index(j_tgt)
                st.success(f"Probability: **{Bh[i,j]:.4f}**")
            except np.linalg.LinAlgError:
                st.warning("Singular (I−Q) – cannot compute.")

# ──────────────────────────────────────────────────────────────────────────
# 5  Placeholder pages
# ──────────────────────────────────────────────────────────────────────────
elif page == "Hidden Markov":
    st.title("🔍 Hidden Markov Model Analysis")

    st.write("""
    This tab applies a Hidden Markov Model (HMM) to the dataset to:
    1. Estimate steady-state probabilities
    2. Compute probability of observing a state sequence (Forward Algorithm)
    3. Infer the most likely hidden state sequence (Viterbi Algorithm)
    """)

    # ✅ Load CSV
    df = pd.read_csv("10066852034034-timeseries.csv")

    # ✅ Check if CPUFrequency exists
    if 'CPUFrequency' not in df.columns:
        st.error("Column 'CPUFrequency' not found in dataset!")
        st.stop()

    # ✅ Step 1: Preview
    st.subheader("Step 1: Data Preview")
    st.write(df[['ElapsedTime', 'CPUFrequency']].head())

    # ✅ Step 2: Normalize CPUFrequency (0–100 scale)
    df['CPUFrequency_normalized'] = df['CPUFrequency'] / df['CPUFrequency'].max() * 100

    # ✅ Discretize
    def discretize(util):
        if util < 30:
            return 0  # Idle
        elif util < 70:
            return 1  # Normal
        else:
            return 2  # Busy

    df['ObsState'] = df['CPUFrequency_normalized'].apply(discretize)

    st.subheader("Step 2: Discretized Observed States")
    st.write(df[['ElapsedTime', 'CPUFrequency_normalized', 'ObsState']].head())

    # ✅ Observed sequence
    observations = df['ObsState'].values.reshape(-1, 1)

    # ✅ Fit HMM
    n_components = 3
    model = hmm.MultinomialHMM(n_components=n_components, n_iter=100, random_state=42)
    model.fit(observations)

    # ✅ Transition Matrix
    st.subheader("Step 3: Transition Matrix (A)")
    transmat_df = pd.DataFrame(model.transmat_, columns=[f"State {i}" for i in range(n_components)])
    st.write(transmat_df)

    # ✅ Emission Matrix
    st.subheader("Step 4: Emission Probabilities (B)")
    emission_probs = model.emissionprob_

    # Build emission dataframe → fill missing cols if fewer symbols
    cols = ["Idle", "Normal", "Busy"]
    if emission_probs.shape[1] < 3:
        emiss_df = pd.DataFrame(0, index=[f"State {i}" for i in range(n_components)], columns=cols)
        for i, symbol in enumerate(np.unique(df['ObsState'])):
            emiss_df[cols[symbol]] = emission_probs[:, i]
    else:
        emiss_df = pd.DataFrame(emission_probs, index=[f"State {i}" for i in range(n_components)], columns=cols)

    st.write(emiss_df)

    # ✅ Steady-State Probabilities
    st.subheader("Step 5: Steady-State Probabilities")
    eigvals, eigvecs = np.linalg.eig(model.transmat_.T)
    steady_state = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    steady_state = steady_state[:, 0] / steady_state[:, 0].sum()
    steady_df = pd.DataFrame(steady_state, index=[f"State {i}" for i in range(n_components)], columns=["Probability"])
    st.write(steady_df)

    # ✅ Forward Algorithm
    st.subheader("Step 6: Forward Algorithm")
    log_prob = model.score(observations)
    st.write(f"Log Probability of observation sequence: {log_prob:.4f}")
    st.write(f"Probability of observation sequence: {np.exp(log_prob):.6f}")

    # ✅ Viterbi Algorithm
    st.subheader("Step 7: Viterbi Algorithm (Most Likely Hidden States)")
    hidden_states = model.predict(observations)
    df['HiddenState'] = hidden_states
    st.write(df[['ElapsedTime', 'CPUFrequency_normalized', 'ObsState', 'HiddenState']].head())

    # ✅ Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['ElapsedTime'], df['CPUFrequency'], label='CPUFrequency')
    ax.scatter(df['ElapsedTime'], df['HiddenState'] * df['CPUFrequency'].max() / 2, color='red', label='Hidden State (scaled)')
    ax.set_xlabel("ElapsedTime")
    ax.set_ylabel("CPUFrequency / HiddenState")
    ax.legend()
    st.pyplot(fig)



elif page == "Queueing":
    st.title("Queueing Theory – Coming Soon")
