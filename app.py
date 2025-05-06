import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from hmmlearn import hmm
import plotly.express as px
from scipy.stats import kstest, expon
import io

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
st.sidebar.markdown("""
[📂 View Source Code on GitHub](https://github.com/ZainabEman/MIT_supercloud-stochastic-analytics.git)
""")
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
git clone https://github.com/ZainabEman/MIT_supercloud-stochastic-analytics.git
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

    # ✅ Dynamic thresholds
    q1 = df['CPUFrequency_normalized'].quantile(0.33)
    q2 = df['CPUFrequency_normalized'].quantile(0.66)

    def discretize(util):
        if util < q1:
            return 0  # Idle
        elif util < q2:
            return 1  # Normal
        else:
            return 2  # Busy

    df['ObsState'] = df['CPUFrequency_normalized'].apply(discretize)

    st.subheader("Step 2: Discretized Observed States")
    st.write(df[['ElapsedTime', 'CPUFrequency_normalized', 'ObsState']].head())

    # ✅ Check unique observed states
    unique_states = np.unique(df['ObsState'])
    st.write(f"Unique Observed States: {unique_states}")

    # ✅ Inject synthetic samples if only 1 unique state
    if len(unique_states) < 2:
        st.warning("Only one observed state detected → injecting synthetic samples for demonstration.")
        df = pd.concat([df, pd.DataFrame({
            'ElapsedTime': [-1, -2],
            'CPUFrequency': [df['CPUFrequency'].max(), df['CPUFrequency'].min()],
            'CPUFrequency_normalized': [99, 1],
            'ObsState': [2, 0]
        })], ignore_index=True)
        unique_states = np.unique(df['ObsState'])
        st.write(f"After injection → Unique Observed States: {unique_states}")

    # ✅ Observed sequence
    observations = df['ObsState'].values.reshape(-1, 1)

    # ✅ Fit HMM
    from hmmlearn import hmm
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

    cols = ["Idle", "Normal", "Busy"]
    emiss_df = pd.DataFrame(0, index=[f"State {i}" for i in range(n_components)], columns=cols)

    symbols_present = np.unique(df['ObsState'])
    model_symbols = emission_probs.shape[1]

    for symbol in symbols_present:
        if symbol >= model_symbols:
            st.warning(f"Symbol {symbol} not learned by HMM → skipping assignment.")
            continue
        col_name = cols[symbol]
        emiss_df[col_name] = emission_probs[:, np.where(np.unique(observations) == symbol)[0][0]]

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
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['ElapsedTime'], df['CPUFrequency'], label='CPUFrequency')
    ax.scatter(df['ElapsedTime'], df['HiddenState'] * df['CPUFrequency'].max() / 2,
               color='red', label='Hidden State (scaled)')
    ax.set_xlabel("ElapsedTime")
    ax.set_ylabel("CPUFrequency / HiddenState")
    ax.legend()
    st.pyplot(fig)

    # ✅ INTERPRETATION
    st.subheader("Step 8: Interpretation of Results")
    st.markdown("""
    **📝 Interpretation Summary:**

    - The **Transition Matrix (A)** shows the probabilities of moving between hidden states (State 0, State 1, State 2).
      For example, a high value on `State 0 → State 0` means the system tends to stay idle.

    - The **Emission Probabilities (B)** tell us how likely each hidden state emits an observed state (Idle, Normal, Busy).
      For example, if `State 1` has a high probability for `Normal`, that state likely represents normal usage.

    - The **Steady-State Probabilities** indicate the long-run percentage of time the system stays in each hidden state.
      For example, a steady state of `State 2: 0.70` implies 70% of the time the CPU is busy.

    - The **Log Probability** from the Forward Algorithm tells us how well the model explains the observed sequence:
      higher values mean better fit.

    - The **Viterbi Algorithm** outputs the most likely hidden state path for the data → useful for inferring the CPU's operational mode over time.

    - The final plot overlays the original CPU frequency and predicted hidden state sequence over time → to visually compare how hidden states align with CPU activity.

    ✅ **In simple terms: This analysis modeled how CPU usage fluctuates between hidden operational modes (idle, normal, busy) over time, and estimated how likely transitions and states are happening based on the data.**
    """)



elif page == "Queueing":
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    from scipy.stats import kstest, expon
    import io

    st.title("📊 M/M/1 Queuing Theory Dashboard (Synthetic Data Mode)")
    st.caption("Dataset: synthetic exponential data → analysis in SECONDS")

    # 🔹 Synthetic data generator
    @st.cache_data
    def generate_synthetic_data(size=1000, interarrival_mean=60, service_mean=45):
        np.random.seed(42)
        interarrival_times = np.random.exponential(scale=interarrival_mean, size=size)
        service_times = np.random.exponential(scale=service_mean, size=size)
        return pd.DataFrame({
            'interarrival_time': interarrival_times,
            'service_time': service_times
        })

    df = generate_synthetic_data()

    st.dataframe(df.head())

    # 🔹 Calculate λ and μ
    lam = 1 / df['interarrival_time'].mean()
    mu = 1 / df['service_time'].mean()

    # 🔹 M/M/1 metrics function
    def mm1_metrics(lam, mu, n=5, k=10):
        rho = lam / mu
        if rho >= 1:
            return {"rho": rho}
        L = rho / (1 - rho)
        Lq = rho**2 / (1 - rho)
        W = 1 / (mu - lam)
        Wq = lam / (mu * (mu - lam))
        P0 = 1 - rho
        Pn = (1 - rho) * rho**n
        Pgt_k = rho**(k + 1)
        return dict(rho=rho, L=L, Lq=Lq, W=W, Wq=Wq, P0=P0, Pn=Pn, Pgt_k=Pgt_k)

    metrics = mm1_metrics(lam, mu)

    st.subheader("🔢 M/M/1 Metrics (in SECONDS)")

    if metrics.get("rho", 2) >= 1:
        st.error(f"⚠️ System unstable (ρ = {metrics['rho']:.3f} ≥ 1). Metrics invalid.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("λ (arrival rate)", f"{lam:.3f} /sec")
        c2.metric("μ (service rate)", f"{mu:.3f} /sec")
        c3.metric("ρ (utilization)", f"{metrics['rho']:.3f}", delta="⚠️" if metrics['rho'] > 0.9 else None)
        c4.metric("P₀ (idle prob)", f"{metrics['P0']:.3f}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("L (avg system)", f"{metrics['L']:.2f}")
        c2.metric("Lq (avg queue)", f"{metrics['Lq']:.2f}")
        c3.metric("W (time system)", f"{metrics['W']:.2f} sec")
        c4.metric("Wq (wait time)", f"{metrics['Wq']:.2f} sec")

        st.markdown(f"""
        - **Pₙ (n=5 customers):** {metrics['Pn']:.4f}
        - **P(N > 10 customers):** {metrics['Pgt_k']:.4f}
        """)

        st.subheader("🧠 Interpretation")
        if metrics['rho'] > 0.9:
            st.warning("⚠️ Utilization exceeds 90% → consider adding capacity.")
        else:
            st.success("✅ System stable with moderate utilization.")

    # 🔹 KS p-value
    def ks_pvalue(sample):
        if len(sample) < 50:
            return np.nan
        mean = np.mean(sample)
        return kstest(sample, expon(scale=mean).cdf).pvalue

    pval_ia = ks_pvalue(df['interarrival_time'])
    pval_sv = ks_pvalue(df['service_time'])

    # 🔹 Plots
    st.subheader("📈 Interarrival Time Distributions")

    fig_ia_lin = px.histogram(df, x='interarrival_time', nbins=50,
                                labels={"interarrival_time": "Interarrival time (sec)"},
                                title=f"Interarrival time (linear scale)\nKS p = {pval_ia:.4f}")
    st.plotly_chart(fig_ia_lin, use_container_width=True)

    st.subheader("📈 Service Time Distributions")

    fig_sv_lin = px.histogram(df, x='service_time', nbins=50,
                                labels={"service_time": "Service time (sec)"},
                                title=f"Service time (linear scale)\nKS p = {pval_sv:.4f}")
    st.plotly_chart(fig_sv_lin, use_container_width=True)
    # 🔹 KDE plots
    st.subheader("📊 Kernel Density Estimations (KDE)")

    fig_kde = px.line()
    fig_kde.add_scatter(x=np.sort(df['interarrival_time']), 
                        y=expon.pdf(np.sort(df['interarrival_time']), scale=df['interarrival_time'].mean()),
                        mode='lines', name='Interarrival KDE')
    fig_kde.add_scatter(x=np.sort(df['service_time']), 
                        y=expon.pdf(np.sort(df['service_time']), scale=df['service_time'].mean()),
                        mode='lines', name='Service KDE')
    fig_kde.update_layout(title="Exponential fit (PDF overlaid)")
    st.plotly_chart(fig_kde, use_container_width=True)

    # 🔹 Boxplots
    st.subheader("📦 Boxplots")

    fig_box = px.box(df.melt(value_vars=['interarrival_time', 'service_time'], var_name='Type', value_name='Seconds'),
                     x='Type', y='Seconds', log_y=True,
                     title="Boxplot of interarrival vs service times (log Y)")
    st.plotly_chart(fig_box, use_container_width=True)

    # 🔹 Utilization-performance curve
    if metrics.get("rho", 2) < 1:
        st.subheader("📈 Utilization-Performance Curve")
        util_range = np.linspace(0.05, 0.99, 100)
        L_curve = util_range / (1 - util_range)
        fig_curve = px.line(x=util_range, y=L_curve,
                            labels={"x": "ρ", "y": "L"},
                            title="Avg number in system vs utilization (M/M/1)")
        fig_curve.add_vline(x=metrics["rho"], line_dash="dash", annotation_text="current ρ")
        st.plotly_chart(fig_curve, use_container_width=True)

    # 🔹 Export
    if st.button("📥 Export Metrics as CSV"):
        if metrics.get("rho", 2) >= 1:
            st.error("⚠️ Cannot export – system unstable.")
        else:
            csv_buf = io.StringIO()
            pd.DataFrame([metrics]).to_csv(csv_buf, index=False)
            st.download_button("Download CSV", csv_buf.getvalue(),
                               file_name="mm1_metrics.csv", mime="text/csv")
