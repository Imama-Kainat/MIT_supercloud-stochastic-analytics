# StocastiQ: Stochastic Process Modeling of Datacenter GPU Resource Utilization

## 📚 Project Overview

This project presents an interactive Streamlit dashboard applying **Markov Chains**, **Hidden Markov Models (HMM)**, and **Queueing Theory** to analyze GPU resource utilization and job state transitions in a high-performance computing environment.

We used a curated subset from the **MIT SuperCloud Datacenter Challenge dataset (over 2TB of SLURM job logs)** to explore patterns of resource allocation, job lifecycle, and system load in a real datacenter. Our goal was to model job behaviors, hidden system states, and queueing performance through stochastic process techniques.

---

## 📂 Dataset

We accessed the publicly hosted **MIT SuperCloud Datacenter Challenge dataset** via AWS S3 using:

```bash
aws s3 ls s3://datacenter-challenge/202201/ --no-sign-request
```
# Dataset Structure

Inside the dataset we identified:

- **Two main folders:** `cpu/` and `gpu/`  
- **Nested sub‑folders:** `0000/` to `0099/` per resource  
- **Paired files for each job:** `*-summary.csv` and `*-timeseries.csv`  

> **Note:** Because the full dataset is ~**2 TB**, we extracted a representative sample of GPU *timeseries* and *summary* files for analysis.

## Key Features Extracted

| Category | Columns |
|----------|---------|
| **Resource metrics** | `CPUUtilization`, `RSS`, `VMSize`, `IORead`, `IOWrite`, `Threads`, `ElapsedTime` |
| **SLURM job metadata** | `time_submit`, `time_start`, `time_end`, `state`, `cpus_req`, `mem_req`, `partition` |

---

# 🏗️ Challenges & Solutions

| Challenge | Our Approach |
|-----------|--------------|
| **Dataset size** | Navigating 2 TB on AWS S3 was non‑trivial. We used the AWS CLI with the `--no-sign-request` flag and **selectively downloaded** only the relevant GPU sub‑folders for local processing. |
| **Data understanding** | The dataset lacked documentation for state definitions. We combined SLURM state logs with timeseries metrics to **infer discrete states** suitable for modelling. |
| **State representation** | We debated using raw SLURM states vs. derived thresholds. We settled on **discretising `CPUUtilization`** into three states:<br>0 = *Idle* (`< 30 %`)<br>1 = *Normal* (`30 – 70 %`)<br>2 = *Busy* (`≥ 70 %`) |
| **Model constraints** | Some subsets lacked certain symbols (e.g., only *Idle*). We built **emission matrices** that gracefully handle missing symbols. |
| **Interpreting results** | We **iteratively validated** transition matrices, emission probabilities and steady‑state distributions against domain expectations. |

---

# 💻 Dashboard Features

| Module | Key Functionality |
|--------|-------------------|
| **Home** | Narrative of dataset journey, challenges & system overview |
| **Markov Chain Analysis** | State transition matrix, steady‑state distribution, visualisation |
| **Hidden Markov Model** | Emission matrix, steady‑state, forward & Viterbi algorithms, plots |
| **Queueing Theory** | Arrival/service rates, average wait time, queue simulation |

Each tab provides both **quantitative outputs** (tables, probabilities) and **visual insights** (Matplotlib charts).

---

# 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/stocastiq.git
cd stocastiq
```
## 📝 Dependencies

- **streamlit**
- **numpy**
- **pandas**
- **hmmlearn**
- **matplotlib**

*Optional*: **AWS CLI** (for dataset fetching)

```bash
pip install streamlit numpy pandas hmmlearn matplotlib
```
# 2. Install dependencies
```
pip install -r requirements.txt
```
# 3. Launch the Streamlit app
```
streamlit run app.py
```
## 🎓 Acknowledgments

- **MIT SuperCloud Datacenter Challenge** team for releasing the dataset  
- **SLURM scheduler documentation** for mapping job‑state codes  
- Open‑source contributors to **hmmlearn**, **streamlit**, and **matplotlib**
