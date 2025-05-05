# StocastiQ: Stochastic Process Modeling of Datacenter GPU Resource Utilization

![StocastiQ Dashboard](./assets/dashboard_screenshot.png)

## 📚 Project Overview

This project presents an interactive Streamlit dashboard applying **Markov Chains**, **Hidden Markov Models (HMM)**, and **Queueing Theory** to analyze GPU resource utilization and job state transitions in a high-performance computing environment.

We used a curated subset from the **MIT SuperCloud Datacenter Challenge dataset (over 2TB of SLURM job logs)** to explore patterns of resource allocation, job lifecycle, and system load in a real datacenter. Our goal was to model job behaviors, hidden system states, and queueing performance through stochastic process techniques.

---

## 📂 Dataset

We accessed the publicly hosted **MIT SuperCloud Datacenter Challenge dataset** via AWS S3 using:

```bash
aws s3 ls s3://datacenter-challenge/202201/ --no-sign-request
