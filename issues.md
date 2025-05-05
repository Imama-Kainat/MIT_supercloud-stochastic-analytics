## 📝 **💯 COMPLETE DETAILED EXPLANATION OF DATA & MODELING CHALLENGES (Markov + HMM)**

---

### ## 1️⃣ **Dataset Exploration: The Beginning**

We started by accessing the  **MIT SuperCloud Datacenter Challenge dataset** , a large-scale SLURM job log dataset (~2TB) hosted on AWS S3.

✅ We listed files using AWS CLI (`--no-sign-request`) and identified the folder structure: `cpu/`, `gpu/`, and hundreds of job subfolders.

We focused on a **GPU timeseries job sample** by downloading a single `*-summary.csv` and corresponding `*-timeseries.csv` for testing.

At this stage, we faced:

| Challenge            | Description                                                       |
| -------------------- | ----------------------------------------------------------------- |
| Large data volume    | Only practical to work with small sample locally                  |
| Unfamiliar structure | Unclear what each CSV column represented; needed schema inference |
| Feature ambiguity    | Several similarly named columns; no documentation for thresholds  |

---

## 2️⃣ **Selecting Features for Modeling**

After examining the columns, we chose:

* `CPUUtilization` → proxy for job load
* `ElapsedTime` → for timeline
* `RSS`, `IORead`, `IOWrite`, `Threads` → secondary features

We noticed:

✅ `CPUUtilization` had extreme values (e.g., 3000%) → far exceeding physical limits

✅ Median value was **0** → **half the rows recorded no CPU activity at all**

---

## 3️⃣ **Modeling as Markov Chain: First Problems**

Our plan:

→ discretize `CPUUtilization` into 3 observable states: `Idle`, `Normal`, `Busy`

→ compute **state transition matrix**

→ derive **steady-state probabilities**

Initial thresholds:

<pre class="overflow-visible!" data-start="1990" data-end="2092"><div class="contain-inline-size rounded-md border-[0.5px] border-token-border-medium relative bg-token-sidebar-surface-primary"><div class="flex items-center text-token-text-secondary px-4 py-2 text-xs font-sans justify-between h-9 bg-token-sidebar-surface-primary dark:bg-token-main-surface-secondary select-none rounded-t-[5px]">python</div><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-sidebar-surface-primary text-token-text-secondary dark:bg-token-main-surface-secondary flex items-center rounded-sm px-2 font-sans text-xs"><span class="" data-state="closed"><button class="flex gap-1 items-center select-none px-4 py-1" aria-label="Copy"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="icon-xs"><path fill-rule="evenodd" clip-rule="evenodd" d="M7 5C7 3.34315 8.34315 2 10 2H19C20.6569 2 22 3.34315 22 5V14C22 15.6569 20.6569 17 19 17H17V19C17 20.6569 15.6569 22 14 22H5C3.34315 22 2 20.6569 2 19V10C2 8.34315 3.34315 7 5 7H7V5ZM9 7H14C15.6569 7 17 8.34315 17 10V15H19C19.5523 15 20 14.5523 20 14V5C20 4.44772 19.5523 4 19 4H10C9.44772 4 9 4.44772 9 5V7ZM5 9C4.44772 9 4 9.44772 4 10V19C4 19.5523 4.44772 20 5 20H14C14.5523 20 15 19.5523 15 19V10C15 9.44772 14.5523 9 14 9H5Z" fill="currentColor"></path></svg>Copy</button></span><span class="" data-state="closed"><button class="flex items-center gap-1 px-4 py-1 select-none"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="icon-xs"><path d="M2.5 5.5C4.3 5.2 5.2 4 5.5 2.5C5.8 4 6.7 5.2 8.5 5.5C6.7 5.8 5.8 7 5.5 8.5C5.2 7 4.3 5.8 2.5 5.5Z" fill="currentColor" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"></path><path d="M5.66282 16.5231L5.18413 19.3952C5.12203 19.7678 5.09098 19.9541 5.14876 20.0888C5.19933 20.2067 5.29328 20.3007 5.41118 20.3512C5.54589 20.409 5.73218 20.378 6.10476 20.3159L8.97693 19.8372C9.72813 19.712 10.1037 19.6494 10.4542 19.521C10.7652 19.407 11.0608 19.2549 11.3343 19.068C11.6425 18.8575 11.9118 18.5882 12.4503 18.0497L20 10.5C21.3807 9.11929 21.3807 6.88071 20 5.5C18.6193 4.11929 16.3807 4.11929 15 5.5L7.45026 13.0497C6.91175 13.5882 6.6425 13.8575 6.43197 14.1657C6.24513 14.4392 6.09299 14.7348 5.97903 15.0458C5.85062 15.3963 5.78802 15.7719 5.66282 16.5231Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path><path d="M14.5 7L18.5 11" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>Edit</button></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-python"><span><span>if</span><span> util < </span><span>30</span><span>: </span><span>return</span><span></span><span>0</span><span></span><span># Idle</span><span>
</span><span>elif</span><span> util < </span><span>70</span><span>: </span><span>return</span><span></span><span>1</span><span></span><span># Normal</span><span>
</span><span>else</span><span>: </span><span>return</span><span></span><span>2</span><span></span><span># Busy</span><span>
</span></span></code></div></div></pre>

BUT:

| Problem         | Effect                                           |
| --------------- | ------------------------------------------------ |
| 50% data = 0    | half dataset fell into Idle                      |
| 25% ≥ 100      | no values in 30–70 range → Normal almost empty |
| Outliers > 3000 | "Busy" overloaded by invalid high values         |

✅ Transition matrix ended up  **biased toward Idle** , with almost no observed transitions out of Idle.

✅ Steady state trivially converged to Idle → **no meaningful insight.**

---

## 4️⃣ **Options Considered for Markov**

We brainstormed:

| Option                   | Description                  | Pros                    | Cons                                             |
| ------------------------ | ---------------------------- | ----------------------- | ------------------------------------------------ |
| Keep original thresholds | keep 30/70 split             | simple                  | Normal state empty, transition matrix degenerate |
| Shift thresholds lower   | e.g., 1/50 split             | balances Normal more    | arbitrary without justification                  |
| Clip values at 100       | cap impossible >100%         | fixes invalid           | still half data = 0                              |
| Remove zeros             | focus only on active samples | only valid CPU activity | loses idle state info                            |

✅ We selected **clip + threshold adjustment (1,50)** to:

* remove invalid >100% values
* ensure Normal state is populated
* keep Idle meaningful (i.e., util=0)

---

## 5️⃣ **Moving to Hidden Markov Model: New Challenges**

HMM needs:

* Observed sequence → from `CPUUtilization`
* Hidden states → learned latent process

Initial runs gave:

✅ **emission matrix:** each hidden state emits only Idle with 100%

✅ **transition matrix:** hidden states barely transition → stuck in one state

✅ **Viterbi:** hidden path flat

→ why?

👉 **because observed data = 0 for 50% rows; other 50% saturate at ≥100**

👉 HMM sees **no variability → learns trivial model mapping everything to Idle emission.**

---

## 6️⃣ **Diagnosis: Root Cause**

 **Data sparsity in active states** :

✅ too many rows idle

✅ when not idle → jumps directly to max

→ no smooth variation in observed `CPUUtilization` → impossible for HMM to learn meaningful emission/transition probabilities.

---

## 7️⃣ **Options We Considered for HMM:**

| Option           | Description                         | Pros                       | Cons                                    |
| ---------------- | ----------------------------------- | -------------------------- | --------------------------------------- |
| Keep thresholds  | 0/30/70                             | preserves initial plan     | Normal state empty; emission degenerate |
| Lower thresholds | 1/50                                | enables more mid-range     | arbitrary cutoff; still skewed          |
| Remove idle      | only active jobs                    | learns active transitions  | loses Idle state; incomplete process    |
| Resample/smooth  | interpolate missing / smooth spikes | fills gaps; adds variation | artificially changes raw data           |

✅ We selected **Option A+B** → **clip at 100, redefine thresholds at 1 and 50:**

<pre class="overflow-visible!" data-start="5214" data-end="5389"><div class="contain-inline-size rounded-md border-[0.5px] border-token-border-medium relative bg-token-sidebar-surface-primary"><div class="flex items-center text-token-text-secondary px-4 py-2 text-xs font-sans justify-between h-9 bg-token-sidebar-surface-primary dark:bg-token-main-surface-secondary select-none rounded-t-[5px]">python</div><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-sidebar-surface-primary text-token-text-secondary dark:bg-token-main-surface-secondary flex items-center rounded-sm px-2 font-sans text-xs"><span class="" data-state="closed"><button class="flex gap-1 items-center select-none px-4 py-1" aria-label="Copy"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="icon-xs"><path fill-rule="evenodd" clip-rule="evenodd" d="M7 5C7 3.34315 8.34315 2 10 2H19C20.6569 2 22 3.34315 22 5V14C22 15.6569 20.6569 17 19 17H17V19C17 20.6569 15.6569 22 14 22H5C3.34315 22 2 20.6569 2 19V10C2 8.34315 3.34315 7 5 7H7V5ZM9 7H14C15.6569 7 17 8.34315 17 10V15H19C19.5523 15 20 14.5523 20 14V5C20 4.44772 19.5523 4 19 4H10C9.44772 4 9 4.44772 9 5V7ZM5 9C4.44772 9 4 9.44772 4 10V19C4 19.5523 4.44772 20 5 20H14C14.5523 20 15 19.5523 15 19V10C15 9.44772 14.5523 9 14 9H5Z" fill="currentColor"></path></svg>Copy</button></span><span class="" data-state="closed"><button class="flex items-center gap-1 px-4 py-1 select-none"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="icon-xs"><path d="M2.5 5.5C4.3 5.2 5.2 4 5.5 2.5C5.8 4 6.7 5.2 8.5 5.5C6.7 5.8 5.8 7 5.5 8.5C5.2 7 4.3 5.8 2.5 5.5Z" fill="currentColor" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"></path><path d="M5.66282 16.5231L5.18413 19.3952C5.12203 19.7678 5.09098 19.9541 5.14876 20.0888C5.19933 20.2067 5.29328 20.3007 5.41118 20.3512C5.54589 20.409 5.73218 20.378 6.10476 20.3159L8.97693 19.8372C9.72813 19.712 10.1037 19.6494 10.4542 19.521C10.7652 19.407 11.0608 19.2549 11.3343 19.068C11.6425 18.8575 11.9118 18.5882 12.4503 18.0497L20 10.5C21.3807 9.11929 21.3807 6.88071 20 5.5C18.6193 4.11929 16.3807 4.11929 15 5.5L7.45026 13.0497C6.91175 13.5882 6.6425 13.8575 6.43197 14.1657C6.24513 14.4392 6.09299 14.7348 5.97903 15.0458C5.85062 15.3963 5.78802 15.7719 5.66282 16.5231Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path><path d="M14.5 7L18.5 11" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>Edit</button></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-python"><span><span>df[</span><span>'CPUUtilization_clipped'</span><span>] = df[</span><span>'CPUUtilization'</span><span>].clip(</span><span>0</span><span>, </span><span>100</span><span>)

</span><span>def</span><span></span><span>discretize</span><span>(</span><span>util</span><span>):
    </span><span>if</span><span> util < </span><span>1</span><span>: </span><span>return</span><span></span><span>0</span><span>
    </span><span>elif</span><span> util < </span><span>50</span><span>: </span><span>return</span><span></span><span>1</span><span>
    </span><span>else</span><span>: </span><span>return</span><span></span><span>2</span><span>
</span></span></code></div></div></pre>

This:

* keeps Idle only for **exact zeros**
* spreads mid-range better
* avoids dominating by extreme outliers
* keeps all rows (no filtering)

---

## 8️⃣ **Outcome of This Approach**

✅ Markov transition matrix populated across all 3 states

✅ Emission matrix in HMM learned non-zero probabilities for more than one observed symbol

✅ Hidden states Viterbi path no longer flat

✅ Visualization now showing HiddenState movement aligned with CPU utilization

---

## 🏆 **Why this was the best approach:**

1. Preserved **Idle rows as valid state** → system idling is a meaningful state
2. Avoided artificial removal of rows
3. Removed invalid impossible values (>100%)
4. Minimal assumptions → did not fabricate data but realigned thresholds based on data distribution
5. Enabled both Markov and HMM to converge to informative matrices

---

## 🎓 **Key Learnings from the process:**

✅ Real-world logs often contain noise, errors, reporting inconsistencies → must validate

✅ Thresholds are not universal → must adapt to dataset’s empirical distribution

✅ HMM can degenerate if observations lack variability

✅ Visualization helped uncover hidden structural issues in data (collapsed emission matrix → symptom of underlying sparsity)

---

## 📝 **Summary Paragraph:**

Throughout this project, we faced several challenges working with a real-world, large-scale datacenter workload dataset. Key issues included skewed data distributions, invalid extreme values, and underpopulated intermediate states. By iteratively analyzing the dataset, testing multiple discretization thresholds, and refining data preprocessing (clipping outliers, adapting thresholds), we enabled both Markov and Hidden Markov Models to extract meaningful insights. The final solution balances preserving data integrity with improving model interpretability, showcasing a practical data science workflow for stochastic process modeling.
