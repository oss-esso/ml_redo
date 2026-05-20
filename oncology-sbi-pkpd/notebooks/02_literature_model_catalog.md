# Literature/model catalogue for drug-specific tumor-growth PK/PD

This catalogue should run in parallel with the forecasting notebooks. The main modelling lesson so far is:

- richer models improve retrospective fit;
- held-out prediction remains difficult;
- adding more compartments blindly is not the right next step;
- future model upgrades should be chosen from a literature-grounded catalogue based on drug mechanism, data granularity, and identifiability.

The attached project notes emphasize the same risks: simulator gap, sparse and irregular clinical measurements, parameter identifiability, summary statistics, posterior predictive validation, and comparison against optimization/MCMC/BSL/baseline methods before trusting SBI.

---

## 1. Model family map

| Family | Best suited for | State variables | Drug-effect structure | Data needed | Main risk | Priority |
|---|---|---|---|---|---|---|
| Empirical baselines | Forecasting benchmark | none | none | SLD time series | not mechanistic | must-have |
| Simeoni TGI | cytotoxic-like delayed shrinkage | proliferating + damaged compartments | kill/damage proportional to exposure | dose + SLD | too generic across mechanisms | already implemented |
| Simeoni + observation shift | measurement-regime discontinuities | same latent model | same as Simeoni | repeated scans | can overfit if retrospective | already implemented diagnostically |
| Nested resistant-Simeoni | response then plateau/regrowth | sensitive Simeoni chain + resistant compartment | sensitive strong kill, resistant weak kill | longer follow-up | identifiability from SLD alone | validate further |
| Emax/saturable TGI | targeted agents, dose-response studies | TGI states | E(C)=Emax*C/(EC50+C) | exposure/dose variation | EC50 weakly identifiable if dose does not vary | medium/high |
| Combination TGI | multi-drug regimens | TGI states with multiple drug inputs | additive/synergy/interaction terms | separate drug histories | interaction can be unidentifiable | high for multi-drug trials |
| Carrying-capacity / anti-angiogenic | anti-VEGF / anti-angiogenic drugs | tumor + vascular support/carrying capacity | drug changes carrying capacity | drug class, biomarkers if available | hard from SLD alone | high if anti-angiogenic drugs present |
| Immune delayed-response | immune checkpoint inhibitors | tumor + immune effector/delay | delayed immune-mediated kill | ICI flag, biomarkers if available | pseudo-progression and delayed response | high if ICI trials present |
| Lesion-level hierarchical TGI | heterogeneous lesions/organs | lesion-specific states | lesion/organ random effects | individual lesion measurements | complexity, lesion tracking issues | high if lesion data exist |
| Tumor-size + survival joint model | clinical endpoint prediction | tumor dynamics + hazard | indirect treatment effect | OS/PFS/event data | endpoint confounding | later milestone |
| PDE reaction-diffusion | spatial glioma/imaging-rich problems | cell density field | therapy source/sink terms | MRI/segmentation | expensive, not SLD-first | only imaging-rich |

---

## 2. Mechanism-to-model routing

### Cytotoxic chemotherapy

Start with:

```text
Simeoni TGI
Simeoni + robust observation model
Nested resistant-Simeoni if rebound/regrowth appears
```

Drug effect:

```text
kill(t) = k_kill C(t) x_sensitive
```

Use when drug action can be approximated as direct tumor-cell damage/killing with delayed observable shrinkage.

---

### Targeted therapy

Start with:

```text
Nested resistant-Simeoni
Emax/saturable kill variant
sensitive/resistant clonal model
```

Candidate effect:

```text
E(C) = Emax C / (EC50 + C)
kill_s(t) = E(C(t)) S(t)
kill_r(t) = q_r E(C(t)) R(t)
```

Reason: targeted therapies often show strong initial response followed by acquired resistance.

---

### Anti-angiogenic therapy

Do not assume the drug only kills tumor cells. Candidate structure:

```text
dV/dt = growth(V, K) - cytotoxic_kill(C) V
dK/dt = vascular_recovery_or_growth(V, K) - k_antiangio C(t) K
```

where `K(t)` is a carrying capacity or vascular-support variable.

Use for anti-VEGF / anti-angiogenic drugs.

---

### Immune checkpoint inhibitors

Use delayed-response / immune-mediated models.

Candidate structure:

```text
dT/dt = rho T(1 - T/K) - k_E E T
dE/dt = activation(C, T, biomarkers) - delta_E E
```

Observation model should allow:

```text
delayed response
apparent early increase
pseudo-progression-like behavior
heavy-tailed noise
```

Use for ICI trials.

---

### Combination therapy

Use multiple exposure functions:

```text
C_1(t), C_2(t), ..., C_M(t)
```

Candidate total effect:

```text
E_total(t) = E_1(C_1) + E_2(C_2) + gamma E_1(C_1) E_2(C_2)
```

Interpretation:

```text
gamma > 0: supra-additive / synergistic
gamma = 0: additive on chosen effect scale
gamma < 0: antagonistic / sub-additive
```

Do not estimate interaction parameters unless the dataset contains enough treatment variation to identify them.

---

## 3. Candidate equations to implement

### 3.1 Baseline Simeoni TGI

```text
x1 = proliferating tumor
x2, x3, x4 = damaged/transit compartments
V = x1 + x2 + x3 + x4
```

```text
dx1/dt = g(x1) - k_kill C(t) x1
dx2/dt = k_kill C(t) x1 - k_tr x2
dx3/dt = k_tr (x2 - x3)
dx4/dt = k_tr (x3 - x4)
```

Use as a baseline for all studies.

---

### 3.2 Nested resistant-Simeoni

```text
s1, s2, s3, s4 = sensitive Simeoni chain
r = resistant compartment
V = s1 + s2 + s3 + s4 + r
```

```text
ds1/dt = g_s(s1) - k_kill_s C(t) s1 - k_sr s1
ds2/dt = k_kill_s C(t) s1 - k_tr s2
ds3/dt = k_tr(s2 - s3)
ds4/dt = k_tr(s3 - s4)
dr/dt  = g_r(r) - k_kill_r C(t) r + k_sr s1
```

Use when trajectories show response followed by plateau or regrowth.

---

### 3.3 Emax drug effect

Replace linear kill:

```text
k_kill C(t)
```

with:

```text
E(C) = Emax C(t) / (EC50 + C(t))
kill(t) = E(C(t)) x
```

Use only when there is enough dose/exposure variation to identify `EC50`.

---

### 3.4 Combination TGI

```text
E1(t) = f1(C1(t))
E2(t) = f2(C2(t))
Etotal(t) = E1(t) + E2(t) + gamma E1(t) E2(t)
kill(t) = Etotal(t) x
```

Use for multi-drug regimens.

---

### 3.5 Anti-angiogenic / carrying-capacity model

```text
V(t) = tumor burden
K(t) = carrying capacity / vascular support
```

Example:

```text
dV/dt = lambda V log(K/V) - kill_direct(C) V
dK/dt = a V^b - d V^(2/3) K - k_antiangio C(t) K
```

or a simpler reduced form:

```text
dV/dt = growth(V, K)
dK/dt = recovery(K) - k_antiangio C(t) K
```

Use for anti-angiogenic mechanisms.

---

### 3.6 Immune delayed-response model

```text
T(t) = tumor burden
E(t) = immune effector / immune activity
```

```text
dT/dt = rho T(1 - T/K) - k_E E T
dE/dt = activation(C, T) - delta_E E
```

Use for ICI trials and delayed-response patterns.

---

### 3.7 Lesion-level hierarchical TGI

For lesion `l` in patient `i`:

```text
dV_il/dt = model(V_il; theta_i, eta_il, organ_l, C_i(t))
eta_il ~ Normal(0, Sigma_lesion)
```

Observed SLD:

```text
SLD_i(t) = sum_l diameter_il(t)
```

Use when individual target lesion measurements are available. This can explain apparent SLD jumps caused by lesion selection or lesion-specific response heterogeneity.

---

## 4. Implementation roadmap

### Phase 1 — Forecasting baselines

Run `02_forecasting_baselines_and_rolling_origin.ipynb`.

Decision:

```text
Does any mechanistic model beat LOCF/log-linear one-step forecasting?
```

If no, improve observation/process models and drug metadata before SBI.

---

### Phase 2 — Drug-class metadata

Create a table:

```text
study_id
patient_id
drug_name
drug_class
mechanism_hint
combination_flag
line_of_therapy if available
```

This table routes patients to model families.

---

### Phase 3 — Model registry

Implement:

```python
MODEL_REGISTRY = {
    "cytotoxic": ["Simeoni", "NestedResistantSimeoni"],
    "targeted": ["NestedResistantSimeoni", "EmaxResistantTGI"],
    "anti_angiogenic": ["CarryingCapacityTGI"],
    "ICI": ["ImmuneDelayedResponseTGI"],
    "combination": ["CombinationTGI"],
}
```

---

### Phase 4 — Synthetic validation per model class

For each model family:

```text
sample parameters
simulate real scan schedules
add realistic observation noise
fit/recover or train SBI
run SBC / posterior predictive checks
```

---

### Phase 5 — Real-data posterior predictive checks

Only after synthetic validation:

```text
infer posterior samples
simulate posterior predictive trajectories
check coverage and calibration
compare against rolling-origin empirical baselines
```

---

## 5. Decision table

| Observation in data | Likely issue | Next model/action |
|---|---|---|
| smooth shrinkage | baseline TGI enough | Simeoni |
| shrinkage then rebound | resistance/regrowth | nested resistant-Simeoni |
| isolated scan jump | outlier/noise | Student-t or mixture observation model |
| later scans shifted | measurement regime shift | changepoint observation model |
| response differs by lesion/organ | lesion heterogeneity | lesion-level hierarchical model |
| delayed response after ICI | immune dynamics | immune delayed-response model |
| different drugs in combination | interaction | combination TGI |
| poor forecasting for all point predictors | high uncertainty | posterior predictive intervals, not deterministic curves |

---

## 6. Literature anchors

- Simeoni et al., “Predictive pharmacokinetic-pharmacodynamic modeling of tumor growth kinetics in xenograft models after administrations of anticancer agents,” Cancer Research, 2004.
- Magni et al., “A minimal model of tumor growth inhibition,” 2008.
- Terranova et al., “A predictive pharmacokinetic-pharmacodynamic model of tumor growth kinetics in xenograft mice after administration of anticancer agents given in combination,” 2013.
- Claret et al., “Model-based prediction of phase III overall survival in colorectal cancer on the basis of phase II tumor dynamics,” Journal of Clinical Oncology, 2009.
- Krishnan et al., “Tumor growth inhibition modeling of individual lesion dynamics and inter-organ variability in HER2-negative breast cancer patients treated with docetaxel,” 2021.
- Courlet et al., “Modeling tumor size dynamics based on real-world clinical and imaging data in advanced melanoma patients receiving immune checkpoint inhibitors,” 2023.
- Oden et al., “Selection and assessment of phenomenological models of tumor growth,” 2013.
- Whitaker et al., “Bayesian inference for stochastic differential equation mixed effects models of a tumour xenography study,” 2019.
- Ezhov et al., “Learn-Morph-Infer,” 2023.
