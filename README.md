# Modeling distinct pathogenesis of early- and late-onset preeclampsia via hierarchical representation learning of interactome

A fully reproducible data analysis report.


---

## Investigators

- Herdiantri Sufriyana, MD, PhD  
  Institute of Biomedical Informatics, College of Medicine, National Yang Ming Chiao Tung University, Taipei, Taiwan  
- Yu-Wei Wu, PhD  
  Graduate Institute of Biomedical Informatics, College of Medical Science and Technology, Taipei Medical University, Taipei, Taiwan  
- Hua-Sheng Chiu, PhD  
  Texas Children's Cancer Center, Baylor College of Medicine, Houston, TX, USA  
- Emily Chia-Yu Su, PhD  
  Institute of Biomedical Informatics, College of Medicine, National Yang Ming Chiao Tung University, Taipei, Taiwan  

---

# Reproducibility Instructions

This project is fully reproducible using Docker.

You do NOT need to install R, RStudio, or Python manually.
Everything runs inside Docker.

---

# 1. First-Time Setup (Do Once Only Per Machine)

## Step 1 — Install Git

Download:

https://git-scm.com/downloads

After installation, verify in terminal:

```bash
git --version
```

---

## Step 2 — Install Docker Desktop

Download:

https://docs.docker.com/get-started/get-docker/

After installation:
- Open Docker Desktop
- Make sure it is running

---

## Step 3 — Install VS Code (Recommended)

Download:

https://code.visualstudio.com/download

---

# 2. Clone This Repository

Open VS Code.

Click:

**Clone Git Repository**

Then paste:

```
https://github.com/herdiantrisufriyana/pec_hrl.git
```

Open the cloned folder.

---

# 3. Build and Run the Environment

In the project root folder, run:

```bash
docker compose up -d --build
```

First build may take several minutes.

---

# 4. Access the Applications

## RStudio (No Login Required)

Open:

http://localhost:8787

Working directory:

```
~/project
```

---

## JupyterLab (No Token Required)

Open:

http://localhost:8888

---

# 5. Run the Analysis

In RStudio, open `index.Rmd` and click **Knit**, or run individual chunks interactively.

By default, `comp_req` is set to `c()` (empty), which reads from saved intermediate files for a fast run. To recompute from scratch, set `comp_req=c(1,2,3,4)` in the first chunk.

---

# 6. Stop the Environment

```bash
docker compose down
```

---

# 7. Rebuild After Adding New Packages

If the Dockerfile is modified (e.g., new R or Python packages):

```bash
docker compose up -d --build
```

---

# Project Structure

- `index.Rmd` — Main analysis vignette
- `revision.Rmd`, `revision2.Rmd`, `revision3.Rmd` — Revision-specific analyses
- `pec_R/` — R utility functions for feature map construction
- `R/` — R utility functions for figures and tables
- `data/` — Intermediate data (TidySets, grad-CAM, history, ontology)
- `pec_real_data/` — Validated model data (weights, evaluation results)
- `ablation/` — Ablation study scripts and results (t-SNE vs PCA vs UMAP)
- `inst/extdata/` — Model registry and other metadata
- `utils.py` — Python utilities
- `Dockerfile`, `docker-compose.yml` — Reproducible environment

---

# Project Rules

- Do NOT commit large datasets (>25 MB).
- Only push code and small metadata files.
- Ignore privacy-sensitive files in .gitignore.
- Always rebuild the image when adding new system dependencies.

---

This repository serves as both a reproducible analysis environment
and a transparent research report.
