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

## Project Repository

GitHub Repository:

https://github.com/herdiantrisufriyana/pec_hrl

GitHub Pages Report:

https://herdiantrisufriyana.github.io/pec_hrl/

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
https://github.com/YOUR_GITHUB_USERNAME/YOUR_PROJECT_NAME.git
```

Open the cloned folder.

---

# 3. Build and Run the Environment

In the project root folder, run:

```bash
docker compose up -d --build
```

Docker automatically uses the repository (folder) name as YOUR_PROJECT_NAME.
Each new project will automatically have its own image using YOUR_PROJECT_NAME and containers using YOUR_PROJECT_NAME_rstudio and YOUR_PROJECT_NAME_jupyter names.

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

# 5. Stop the Environment

```bash
docker compose down
```

---

# 6. Rebuild After Adding New Packages

If the Dockerfile is modified (e.g., new R or Python packages):

```bash
docker compose up -d --build
```

---

# Project Structure

- `/data` — R-specific data (.rds))
- `/R` — R utility functions
- `/inst/extdata` — Other non-rds files
- `utils.py` — Python utilities
- `index.R` or `index.ipynb` — Main analysis entry point

---

# Project Rules

- Do NOT commit large datasets (>25 MB).
- Only push code and small metadata files.
- Ignore privacy-sensitive in .gitignore.
- Always rebuild the image when adding new system dependencies.

---

This repository serves as both a reproducible analysis environment
and a transparent research report.
