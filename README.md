# PROJECT_TITLE

A fully reproducible data analysis report.

Please replace this, accordingly:
PROJECT_TITLE
YOUR_GITHUB_USERNAME
YOUR_PROJECT_NAME


---

## Investigators

- Name 1, Degree  
  Affiliation  
- Name 2, Degree  
  Affiliation  

---

## Project Repository

GitHub Repository:

https://github.com/YOUR_GITHUB_USERNAME/YOUR_PROJECT_NAME

GitHub Pages Report:

https://YOUR_GITHUB_USERNAME.github.io/YOUR_PROJECT_NAME/

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
