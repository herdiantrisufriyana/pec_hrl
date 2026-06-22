# Start with the official RStudio image
FROM rocker/rstudio:4.4.1

# Avoid user interaction with tzdata
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libcurl4-gnutls-dev \
    libxml2-dev \
    libssl-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libxt-dev \
    xorg-dev \
    libreadline-dev \
    libbz2-dev \
    liblzma-dev \
    zlib1g-dev \
    gfortran \
    software-properties-common \
    bash \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    libtiff5-dev \
    libjpeg-dev \
    cmake \
    libglpk-dev \
    && rm -rf /var/lib/apt/lists/*

# Detect architecture and download the appropriate Miniconda installer
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; \
    elif [ "$ARCH" = "aarch64" ]; then \
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    bash Miniconda3-latest-Linux-*.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-*.sh

# Add Conda to the PATH and initialize Conda globally for all users
ENV PATH="/opt/conda/bin:$PATH"
RUN /opt/conda/bin/conda init bash && \
    echo ". /opt/conda/etc/profile.d/conda.sh" > /etc/profile.d/conda.sh

# Install Python and its libraries
RUN /opt/conda/bin/conda install -y python=3.11
RUN /opt/conda/bin/conda install -y jupyterlab=4.0.11
RUN /opt/conda/bin/conda install -y pandas=2.2.2
RUN /opt/conda/bin/conda install -y scikit-learn=1.5.1
RUN /opt/conda/bin/conda install -y shap=0.42.1
RUN /opt/conda/bin/conda install -y pytorch=2.5.1 -c pytorch
RUN /opt/conda/bin/conda install -y torchvision=0.15.2 -c pytorch
RUN /opt/conda/bin/conda install -y torchaudio=2.5.1 -c pytorch

# Set base_url so ellmer can connect to host's Ollama server
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Install R packages
RUN R -e "install.packages('BiocManager', repos='http://cran.rstudio.com/')"
RUN R -e "BiocManager::install('remotes', version='3.20', ask=FALSE, update=FALSE, force=TRUE)"
RUN R -e "BiocManager::install('tidyverse', version='3.20', ask=FALSE, update=FALSE, force=TRUE)"
RUN R -e "BiocManager::install('kableExtra', version='3.20', ask=FALSE, update=FALSE, force=TRUE)"
RUN R -e "BiocManager::install('ggpubr', version='3.20', ask=FALSE, update=FALSE, force=TRUE)"
RUN R -e "BiocManager::install('igraph', version='3.20', ask=FALSE, update=FALSE, force=TRUE)"
RUN R -e "BiocManager::install('ggnetwork', version='3.20', ask=FALSE, update=FALSE, force=TRUE)"
RUN R -e "BiocManager::install('brms', version='3.20', ask=FALSE, update=FALSE, force=TRUE)"
RUN R -e "BiocManager::install('broom.mixed', version='3.20', ask=FALSE, update=FALSE, force=TRUE)"
RUN R -e "BiocManager::install('pbapply', version='3.20', ask=FALSE, update=FALSE, force=TRUE)"
RUN R -e "BiocManager::install('mice', version='3.20', ask=FALSE, update=FALSE, force=TRUE)"
RUN R -e "BiocManager::install('doParallel', version='3.20', ask=FALSE, update=FALSE, force=TRUE)"
RUN R -e "remotes::install_github('RConsortium/S7@v0.2.0')"
RUN R -e "remotes::install_github('r-lib/httr2@v1.1.0')"
RUN R -e "remotes::install_github('jeroen/curl@v6.2.0')"
RUN R -e "remotes::install_github('tidyverse/ellmer@v0.1.1')"

# Install additional system dependencies, Python libraries and R packages, chronologically


# Set the working directory to ~/project on R session start
RUN echo 'setwd("~/project")' >> /home/rstudio/.Rprofile

# Reset DEBIAN_FRONTEND variable
ENV DEBIAN_FRONTEND=
