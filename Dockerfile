# syntax=docker/dockerfile:1.4
# ==============================================================================
# CellAtria Docker Image
# ==============================================================================
# This Dockerfile creates the CellAtria application on top of the base image
# with all dependencies pre-installed.
# ==============================================================================

# Build from the cellatria-base image
# For local builds: docker build -f Dockerfile.base -t cellatria-base:latest .
# For CI/CD builds: The base image is pulled from ghcr.io/astrazeneca/cellatria:base
# To use a specific base image: docker build --build-arg BASE_IMAGE=ghcr.io/astrazeneca/cellatria:base -t cellatria:latest .
ARG BASE_IMAGE=cellatria-base:latest
FROM ${BASE_IMAGE}

# ---- Build-time arguments with safe defaults ----
ARG IMAGE_REPO="AstraZeneca/cellatria"
ARG IMAGE_TAG="v1.0.0"
ARG IMAGE_VERSION="v1.0.0"
ARG VCS_REF="unknown"
ARG BUILD_DATE="1970-01-01T00:00:00Z"
ARG TREE_STATE="clean"

# ---- OCI-compliant image metadata ----
LABEL org.opencontainers.image.title="CellAtria"
LABEL org.opencontainers.image.description="CellAtria: Agentic Triage of Regulated single-cell data Ingestion and Analysis."
LABEL org.opencontainers.image.authors="Nima Nouri <nima.nouri@astrazeneca.com>"
LABEL org.opencontainers.image.version="${IMAGE_VERSION}"
LABEL org.opencontainers.image.revision="${VCS_REF}" 
LABEL org.opencontainers.image.created="${BUILD_DATE}" 
LABEL org.opencontainers.image.source="https://github.com/${IMAGE_REPO}"
LABEL org.opencontainers.image.url="https://github.com/${IMAGE_REPO}"
LABEL org.opencontainers.image.ref.name="ghcr.io/${IMAGE_REPO}:${IMAGE_TAG}" 
LABEL org.opencontainers.image.tree_state="${TREE_STATE}"

# ---- Propagate to runtime ENV for in-container inspection ----
ENV IMAGE_REPO="${IMAGE_REPO}"
ENV IMAGE_TAG="${IMAGE_TAG}"
ENV IMAGE_VERSION="${IMAGE_VERSION}"
ENV IMAGE_VCS_REF="${VCS_REF}"
ENV IMAGE_BUILD_DATE="${BUILD_DATE}"
ENV IMAGE_TREE_STATE="${TREE_STATE}"

# ---- Embed a JSON manifest inside the image ----
RUN mkdir -p /usr/local/share/cellatria && \
    printf '{\n  "image_repo": "%s",\n  "image_tag": "%s",\n  "image_version": "%s",\n  "vcs_ref": "%s",\n  "build_date": "%s",\n  "tree_state": "%s"\n}\n' \
      "$IMAGE_REPO" "$IMAGE_TAG" "$IMAGE_VERSION" "$VCS_REF" "$BUILD_DATE" "$TREE_STATE" \
    > /usr/local/share/cellatria/image-meta.json

ENV IMAGE_META_PATH=/usr/local/share/cellatria/image-meta.json

# -----------------------------------
# Copy the CellAtria application files into the container
# and set up the working directory structure
# -----------------------------------
# Copy all files into Docker
RUN mkdir -p /opt/cellatria
WORKDIR /opt/cellatria
COPY . /opt/cellatria/
# -----------------------------------
# Make cellatria CLI callable via `cellatria`
RUN chmod +x /opt/cellatria/agent/chatbot.py
RUN ln /opt/cellatria/agent/chatbot.py /usr/local/bin/cellatria
# -----------------------------------
# Make cellexpress CLI callable via `cellexpress`
RUN chmod +x /opt/cellatria/cellexpress/main.py
RUN ln -s /opt/cellatria/cellexpress/main.py /usr/local/bin/cellexpress
# -----------------------------------
# The VOLUME instruction and the -v option to docker run 
VOLUME /data
WORKDIR /data
# -----------------------------------
# Expose the port used by Gradio
EXPOSE 7860
# -----------------------------------
# Configure Python paths and data locations
ENV PYTHONPATH=/opt/cellatria/agent
ENV ENV_PATH=/data
# Disable Python output buffering for real-time logs in Docker
ENV PYTHONUNBUFFERED=1
# -----------------------------------
# Default command launches the CellAtria chatbot interface
CMD ["/usr/local/bin/cellatria"]
# -----------------------------------