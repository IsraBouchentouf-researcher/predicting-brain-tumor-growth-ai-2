# Phase 2: Deep Learning Transition via Specialized U-Net Architecture

**Author:** Isra Bouchentouf  
**Context:** Independent Neuro-Technology Research | Next Steps  | JUN 2026

## Objective & The Solution
To bypass the human anatomical registration issues and cross-scanner parameters (T1/T2 relaxation time mismatches) documented in Phase 1, this repository contains the implementation of a specialized **U-Net Convolutional Neural Network (CNN) built on PyTorch**. 

By moving from deterministic pixel arithmetic to stochastic deep learning, the model autonomously recognizes generalized geometric and textural patterns of neuro-oncological tissues, rendering structural variances irrelevant.

## Technical Implementation & Architecture Layout

1. **The Encoder-Decoder Backbone (`model.py`):**
   * **Contracting Path (Encoder):** Successive blocks of $3\times3$ convolutions followed by ReLU activation and $2\times2$ Max Pooling to downsample the MRI matrix, extracting high-level abstract features (tumor tissue textures) while discarding variable anatomical contours.
   * **Expanding Path (Decoder):** Up-sampling layers combined with **Skip Connections** that concatenate high-resolution features from the encoder. This preserves the precise spatial and geometric boundaries of the oncological mass.

2. **Multi-Parametric Input Normalization (`dataset.py`):**
   Instead of rigid manual pixel inversion ($255 - I$), input slices are normalized using multi-channel tensor formatting, allowing the network to interpret T1-weighted, T2-weighted, or Fluid-Attenuated Inversion Recovery (FLAIR) sequences simultaneously without physical scanner mismatches.

3. **Loss Function Optimization (`train.py`):**
   Deployment of a combined **Dice Loss** and **Binary Cross-Entropy (BCE)** loss function. This handles the critical class imbalance problem, as tumor pixels occupy a statistically small fraction of the total $512\times512$ brain matrix.

## Core Technical Notes & Observations
* **Quantum Feature Extraction:** Convolutional kernels inherently filter out systemic artifacts caused by quantum relaxation times, focusing entirely on spatial tissue boundaries.
* **Generalization Power:** Training the architecture on multi-subject clinical repositories (e.g., BraTS dataset) forces the network to generalize tumor shapes, entirely avoiding the "Cross-Subject Anatomical Variance Trap".
* **Execution Environment:** Accelerated tensor computation utilizing CUDA on cloud-based GPU nodes via Google Colab.
