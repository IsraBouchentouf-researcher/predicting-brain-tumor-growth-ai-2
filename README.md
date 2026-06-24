# Phase 2: Deep Learning Transition via Specialized U-Net Architecture

**Author:** Isra Bouchentouf  
**Context:** Independent Neuro-Technology Research | Next Steps | JUN 2026  

### Objective & The Solution
To bypass the human anatomical registration issues and cross-scanner parameters (T1/T2 relaxation time mismatches) documented in Phase 1, this repository contains the implementation of a specialized U-Net Convolutional Neural Network (CNN) backbone built on PyTorch, adapted for multi-class classification.

By moving from deterministic pixel arithmetic to stochastic deep learning, the model autonomously recognizes generalized geometric and textural patterns of neuro-oncological tissues, rendering structural variances irrelevant.

### Technical Implementation & Architecture Layout

* **The Feature Extraction Backbone (`model.py`):**
  * **Contracting Path (Encoder):** Successive blocks of $3\times3$ convolutions followed by ReLU activation and $2\times2$ Max Pooling to downsample the MRI matrix, extracting high-level abstract features (tumor tissue textures) while discarding variable anatomical contours.
* **The Classification Head:**
  Instead of a standard pixel-wise decoder, the latent feature map from the deepest layer of the U-Net Encoder is flattened and passed through a fully connected **Linear Classifier Head**. This maps the deep generalized structural features directly into one of 4 clinical diagnostic categories (Glioma, Meningioma, Pituitary, or No Tumor).
* **Multi-Parametric Input Normalization (`dataset.py`):** Instead of rigid manual pixel inversion ($255-I$), input slices are normalized and adapted into multi-channel torch tensors, allowing the network to process various MRI sequences efficiently without physical scanner mismatches.
* **Loss Function Optimization (`train.py`):** Deployment of standard **Cross-Entropy Loss** to optimize the 4-class multi-label distribution, effectively driving gradient descent based on the clinical accuracy of tissue type identification.

### Core Technical Notes & Observations

* **Quantum Feature Extraction:** Convolutional kernels inherently filter out systemic artifacts caused by quantum relaxation times, focusing entirely on spatial tissue boundaries.
* **Generalization Power:** Training the architecture on multi-subject clinical repositories forces the network to generalize tumor shapes, entirely avoiding the "Cross-Subject Anatomical Variance Trap". 
* **Empirical Validation & Scalability:** The pipeline is evaluated periodically against an independent testing partition containing unseen clinical scans. Preliminary testing demonstrated strong generalization power early in the training phase, with the system showing a consistent upward trajectory in test accuracy as training data scales and optimization epochs progress.
* **Execution Environment:** Accelerated tensor computation utilizing CUDA on cloud-based GPU nodes via Google Colab.
