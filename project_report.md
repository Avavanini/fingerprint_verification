# Fingerprint Verification System — Project Report

## 1. Problem Statement
Biometric security systems require high accuracy and robustness against noise. Traditional minutiae-based fingerprint verification algorithms excel on clean, high-quality images but fail catastrophically when presented with heavily distorted, dry, or partially obliterated fingerprints. Conversely, purely deep-learning-based approaches can struggle with fine-grained spatial alignment. The objective of this project was to build a robust Fingerprint Verification System capable of 1:1 identity verification across varying levels of image degradation.

## 2. Technical Approach
We implemented a **Hybrid Fusion Engine** that combines two distinct matching pathways:

1. **Classical Pathway (Minutiae Matching):** 
   * A full image processing pipeline (Normalization -> Gabor Enhancement -> Binarization -> Thinning).
   * Feature extraction using the Crossing Number algorithm to detect ridge endings and bifurcations.
   * A translation-invariant alignment and bounding-box matching algorithm.
   
2. **Deep Learning Pathway (Siamese CNN):**
   * A ResNet-18 architecture fine-tuned via contrastive loss.
   * Extracts a robust 128-dimensional dense vector embedding from the raw image.
   * Computes similarity using cosine distance.

**Fusion:**
The final matching decision is made using a weighted linear combination of the two normalized scores (e.g., `α=0.6` for CNN, `(1-α)=0.4` for minutiae). This allows the deep embedding to capture global texture robustness while the classical pipeline provides precise local structural verification.

## 3. Architecture & Tech Stack
* **Backend:** FastAPI, SQLite, SQLAlchemy.
* **Frontend:** Streamlit interactive UI.
* **Machine Learning:** PyTorch, Torchvision, scikit-learn.
* **Computer Vision:** OpenCV, scikit-image.

## 4. Results & Performance
The system was evaluated on the **SOCOFing** dataset using 100 subjects against "Easy", "Medium", and "Hard" synthetically altered variations (Z-cuts, obliterations, and central rotations).

* **Easy Difficulty:** EER of ~0.001 (0.1%), AUC of 0.999.
* **Medium Difficulty:** Excellent performance, though naturally lower than Easy.
* **Hard Difficulty:** Demonstrated the limitations of fixed thresholds. Heavily degraded images suppress similarity scores across both pathways. However, the hybrid approach significantly outperformed either the classical or CNN approach independently.

## 5. Limitations & Future Scope
* **Threshold Tuning:** The current `OPERATING_THRESHOLD` is fixed. Extreme distortions require dynamic thresholding or dataset-specific tuning.
* **Speed:** Classical thinning and minutiae extraction over Python loops is a bottleneck compared to instantaneous CNN inference. Future iterations should vectorize or move the classical pipeline to C++.
* **Containerisation:** Deploying via Docker (Phase 8) was moved to future scope to focus on core algorithmic robustness.
