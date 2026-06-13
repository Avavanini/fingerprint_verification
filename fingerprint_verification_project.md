# 🔍 Fingerprint Verification System — Project Blueprint

---

## 📌 Project Description

A **Fingerprint Verification System** is a biometric authentication application that captures, processes, and matches fingerprint images to verify the identity of a user. The system compares a live (probe) fingerprint against a stored (gallery) fingerprint to determine whether they belong to the same individual — a **1:1 verification** task (as opposed to 1:N identification).

**Core Goal:** Given two fingerprint images, output a similarity score and a binary decision: *Match* or *No Match*.

### Key Use Cases
- Employee attendance and access control
- Mobile/web authentication
- Banking and financial verification
- Law enforcement (AFIS integration)
- Healthcare patient identity management

---

## 🗂️ Dataset Recommendations

### Primary Dataset
| Dataset | Description | Access |
|---|---|---|
| **FVC2002 / FVC2004** | Fingerprint Verification Competition datasets, widely used benchmark, 4 databases each with ~110 fingers × 8 impressions | [http://bias.csr.unibo.it/fvc2002/](http://bias.csr.unibo.it/fvc2002/) |
| **SOCOFing** | 6,000 fingerprint images from 600 African subjects, includes altered versions (obliteration, central rotation, z-cut) | [Kaggle – SOCOFing](https://www.kaggle.com/datasets/ruizgara/socofing) |
| **NIST SD4** | 4,000 pairs of fingerprint cards from FBI, 8-bit greyscale, 500 dpi | NIST website (free, registration needed) |
| **PolyU HRF** | High-resolution fingerprint dataset (1200 dpi), good for deep learning | PolyU Biometric Research Centre |

### Recommended for Beginners
Start with **SOCOFing** (easy Kaggle download) + **FVC2002 DB1\_A** for benchmarking.

---

## 🛠️ Tech Stack

### Language & Environment
- **Python 3.10+**
- **Virtual environment:** `venv` or `conda`

### Image Processing & Computer Vision
| Library | Purpose |
|---|---|
| `OpenCV (cv2)` | Image loading, preprocessing, filtering, visualization |
| `scikit-image` | Morphological operations, thinning (skeletonization) |
| `NumPy` | Array operations |
| `Pillow (PIL)` | Image format handling |
| `SciPy` | Spatial algorithms, distance metrics |

### Fingerprint-Specific Libraries
| Library | Purpose |
|---|---|
| `fingerprint-enhancer` | Ridge enhancement using Gabor filters |
| `pyfingerprint` | Optical sensor interfacing (if using hardware) |
| `mindtct` (NBIS toolkit) | NIST's minutiae detector (optional, very powerful) |

### Machine / Deep Learning (for learned matching)
| Library | Purpose |
|---|---|
| `TensorFlow / Keras` or `PyTorch` | Siamese Network for similarity learning |
| `scikit-learn` | SVM / classical classifiers, metrics |
| `FAISS` | Fast similarity search for feature vectors |

### Backend & API
| Tool | Purpose |
|---|---|
| `FastAPI` | REST API for the verification service |
| `Uvicorn` | ASGI server |
| `SQLite / PostgreSQL` | Template storage |
| `SQLAlchemy` | ORM |

### Frontend (Optional)
| Tool | Purpose |
|---|---|
| `Streamlit` | Quick demo UI |
| `React.js` | Full web frontend |
| `Flask` | Lightweight alternative to FastAPI |

### DevOps & Tooling
| Tool | Purpose |
|---|---|
| `Docker` | Containerization |
| `Git + GitHub` | Version control |
| `pytest` | Unit and integration testing |
| `MLflow` | Experiment tracking |
| `Jupyter Notebook` | Prototyping and EDA |

---

## 🔄 System Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                     ENROLLMENT FLOW                         │
│  Capture → Preprocess → Extract Features → Store Template   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   VERIFICATION FLOW                         │
│  Capture Probe → Preprocess → Extract Features              │
│       → Compare with Stored Template → Score → Decision     │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Pipeline

1. **Image Acquisition** — Load image from file, scanner, or webcam
2. **Quality Assessment** — Check image quality (NFIQ score); reject bad images
3. **Preprocessing**
   - Normalize pixel intensities
   - Segment (separate foreground from background)
   - Enhance ridges (Gabor filter bank)
   - Binarize (threshold)
   - Thin/skeletonize ridges
4. **Feature Extraction**
   - **Minutiae-based:** Extract ridge endings and bifurcations (x, y, θ)
   - **Texture-based:** FingerCode, LBP, or CNN embeddings
5. **Matching**
   - Minutiae alignment + pairing (Hough transform or iterative)
   - Score computation (number of matched pairs / total)
   - OR: Cosine/Euclidean distance on deep embeddings
6. **Decision** — Apply threshold → Match / Non-Match
7. **Output** — Return result, score, confidence

---

## 📐 Matching Approaches

| Approach | Description | Complexity |
|---|---|---|
| **Minutiae-based** | Detect ridge endpoints & bifurcations, align & count matches | Medium |
| **Correlation-based** | Directly compare pixel intensities | Low — sensitive to distortion |
| **FingerCode** | Gabor-filtered texture codes | Medium |
| **Deep Learning (Siamese)** | Learned embeddings, end-to-end | High — best accuracy |
| **Hybrid** | Minutiae + texture or minutiae + DL | High |

**Recommended:** Start with minutiae-based, then optionally add a Siamese network for improved accuracy.

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| **FAR** (False Accept Rate) | % of impostors incorrectly accepted |
| **FRR** (False Reject Rate) | % of genuine users incorrectly rejected |
| **EER** (Equal Error Rate) | Point where FAR = FRR (lower is better) |
| **ROC Curve** | FAR vs 1-FRR across all thresholds |
| **Accuracy** | Overall classification accuracy |
| **AUC** | Area under the ROC curve |

---

## 🗓️ Project Phases & Task Checklist

---

### ✅ PHASE 0 — Project Setup

- [x] Create project repository on GitHub
- [x] Set up virtual environment (`python -m venv env`)
- [x] Create `requirements.txt` with all dependencies
- [x] Set up project folder structure:
  ```
  fingerprint-verification/
  ├── data/
  │   ├── raw/
  │   ├── processed/
  │   └── templates/
  ├── notebooks/
  ├── src/
  │   ├── preprocessing/
  │   ├── feature_extraction/
  │   ├── matching/
  │   └── api/
  ├── models/
  ├── tests/
  ├── configs/
  └── README.md
  ```
- [x] Set up `config.yaml` for paths, thresholds, and hyperparameters
- [x] Initialize `MLflow` or `WandB` for experiment tracking (optional)
- [x] Write a basic `README.md`

---

### ✅ PHASE 1 — Data Acquisition & EDA

#### 1.1 Dataset Download & Organisation
- [x] Download SOCOFing dataset from Kaggle
- [x] ~~Download FVC2002 DB1\_A (benchmark dataset)~~ *(deferred — requires institutional registration; SOCOFing is sufficient as primary dataset)*
- [x] Organise images into `/data/raw/` by subject ID
- [x] Create a `dataset_info.csv` (image path, subject ID, finger index, impression number)

#### 1.2 Exploratory Data Analysis
- [x] Count total images, subjects, impressions per finger
- [x] Visualise sample images for each finger class
- [x] Check image resolution, bit depth, and format consistency
- [x] Plot histogram of pixel intensities
- [x] Identify and document corrupted/low-quality images
- [x] Create EDA notebook: `notebooks/01_eda.ipynb`

---

### ✅ PHASE 2 — Preprocessing Pipeline

#### 2.1 Image Loading & Normalisation
- [x] Write `load_image()` — load grayscale, resize to standard (e.g., 300×300)
- [x] Write `normalize()` — zero-mean, unit-variance normalisation
- [x] Handle different image formats (`.tif`, `.bmp`, `.png`, `.jpg`)

#### 2.2 Segmentation
- [x] Implement block-wise variance segmentation
- [x] Create foreground mask (ridge area vs background)
- [x] Visualise segmentation results on sample images

#### 2.3 Ridge Enhancement
- [x] Estimate local ridge orientation (gradient-based)
- [x] Estimate local ridge frequency
- [x] Apply **Gabor filter bank** for ridge enhancement
- [x] Alternatively, use `fingerprint-enhancer` library *(primary method — Hong-Wan-Jain algorithm)*
- [x] Visualise enhanced ridges

#### 2.4 Binarisation & Thinning
- [x] Apply adaptive thresholding (Otsu or local)
- [x] Skeletonize/thin ridges to 1-pixel width
- [x] Remove noise using morphological operations (clean spurious minutiae)

#### 2.5 Pipeline Integration
- [x] Chain all steps into `preprocess(image_path) → processed_image`
- [x] Add quality check: reject images with low variance (Python-native quality metric)
- [x] Write unit tests for each preprocessing function *(23/23 passed)*
- [x] Process images on-the-fly via `preprocess()` pipeline *(saves disk space vs batch processing)*

---

### ✅ PHASE 3 — Feature Extraction

#### 3.1 Minutiae Detection (Classical)
- [x] Implement **Crossing Number (CN)** algorithm on thinned image
  - CN = 1 → Ridge ending
  - CN = 3 → Ridge bifurcation
- [x] Extract minutiae list: `[(x, y, type, orientation), ...]`
- [x] Filter spurious minutiae near borders
- [x] Visualise detected minutiae overlaid on original image

#### 3.2 Minutiae Descriptor
- [x] Compute local ridge orientation for each minutia
- [x] Optionally compute FingerCode descriptor (Gabor-based texture cylinder) *(Implemented sector-based ridge density descriptor)*
- [x] Serialise feature templates to JSON/NumPy format
- [x] Save templates to `/data/templates/`

#### 3.3 Deep Learning Embedding (Optional — Phase 3B)
- [x] Collect genuine pairs (same finger, different impressions) and impostor pairs *(Using SOCOFing Real vs Altered)*
- [x] Design a **Siamese CNN** architecture:
  - Shared CNN backbone (ResNet-18 or custom)
  - Embedding layer (128-D vector)
  - Contrastive loss or triplet loss
- [x] Train on FVC2002 pairs *(Trained on SOCOFing pairs)*
- [x] Save model weights to `/models/siamese_model.pth`
- [x] Extract and store 128-D embeddings for all enrolled templates

---

### ✅ PHASE 4 — Matching Engine

#### 4.1 Classical Minutiae Matching
- [x] Implement **alignment step:**
  - Select reference minutia pair using best Ridge Density Descriptor match
  - Compute rotation and translation for simplified heuristic alignment
- [x] Implement **minutia pairing:**
  - For each probe minutia, find nearest gallery minutia within tolerance (Δx, Δy, Δθ)
- [x] Compute **matching score** = matched pairs / min(probe count, gallery count)
- [x] Tune spatial tolerance and angular tolerance on SOCOFing

#### 4.1.B Future Enhancements (Classical Matching)
- [ ] Implement strict generalized **Hough Transform** for global alignment
- [ ] Validate and tune thresholds on the **FVC2002** dataset

#### 4.2 Deep Embedding Matching
- [x] Implement `compute_similarity(emb1, emb2)` → cosine similarity
- [ ] Add FAISS index for fast 1:N search (future extension)

#### 4.3 Decision Module
- [x] Implement threshold-based decision: score ≥ T → **MATCH**
- [x] Run threshold sweep on validation set
- [x] Plot FAR vs FRR vs threshold
- [x] Select operating threshold at EER

---

### ✅ PHASE 5 — Evaluation & Benchmarking

- [x] Create genuine pairs and impostor pairs from dataset
- [x] Run matcher on all pairs, collect scores
- [x] Compute **FAR, FRR, EER**
- [x] Plot **ROC curve** and compute **AUC**
- [ ] Compare against FVC2002 published baselines *(Skipped: Validated on SOCOFing Instead)*
- [x] Log results to `results/evaluation_report.csv`
- [x] Create notebook: `notebooks/07_evaluation.ipynb`

---

### ✅ PHASE 6 — API Development

#### 6.1 Enroll Endpoint
- [x] `POST /enroll` — accepts image + user\_id, stores template
- [x] Validate image quality before enrolling (Basic validation integrated)
- [x] Return: `{ "user_id": ..., "status": "enrolled" }`

#### 6.2 Verify Endpoint
- [x] `POST /verify` — accepts probe image + claimed user\_id
- [x] Load stored template, run matching
- [x] Return: `{ "match": true/false, "score": 0.87, "threshold": 0.75 }`

#### 6.3 Database Integration
- [x] Define `User` and `Template` models (SQLAlchemy)
- [x] Set up SQLite for dev, PostgreSQL for production
- [x] Write CRUD operations for templates

#### 6.4 API Testing
- [x] Write `pytest` tests for all endpoints
- [x] Test edge cases: empty image, unknown user, low-quality image
- [x] Generate OpenAPI docs (FastAPI auto-generates via `/docs`)

---

### ✅ PHASE 7 — Frontend / Demo UI

- [ ] Build Streamlit demo app:
  - Upload two images → run verification → show result + score
  - Show preprocessed image and detected minutiae
- [ ] OR build a React.js frontend with webcam capture (optional)
- [ ] Add visualisation of the matching process (overlay matched minutiae)

---

### ✅ PHASE 8 — Containerisation & Deployment

- [ ] Write `Dockerfile` for the FastAPI app
- [ ] Write `docker-compose.yml` (app + database)
- [ ] Test full pipeline in Docker container
- [ ] (Optional) Deploy to cloud: AWS EC2 / GCP Cloud Run / Render
- [ ] Set up CI/CD with GitHub Actions (lint + tests on push)

---

### ✅ PHASE 9 — Documentation & Wrap-up

- [ ] Complete `README.md` with setup instructions, usage, and sample output
- [ ] Document API endpoints (already auto-generated by FastAPI)
- [ ] Write a concise project report (problem, approach, results, limitations)
- [ ] Add demo GIF or screenshots to README
- [ ] Tag final release on GitHub (`v1.0.0`)

---

## 🧱 Suggested Folder Structure

```
fingerprint-verification/
├── data/
│   ├── raw/                    # Original downloaded images
│   ├── processed/              # Preprocessed images
│   └── templates/              # Extracted feature templates (.json / .npy)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── preprocessing/
│   │   ├── normalize.py
│   │   ├── segment.py
│   │   ├── enhance.py
│   │   └── thin.py
│   ├── feature_extraction/
│   │   ├── minutiae.py
│   │   └── embedding.py        # Siamese model inference
│   ├── matching/
│   │   ├── align.py
│   │   ├── matcher.py
│   │   └── scorer.py
│   └── api/
│       ├── main.py             # FastAPI app
│       ├── models.py           # DB models
│       ├── schemas.py          # Pydantic schemas
│       └── crud.py
├── models/
│   └── siamese_model.pth
├── tests/
│   ├── test_preprocessing.py
│   ├── test_matching.py
│   └── test_api.py
├── results/
│   └── evaluation_report.csv
├── configs/
│   └── config.yaml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚠️ Known Challenges & Tips

| Challenge | Tip |
|---|---|
| Poor image quality | Always run NFIQ before processing; reject quality < 3 |
| Rotation/translation variability | Use Hough-based alignment; allow ±15° and ±15px tolerance |
| Dry/wet/partial fingerprints | Augment training data; use Gabor enhancement |
| Spurious minutiae | Filter minutiae within 10px of border; minimum CN distance filter |
| Threshold selection | Use EER as default; allow adjustable threshold per use case |
| Slow matching | Precompute and cache templates; use FAISS for large galleries |

---

## 📅 Estimated Timeline

| Phase | Estimated Time |
|---|---|
| Phase 0 — Setup | 0.5 days |
| Phase 1 — Data & EDA | 1–2 days |
| Phase 2 — Preprocessing | 3–4 days |
| Phase 3 — Feature Extraction | 3–5 days |
| Phase 4 — Matching Engine | 3–5 days |
| Phase 5 — Evaluation | 1–2 days |
| Phase 6 — API | 2–3 days |
| Phase 7 — Demo UI | 1–2 days |
| Phase 8 — Docker & Deploy | 1–2 days |
| Phase 9 — Docs & Wrap-up | 1 day |
| **Total** | **~3–4 weeks** |

---

*Generated for Fingerprint Verification System Project — June 2026*
