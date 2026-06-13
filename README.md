# 🔍 Fingerprint Verification System

A biometric authentication system that captures, processes, and matches fingerprint images to verify user identity. The system performs **1:1 verification** — comparing a live (probe) fingerprint against a stored (gallery) template to determine whether they belong to the same individual.

---

## 🎯 Core Goal

> Given two fingerprint images, output a similarity score and a binary decision: **Match** or **No Match**.

---

## ✨ Features

- **Complete Preprocessing Pipeline** — Normalization, segmentation, Gabor enhancement, binarization, and skeletonization
- **Minutiae-Based Feature Extraction** — Ridge ending and bifurcation detection using the Crossing Number algorithm
- **Deep Learning Matching** — Optional Siamese CNN for learned similarity embeddings
- **REST API** — FastAPI-powered enrollment and verification endpoints
- **Interactive Demo** — Streamlit UI for quick fingerprint comparison
- **Comprehensive Evaluation** — FAR, FRR, EER, ROC curve, and AUC metrics

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.10+ |
| **CV / Image Processing** | OpenCV, scikit-image, NumPy, Pillow, SciPy |
| **Fingerprint** | fingerprint-enhancer, Crossing Number algorithm |
| **Machine Learning** | PyTorch, scikit-learn, FAISS |
| **Backend** | FastAPI, Uvicorn, SQLAlchemy, SQLite |
| **Frontend** | Streamlit |
| **DevOps** | Docker, GitHub Actions, pytest |

---

## 📁 Project Structure

```
fingerprint-verification/
├── data/
│   ├── raw/                    # Original downloaded images
│   ├── processed/              # Preprocessed images
│   └── templates/              # Extracted feature templates
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── preprocessing/          # Image preprocessing modules
│   ├── feature_extraction/     # Minutiae & embedding extraction
│   ├── matching/               # Alignment, matching, scoring
│   └── api/                    # FastAPI application
├── models/                     # Trained model weights
├── tests/                      # Unit & integration tests
├── configs/
│   └── config.yaml             # Central configuration
├── results/                    # Evaluation reports
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/fingerprint-verification.git
cd fingerprint-verification
```

### 2. Set Up Virtual Environment
```bash
python -m venv env
# Windows
env\Scripts\activate
# macOS/Linux
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure
Edit `configs/config.yaml` to set paths and parameters for your environment.

### 5. Run the API
```bash
uvicorn src.api.main:app --reload
```
Open **http://localhost:8000/docs** for interactive API documentation.

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/enroll` | Enroll a fingerprint (image + user_id → stored template) |
| `POST` | `/verify` | Verify a fingerprint against enrolled template |

---

## 📈 Evaluation Metrics

| Metric | Description |
|---|---|
| **FAR** | False Accept Rate — impostors incorrectly accepted |
| **FRR** | False Reject Rate — genuine users incorrectly rejected |
| **EER** | Equal Error Rate — where FAR = FRR (lower is better) |
| **ROC / AUC** | Receiver Operating Characteristic curve and area under it |

---

## 📦 Datasets

- **SOCOFing** — 6,000 fingerprint images from 600 subjects ([Kaggle](https://www.kaggle.com/datasets/ruizgara/socofing))
- **FVC2002** — Fingerprint Verification Competition benchmark ([Website](http://bias.csr.unibo.it/fvc2002/))

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ for biometric security research — June 2026*
