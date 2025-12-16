# MLOps Review Analyzer

**An On-Demand System for E-Commerce Review Analysis and Topic Classification**

Authors: Eda Küçükkara (150210325), Revna Altınöz (150220756)  
Course: YZV448E - MLOps

---

## 📋 Table of Contents

1. [Problem Definition](#problem-definition)
2. [System Architecture](#system-architecture)
3. [MLOps Components](#mlops-components)
4. [Installation & Setup](#installation--setup)
5. [Monitoring & Observability](#monitoring--observability)
6. [Project Structure](#project-structure)
7. [Design Decisions & Trade-offs](#design-decisions--trade-offs)

---

## 🎯 Problem Definition

### What We Solve
E-commerce product pages feature thousands of customer reviews. Manual reading is impractical and leads to decision fatigue. Our system accepts a product ID (ASIN) and provides an on-demand aggregated summary of main topics discussed in reviews.

### Why It's Important
- **User Experience**: Reduces information overload
- **Business Impact**: Improves conversion rates
- **Seller Insights**: Rapid overview of customer feedback

### Stakeholders
- **End-Users**: Receive instant product insights
- **Platforms/Sellers**: Improved UX and actionable feedback
- **MLOps Engineers**: System reliability and maintainability

---

## 🏗️ System Architecture

```
User Input (ASIN)
      ↓
Streamlit UI (streamlit_app.py) / FastAPI Backend (app.py)
      ↓
DVC-versioned Dataset (701K reviews)
      ↓
Zero-Shot Classifier (DeBERTa)
      ↓
Aggregated Analysis & Visualizations
```

### Components
- **Data**: 701K Amazon Beauty Reviews (DVC + Git LFS)
- **Model**: DeBERTa Zero-Shot Classification
- **Backend**: FastAPI (optional, for Docker mode)
- **Frontend**: Streamlit
- **Deployment**: Streamlit Cloud / Docker Compose

---

## 🔧 MLOps Components

### 1. Data Version Control (DVC)
```bash
dvc pull  # Pull versioned dataset from Google Drive
```

### 2. Model Caching & Optimization
- `torch.inference_mode()`: 20-30% speedup
- Batch processing: `batch_size=16`
- Streamlit caching: Model loaded once, results cached

### 3. Monitoring
- Latency logging for every inference
- HITL feedback buttons (✅ Verified / ❌ Inaccurate)
- Drift detection: `python drift_detector.py`

### 4. Performance Testing
```bash
python test_performance.py
```

---

## 🚀 Installation & Setup

### Option 1: Streamlit Cloud (Simplest)

**Live Demo**: https://mlops-review-analyzer-idgpkjjlmvgmlxmtnducy5.streamlit.app

```bash
git clone https://github.com/edakucukkara/mlops-review-analyzer.git
cd mlops-review-analyzer
pip install -r requirements.txt
git lfs pull  # Data automatically available
streamlit run streamlit_app.py
```

### Option 2: Docker Compose (Full Stack)

```bash
git clone https://github.com/edakucukkara/mlops-review-analyzer.git
cd mlops-review-analyzer
dvc pull  # or git lfs pull
docker-compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:8501

---

## 📊 Monitoring & Observability

### Performance Metrics
- **First request**: 76-327s (model loading)
- **Cached requests**: 10-12ms (27,000x faster)

### HITL Feedback
```
HITL_LOG: 2025-12-16T16:25:32, ASIN: B00YQ6X8EO, Feedback: POSITIVE
```

### Drift Detection
```bash
python drift_detector.py
# Output: KS Test p-value, drift alerts
```

---

## 📁 Project Structure

```
mlops-review-analyzer/
├── data/
│   ├── gold_reviews.parquet       # Dataset (Git LFS)
│   └── gold_reviews.parquet.dvc   # DVC metadata
├── .streamlit/config.toml          # Streamlit configuration
├── app.py                          # FastAPI backend
├── ui.py                           # Streamlit UI (Docker mode)
├── streamlit_app.py                # Standalone app (Cloud mode)
├── drift_detector.py               # Drift monitoring
├── test_performance.py             # Performance tests
├── docker-compose.yml              # Container orchestration
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
```

---

## ⚖️ Design Decisions & Trade-offs

### Model: Zero-Shot Classification
- ✅ No training needed, flexible topics
- ❌ Slower than fine-tuned models
- **Justification**: Business flexibility > speed

### Caching Strategy
- ✅ 27,000x speedup for popular products
- ❌ First request slow
- **Justification**: Most users query popular items

### Deployment: Dual Mode
- **Cloud**: Streamlit Cloud (simple, auto-scale)
- **Local**: Docker Compose (full control)

---

## 🛡️ Ethical Considerations

- **Data**: Public Amazon reviews (no PII)
- **Bias**: Zero-shot reduces language bias
- **Transparency**: Statistical summaries (no hallucinations)
- **Security**: No external API calls, no user data stored

---

## 👥 Team Contributions

**Eda Küçükkara**: Data pipeline, model optimization, cloud deployment, UI/UX  
**Revna Altınöz**: FastAPI backend, Docker, drift detection

---

## 📚 References

1. Blei et al. (2003). *Latent Dirichlet Allocation*
2. Grootendorst (2022). *BERTopic*
3. Huyen (2022). *Designing Machine Learning Systems*
4. Yin et al. (2019). *Zero-shot Text Classification*

---

## 🐛 Troubleshooting

**Streamlit Cloud health check errors**:
- Temporary during model loading (2-3 min)
- Reboot app if persists

**Data not found**:
```bash
git lfs install && git lfs pull
```

**Docker connection refused**:
```bash
docker-compose ps
docker-compose logs
```

---

**Live Demo**: https://mlops-review-analyzer-idgpkjjlmvgmlxmtnducy5.streamlit.app  
**GitHub**: https://github.com/edakucukkara/mlops-review-analyzer
