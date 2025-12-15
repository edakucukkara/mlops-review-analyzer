# Product Review Analyzer - MLOps Project

An on-demand system for e-commerce review analysis and topic classification using Zero-Shot Learning.

## Project Overview

This MLOps project demonstrates a complete pipeline from data ingestion to deployment, focusing on analyzing Amazon product reviews to extract actionable insights. The system uses **DeBERTa Zero-Shot Classification** to categorize reviews into predefined topics without requiring task-specific training.

### Key Features

- **Zero-Shot Topic Classification**: Analyzes reviews across 6 predefined topics (Quality, Price, Scent, Packaging, Safety, Service)
- **Real-time Inference**: On-demand analysis with optimized latency
- **Human-in-the-Loop (HITL)**: Feedback mechanism for continuous model improvement
- **Data Drift Detection**: Statistical monitoring using Kolmogorov-Smirnov test
- **Reproducible Pipeline**: DVC for data versioning, MLflow for experiment tracking
- **Containerized Deployment**: Docker-based architecture for cross-platform compatibility

---

## Architecture

```
+-------------+      +--------------+      +-------------+
|  Streamlit  |----->|   FastAPI    |----->|   DeBERTa   |
|     UI      |      |   Backend    |      |   Model     |
+-------------+      +--------------+      +-------------+
                            |
                            v
                    +--------------+
                    |  Data Store  |
                    |   (Parquet)  |
                    +--------------+
```

### Components

1. **Frontend**: Streamlit web interface for product selection and visualization
2. **Backend**: FastAPI service with inference caching and latency monitoring
3. **Model**: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (982ms avg latency, selected via MLflow experiments)
4. **Data**: Amazon Beauty Reviews dataset (701K+ reviews) versioned with DVC

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.10+ (for local development)
- Git

### Running with Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/edakucukkara/mlops-review-analyzer
cd mlops-review-analyzer

# Start the services
docker-compose up --build

# Access the application
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Pull data from DVC
dvc pull

# Start backend
uvicorn app:app --reload

# Start frontend (in another terminal)
streamlit run ui.py
```

---

## Model Selection Process

We evaluated 3 models using **MLflow** for experiment tracking:

| Model | Avg Latency (ms) | Decision |
|-------|------------------|----------|
| BART-Large | 1816 | Too slow |
| DistilBART | 1058 | Lower accuracy |
| **DeBERTa** | **982** | **Winner** (1.85x faster than baseline) |

**Rationale**: DeBERTa provides the best balance between accuracy and inference speed for real-time analysis.

See full experiments in `experiments/model_selection.ipynb`

---

## Data Pipeline

### Dataset
- **Source**: [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (All Beauty category)
- **Size**: 701,528 reviews
- **Format**: Parquet (efficient columnar storage)
- **Versioning**: DVC with Google Drive remote storage

### Preprocessing Steps
1. Download raw JSONL data from HuggingFace
2. Merge reviews with product metadata
3. Extract product images and ratings
4. Filter products with minimum 5 reviews
5. Export to Parquet format

Run the pipeline:
```bash
python data_ingestion.py
```

---

## Monitoring and Observability

### 1. Latency Tracking
Every API request logs response time:
```python
# See app.py line 163
print(f"[MONITORING] Analysis Latency for {asin}: {latency_ms:.2f} ms")
```

### 2. Human-in-the-Loop Feedback
UI includes "Verified" and "Inaccurate" buttons to collect user feedback:
```python
# See ui.py lines 171-176
log_feedback(asin, "POSITIVE")  # Logged to container output
```

### 3. Data Drift Detection
Simulated drift monitoring using statistical tests:
```bash
# Run drift detection
docker-compose exec backend python drift_detector.py
```

Output example:
```
KS Statistic: 0.0234
P-Value: 0.0012
ALERT: Data Drift Detected (p < 0.05)
```

---

## Performance Optimizations

Applied optimizations:

1. **Batch Processing**: Process 50 reviews in single forward pass with `batch_size=16`
2. **Torch Inference Mode**: Disabled gradient computation for 20-30% speedup
3. **Model Warm-up**: Pre-load weights during container startup
4. **LRU Caching**: Cache analysis results for popular products (`@lru_cache(maxsize=50)`)
5. **Environment Tuning**: Optimized threading and parallelism settings

**Performance improvement**: 40-50% reduction in inference latency compared to naive implementation.

---

## Testing Performance

### Test a Product Analysis

```bash
# Using curl
curl http://localhost:8000/analyze/B00YQ6X8EO

# Check latency in backend logs
docker-compose logs backend | grep MONITORING
```

### Run Performance Test Suite

```bash
python test_performance.py
```

### View MLflow Experiments

```bash
cd experiments
mlflow ui

# Open http://localhost:5000
```

---

## Project Structure

```
mlops-review-analyzer/
├── app.py                      # FastAPI backend
├── ui.py                       # Streamlit frontend
├── data_ingestion.py           # Data pipeline script
├── drift_detector.py           # Drift monitoring script
├── test_performance.py         # Performance testing script
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Multi-container setup
├── Dockerfile.backend          # Backend container config
├── Dockerfile.frontend         # Frontend container config
├── .dvc/                       # DVC configuration
├── data/
│   └── gold_reviews.parquet    # Processed dataset (DVC tracked)
├── experiments/
│   └── model_selection.ipynb   # MLflow experiments
└── .streamlit/
    └── config.toml             # Streamlit configuration
```

---

## Configuration

### Environment Variables

**Backend** (`docker-compose.yml`):
```yaml
environment:
  - DATA_PATH=data/gold_reviews.parquet
  - OMP_NUM_THREADS=4
  - TOKENIZERS_PARALLELISM=false
```

**Frontend** (`docker-compose.yml`):
```yaml
environment:
  - API_URL=http://backend:8000
```

### Modifying Topics

Edit the `LABELS` list in `app.py`:
```python
LABELS = [
    "Quality & Effectiveness",
    "Scent & Texture",
    "Price & Value",
    "Packaging & Shipping",
    "Safety & Authenticity",
    "Service"
]
```

---

## MLOps Best Practices Implemented

| Practice | Implementation |
|----------|----------------|
| **Data Versioning** | DVC with remote storage |
| **Experiment Tracking** | MLflow with 3 model comparisons |
| **Model Registry** | MLflow model registry (local) |
| **CI/CD** | Docker multi-stage builds |
| **Monitoring** | Latency logging + drift detection |
| **Reproducibility** | Pinned dependencies + Docker |
| **Feedback Loop** | HITL mechanism in UI |

---

## References

1. Blei et al. (2003) - Latent Dirichlet Allocation
2. Grootendorst (2022) - BERTopic
3. Huyen (2022) - Designing Machine Learning Systems
4. Yin et al. (2019) - Benchmarking Zero-shot Text Classification

---

## Authors

- Eda Kucukkara (150210325)
- Revna Altinoz (150220756)

**Course**: YZV448E - MLOps (Istanbul Technical University)

---

## License

This project is for educational purposes as part of the MLOps course curriculum.
