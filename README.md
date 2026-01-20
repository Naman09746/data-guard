# Automated Data Quality & Leakage Detection System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A **production-grade data quality and leakage detection system** for ML pipelines. This system transforms from a rule-based validator to a **meta-ML platform** that learns failure patterns.

## 🌟 Key Features

### 🔥 1. ML-Based Leakage Risk Scoring
Instead of rule-based detection ("correlation > 0.95"), get **probability-based risk scores**:
```
Feature X has an 87% leakage risk based on learned patterns
```

Features extracted per column:
- Correlation with target
- Mutual information proxy
- Stability across cross-validation splits
- Time-lag correlation behavior
- High-cardinality detection

### 🧪 2. Before vs After Impact Experiments
Prove business value by comparing:
- Model trained **WITH** leaky features (inflated metrics)
- Model trained **AFTER** removal (realistic metrics)

Generates reports showing:
- Accuracy drop
- Generalization gap improvement
- Cross-validation stability

### 📊 3. Data Versioning & Scan History
Track dataset evolution with:
- Hash-based dataset versioning
- Scan history storage
- Diff between scans
- **Regression alerts** when quality degrades

### 🚨 4. Drift → Alert → Action Loop
End-to-end ML lifecycle management:
- Drift detection with severity scoring
- Alerts with actionable recommendations
- Affected model tracking
- Retrain/revalidate suggestions

### 📚 5. Real-World Case Study
Complete churn prediction case study demonstrating:
- Leakage detection in action
- Impact experiments
- Before/after comparison

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/data-quality-leakage-detection.git
cd data-quality-leakage-detection

# Install dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.leakage_detection.leakage_engine import LeakageDetectionEngine
from src.data_quality.quality_engine import DataQualityEngine
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Data Quality Check
quality_engine = DataQualityEngine()
quality_report = quality_engine.validate(df)
print(f"Quality Score: {quality_engine.get_quality_score(df):.2f}")

# Leakage Detection
leakage_engine = LeakageDetectionEngine()
leakage_report = leakage_engine.detect(df, target_column="target")
print(f"Leakage Status: {leakage_report.status.value}")

# ML-Based Risk Scores
risk_result = leakage_engine.get_risk_scores(df, "target")
print(f"High-Risk Features: {risk_result.high_risk_features}")
```

### Running Impact Experiments

```python
from src.leakage_detection.impact_experiment import LeakageImpactExperiment

experiment = LeakageImpactExperiment(model_type="random_forest")
result = experiment.run_experiment(df, "target")

print(f"Accuracy WITH leakage: {result.metrics_with_leakage.accuracy:.1%}")
print(f"Accuracy AFTER removal: {result.metrics_after_removal.accuracy:.1%}")
print(f"Accuracy drop: {result.accuracy_drop:.1%}")
```

### Data Versioning

```python
from src.core.data_versioning import DataVersioner, ScanHistoryStore

# Version your dataset
versioner = DataVersioner()
version = versioner.create_version(df)
print(f"Dataset Version: {version.version_hash}")

# Track scan history
store = ScanHistoryStore()
history = store.get_scans(limit=10)
```

### Alert Management

```python
from src.core.alert_system import AlertManager

manager = AlertManager()

# Get open alerts
alerts = manager.get_open_alerts()
for alert in alerts:
    print(f"{alert.severity.value}: {alert.title}")

# Create drift alert
alert = manager.create_drift_alert(
    feature_name="price",
    drift_score=0.8,
    drift_type="distribution_shift"
)
```

## 🧪 Running the Case Study

```bash
cd case_studies/churn_dataset
python run_case_study.py
```

This generates:
- `raw_data.csv` - Synthetic churn dataset with intentional leakage
- `leakage_found.md` - Detection report
- `metrics_before.md` - Inflated model metrics
- `metrics_after.md` - Realistic metrics after cleanup

## 🔌 API Endpoints

Start the server:
```bash
uvicorn src.api.main:app --reload --port 8000
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/leakage/detect` | POST | Upload CSV for leakage detection |
| `/api/leakage/risk-scores` | POST | Get ML-based risk scores |
| `/api/leakage/experiments/impact` | POST | Run before/after experiment |
| `/api/leakage/history` | GET | Get scan history |
| `/api/leakage/alerts` | GET | Get alerts |
| `/api/leakage/alerts/{id}/acknowledge` | POST | Acknowledge alert |
| `/api/leakage/alerts/{id}/resolve` | POST | Resolve alert |
| `/api/quality/validate` | POST | Run quality validation |

## 🏗️ Project Structure

```
├── src/
│   ├── api/                    # FastAPI routes
│   ├── core/
│   │   ├── alert_system.py     # Alert management
│   │   ├── data_versioning.py  # Dataset versioning
│   │   └── config.py           # Configuration
│   ├── data_quality/
│   │   ├── quality_engine.py   # Main quality validator
│   │   └── validators/         # Individual validators
│   ├── leakage_detection/
│   │   ├── leakage_engine.py   # Main detection orchestrator
│   │   ├── risk_scoring_model.py   # ML-based risk scoring
│   │   ├── impact_experiment.py    # Before/after experiments
│   │   └── detectors/          # Individual detectors
│   └── utils/
├── dashboard/                  # React + Vite frontend
├── case_studies/
│   └── churn_dataset/          # Complete case study
├── tests/
│   ├── unit/
│   │   ├── test_risk_scoring.py
│   │   ├── test_impact_experiment.py
│   │   ├── test_data_versioning.py
│   │   ├── test_alert_system.py
│   │   ├── test_detectors.py
│   │   └── test_validators.py
│   └── integration/
└── pyproject.toml
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_risk_scoring.py -v
```

## 📈 Why This Matters for ML

1. **Leakage causes unrealistic expectations** - Models appear to perform much better than they will in production
2. **ML-based detection catches subtle leaks** - Rule-based approaches miss complex patterns
3. **Impact experiments prove business value** - Show stakeholders exactly why leakage matters
4. **Version tracking enables observability** - Know when quality degrades before it affects models

## 🔧 Configuration

Create a `.env` file (see `.env.example`):

```bash
# Logging
LOG_LEVEL=INFO

# API
API_HOST=0.0.0.0
API_PORT=8000

# Alert thresholds
CORRELATION_THRESHOLD=0.95
DRIFT_THRESHOLD=0.1
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built for production ML pipelines** - ensuring data quality and preventing leakage before it affects your models.
