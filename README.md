<div align="center">

# 🛡️ DataGuard

### Automated Data Quality & Leakage Detection System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A production-ready, full-stack solution for automated data quality validation and ML data leakage detection.**

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [API Reference](#-api-reference) • [Contributing](#-contributing)

</div>

---

## 📋 Overview

DataGuard is an enterprise-grade data quality management platform that helps data scientists and ML engineers ensure their datasets meet quality standards and are free from data leakage issues before model training.

### Why DataGuard?

- **Prevent Model Failures**: Catch data quality issues before they corrupt your ML pipeline
- **Detect Data Leakage**: Identify train-test contamination, target leakage, and temporal issues
- **Modern Dashboard**: Beautiful React UI for real-time validation and monitoring
- **Production Ready**: FastAPI backend with comprehensive error handling and logging
- **Extensible**: Custom validation rules with YAML configuration support

---

## ✨ Features

### 🔍 Data Quality Validation

| Validator | Description |
|-----------|-------------|
| **Schema Validator** | Type checking, constraint validation, pattern matching, schema inference |
| **Completeness Checker** | Missing value detection, null pattern analysis, configurable thresholds |
| **Consistency Analyzer** | Cross-column rules, referential integrity, expression validation |
| **Accuracy Validator** | Range checks, outlier detection, domain validation, format verification |
| **Timeliness Monitor** | Data freshness, future date detection, temporal gap analysis |
| **Custom Rules** | 16+ built-in validators (email, phone, credit card, etc.) with YAML support |

### 🚨 Leakage Detection

| Detector | Description |
|----------|-------------|
| **Train-Test Detector** | Exact/near-duplicate detection, distribution comparison |
| **Target Leakage Detector** | High correlation analysis, mutual information scoring |
| **Feature Leakage Detector** | Suspicious patterns, constant features, metadata analysis |
| **Temporal Leakage Detector** | Time overlap, look-ahead bias, ordering validation |
| **Data Drift Detector** | KS test, Chi-squared test, Population Stability Index (PSI) |

### 🎨 Modern Dashboard

- **Real-time Validation**: Drag & drop CSV upload with instant results
- **Quality Scoring**: Industry-standard tier-based scoring system
- **Session Tracking**: Validation history with trend visualization
- **Custom Rules UI**: Configure and manage validation rules visually
- **Dark Theme**: Beautiful glassmorphism design with animations

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- pip or uv (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/Naman09746/data-guard.git
cd automated-data-quality-leakage-detection

# Install Python dependencies
pip install -e ".[dev]"

# Install dashboard dependencies
cd dashboard
npm install
cd ..
```

### Running the Application

**Terminal 1 - Backend API:**
```bash
make run
# or
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend Dashboard:**
```bash
cd dashboard
npm run dev
```

**Access the application:**
- 🌐 Dashboard: http://localhost:5173
- 📡 API Docs: http://localhost:8000/docs
- ❤️ Health Check: http://localhost:8000/api/v1/health

---

## 📖 Documentation

### Python API Usage

```python
import pandas as pd
from src.data_quality import DataQualityEngine
from src.leakage_detection import LeakageDetectionEngine

# Load your data
df = pd.read_csv("your_data.csv")

# Data Quality Validation
quality_engine = DataQualityEngine()
quality_report = quality_engine.validate(df)

print(f"Status: {quality_report.status}")
print(f"Issues Found: {quality_report.total_issues}")
print(quality_report.to_markdown())

# Leakage Detection
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

leakage_engine = LeakageDetectionEngine()
leakage_report = leakage_engine.detect(
    train_data=train_df,
    test_data=test_df,
    target_column="target"
)

print(f"Clean: {leakage_report.is_clean}")
print(leakage_report.to_markdown())
```

### Custom Validation Rules

```python
from src.data_quality.validators.custom_rules import CustomRulesValidator, CustomRule

# Define custom rules
rules = [
    CustomRule(
        name="Email Validation",
        rule_type="email",
        column="email",
        severity="error"
    ),
    CustomRule(
        name="Age Range",
        rule_type="range",
        column="age",
        parameters={"min": 0, "max": 120}
    ),
    CustomRule(
        name="Phone Format",
        rule_type="phone",
        column="phone",
        parameters={"format": "international"}
    )
]

validator = CustomRulesValidator(rules=rules)
result = validator.validate(df)
```

### YAML Rule Configuration

```yaml
# config/custom_rules.yaml
rules:
  - name: Email Validation
    rule_type: email
    column: customer_email
    enabled: true
    severity: error

  - name: Transaction Amount
    rule_type: range
    column: amount
    parameters:
      min: 0
      max: 1000000
    severity: warning
```

---

## 🔌 API Reference

### Data Quality Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/quality/validate` | Validate CSV file upload |
| `POST` | `/api/v1/quality/validate/json` | Validate JSON data |

### Leakage Detection Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/leakage/detect` | Detect leakage in uploaded files |
| `POST` | `/api/v1/leakage/detect/json` | Detect leakage in JSON data |

### Example Request

```bash
# Quality Validation
curl -X POST "http://localhost:8000/api/v1/quality/validate" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_data.csv"

# Leakage Detection
curl -X POST "http://localhost:8000/api/v1/leakage/detect" \
  -H "accept: application/json" \
  -F "train_file=@train.csv" \
  -F "test_file=@test.csv" \
  -F "target_column=target"
```

### Response Format

```json
{
  "status": "warning",
  "passed": false,
  "total_issues": 3,
  "duration_seconds": 0.017,
  "summary": {
    "data_rows": 2880,
    "data_columns": 7,
    "validators_run": 5,
    "validators_passed": 2
  },
  "results": [
    {
      "validator_name": "SchemaValidator",
      "status": "warning",
      "issues": [...]
    }
  ]
}
```

---

## 📁 Project Structure

```
automated-data-quality-leakage-detection/
├── src/
│   ├── core/                    # Configuration, exceptions, logging
│   │   ├── config.py            # Pydantic settings management
│   │   ├── exceptions.py        # Custom exception hierarchy
│   │   └── logging_config.py    # Structured logging with structlog
│   │
│   ├── data_quality/            # Data quality validation
│   │   ├── validators/          # Individual validators
│   │   │   ├── schema_validator.py
│   │   │   ├── completeness_checker.py
│   │   │   ├── consistency_analyzer.py
│   │   │   ├── accuracy_validator.py
│   │   │   ├── timeliness_monitor.py
│   │   │   └── custom_rules.py
│   │   ├── quality_engine.py    # Orchestration engine
│   │   └── quality_report.py    # Report generation
│   │
│   ├── leakage_detection/       # ML leakage detection
│   │   ├── detectors/           # Individual detectors
│   │   │   ├── train_test_detector.py
│   │   │   ├── target_leakage_detector.py
│   │   │   ├── feature_leakage_detector.py
│   │   │   ├── temporal_leakage_detector.py
│   │   │   └── data_drift_detector.py
│   │   ├── leakage_engine.py    # Orchestration engine
│   │   └── leakage_report.py    # Report generation
│   │
│   ├── api/                     # FastAPI REST API
│   │   ├── main.py              # App initialization
│   │   ├── routes/              # API endpoints
│   │   └── schemas/             # Pydantic request/response models
│   │
│   └── utils/                   # Utilities
│       ├── data_loader.py       # Multi-format data loading
│       └── statistics.py        # Statistical analysis
│
├── dashboard/                   # React + TypeScript frontend
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   ├── pages/               # Page components
│   │   ├── api/                 # API client
│   │   └── store/               # State management
│   └── package.json
│
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
│
├── config/                      # Configuration files
│   └── settings.yaml            # Application settings
│
├── Dockerfile                   # Container configuration
├── docker-compose.yml           # Multi-service orchestration
├── Makefile                     # Development commands
└── pyproject.toml               # Python project configuration
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# .env file
DQ_ENVIRONMENT=development
DQ_DEBUG=true
DQ_LOG_LEVEL=INFO

# Quality thresholds
DQ_QUALITY__MISSING_VALUE_THRESHOLD=0.1
DQ_QUALITY__OUTLIER_THRESHOLD=3.0

# Leakage thresholds
DQ_LEAKAGE__CORRELATION_THRESHOLD=0.95
DQ_LEAKAGE__SIMILARITY_THRESHOLD=0.9

# API settings
DQ_API__HOST=0.0.0.0
DQ_API__PORT=8000
```

### YAML Configuration

```yaml
# config/settings.yaml
quality:
  missing_value_threshold: 0.1
  outlier_threshold: 3.0
  enable_schema_validation: true
  enable_completeness_check: true

leakage:
  correlation_threshold: 0.95
  similarity_threshold: 0.9
  check_temporal_leakage: true
```

---

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/unit/test_validators.py -v

# Run linting
make lint

# Run type checking
make type-check
```

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t dataguard .
docker run -p 8000:8000 dataguard
```

---

## 🔐 Quality Score Calculation

DataGuard uses an industry-standard tier-based scoring system:

| Status | Base Score | Issue Penalty | Score Range |
|--------|------------|---------------|-------------|
| **PASSED** | 100% | -2 pts/issue (max -15) | 85-100% |
| **WARNING** | 80% | -2 pts/issue (max -30) | 50-84% |
| **FAILED** | 45% | -2 pts/issue (max -45) | 0-49% |

**Example:**
- Passed with 2 issues: `100 - (2 × 2) = 96%`
- Warning with 5 issues: `80 - (5 × 2) = 70%`

---

## 🛠️ Built With

**Backend:**
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [scikit-learn](https://scikit-learn.org/) - ML utilities
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [structlog](https://www.structlog.org/) - Structured logging

**Frontend:**
- [React 18](https://react.dev/) - UI library
- [TypeScript](https://www.typescriptlang.org/) - Type safety
- [Vite](https://vitejs.dev/) - Build tool
- [Tailwind CSS](https://tailwindcss.com/) - Styling
- [Recharts](https://recharts.org/) - Data visualization
- [React Query](https://tanstack.com/query) - API state management

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📬 Support

- 📧 Email: namanjoshi09746@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/Naman09746/data-guard/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Naman09746/data-guard/discussions)

---

<div align="center">

Made with ❤️ for the Data Science Community

**⭐ Star this repo if you find it useful!**

</div>
