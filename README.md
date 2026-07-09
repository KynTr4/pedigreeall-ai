# Pedigreeall AI Horse Racing Prediction Platform

> Open-source AI platform for collecting, analyzing, and predicting Turkish horse racing results using machine learning.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-success)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Discovery](#api-discovery)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Data Lifecycle](#data-lifecycle)
- [Dashboard](#dashboard)
- [Screenshots](#screenshots)
- [VPS Deployment](#vps-deployment)
- [Backup Strategy](#backup-strategy)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)
- [Technical Documentation](#technical-documentation)
- [Project Documents](#project-documents)

---

## Overview

Pedigreeall is an open-source platform for collecting, processing, analyzing, and predicting Turkish horse racing data.

The project automatically discovers public API endpoints, downloads available race information, builds a historical SQLite warehouse, prepares machine-learning-ready datasets, evaluates prediction models, and serves results through a read-only FastAPI dashboard.

The goal of the project is to provide a reproducible data and machine learning pipeline for horse racing analysis, including data collection, normalization, feature engineering, prediction, evaluation, and production monitoring.

---

## Key Features

- Automated API endpoint discovery
- Anonymous public endpoint probing
- Historical horse racing data warehouse
- Pedigree and race performance analysis
- SQLite-based storage and checkpoint system
- Raw JSON preservation
- Relational data normalization
- CSV and Parquet analytics export
- Machine learning prediction pipeline
- Model evaluation and performance tracking
- SHAP explainability support
- FastAPI read-only dashboard
- VPS deployment workflow
- systemd timer automation
- SQLite hot backup support
- Rollback and health-check support

---

## Architecture

```text
Pedigreeall API
      │
      ▼
Endpoint Discovery
      │
      ▼
Data Collection
      │
      ▼
Raw JSON Storage
      │
      ▼
SQLite Warehouse
      │
      ▼
Data Normalization
      │
      ▼
Feature Engineering
      │
      ▼
Machine Learning Models
      │
      ▼
Prediction Engine
      │
      ▼
FastAPI Dashboard
```

---

## Project Structure

```text
discover_endpoints.py       # Discovers and catalogs API endpoints
probe_public_endpoints.py   # Safely probes anonymous public endpoints
discover_horses.py          # Discovers Turkish horses from TJK/API sources
scrape_pedigreeall.py       # Collects public horse and race data
normalize_data.py           # Converts raw JSON into relational tables
analyze_dataset.py          # Generates analytics outputs and quality reports
pedigreeall_core.py         # Core HTTP, retry, rate limit, SQLite logic
web_app.py                  # Read-only FastAPI dashboard
tests/                      # Unit and integration tests
deploy/                     # VPS deployment files
lake/                       # Analytics data lake outputs
reports/                    # Generated reports
output/                     # Model and pipeline outputs
```

---

## Installation

```bash
git clone https://github.com/KynTr4/pedigreeall-ai.git
cd pedigreeall-ai

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick Start

```bash
python discover_endpoints.py
python discover_horses.py
python scrape_pedigreeall.py
python normalize_data.py
python analyze_dataset.py
```

For a small smoke test:

```bash
python discover_horses.py --tjk-start 1 --tjk-end 20 --no-race-program --no-graph
python scrape_pedigreeall.py --rps 0.75 --concurrency 1 --batch-size 20
python normalize_data.py
python analyze_dataset.py --output lake/analytics
```

---

## API Discovery

The project validates available endpoints before downloading data.

### Supported capabilities

- Endpoint discovery
- Anonymous endpoint probing
- Access restriction reporting
- Safe public-only mode
- Automatic retry
- Resume support
- Rate limiting
- Request deduplication
- Raw response preservation
- SQLite checkpointing

The system avoids undocumented or guessed routes. API behavior and endpoint availability are recorded in reports so that restricted, unavailable, or timeout-prone endpoints are visible.

---

## Machine Learning Pipeline

The machine learning workflow consists of the following stages:

1. Data collection
2. Raw response validation
3. Data normalization
4. Feature engineering
5. Dataset generation
6. Model training
7. Model evaluation
8. Prediction generation
9. Performance tracking
10. Dashboard visualization

The project is designed to support multiple models and repeatable experiments.

---

## Data Lifecycle

```text
Raw JSON
    │
    ▼
SQLite Warehouse
    │
    ▼
Normalized Tables
    │
    ▼
Feature Engineering
    │
    ▼
Training Dataset
    │
    ▼
Prediction Models
    │
    ▼
Prediction Results
    │
    ▼
FastAPI Dashboard
```

---

## Dashboard

The project includes a read-only FastAPI dashboard for monitoring data collection, predictions, model outputs, and system health.

### Dashboard capabilities

- Race prediction display
- Historical race analysis
- Model comparison
- Prediction result tracking
- Data quality visibility
- Read-only SQLite access
- Basic Authentication support
- `/api/*` endpoints for structured access

---

## Screenshots

### System Dashboard

The system dashboard shows database status, race-day data, prediction snapshots, result matching, and shadow-mode monitoring.

![System Dashboard](docs/images/dashboard-status.png)

---

### Betting Simulation

The betting simulation page compares model outputs such as CatBoost, XGBoost, Logistic Regression, and Ensemble under a configurable flat-stake strategy.

![Betting Simulation](docs/images/dashboard-betting.png)

---

### Shadow Monitoring

The shadow monitoring page tracks model performance, Top-1 / Top-3 / Top-5 accuracy, ROI, model comparison, and segment-level performance.

> Note: Screenshots are taken from a local development environment. Some metrics may appear as zero until enough evaluated races and prediction results are collected.

![Shadow Monitoring](docs/images/dashboard-shadow.png)

---

## VPS Deployment

The repository includes a production-oriented VPS deployment workflow.

### Deployment features

- Git-based deployment
- Automatic database migrations
- Health checks
- systemd services
- systemd timers
- Read-only FastAPI dashboard
- Nginx reverse proxy support
- Backup automation
- Rollback support
- Log rotation

Typical deployment flow:

```bash
cd /opt/pedigreeall-ai
sudo ./deploy.sh
```

---

## Backup Strategy

The project includes an automated SQLite backup system.

### Backup schedule

- Daily backups
- Weekly backups
- Monthly backups

### Backup features

- SQLite hot backup
- Retention policy
- Automatic pruning
- Restore support
- Cleanup automation
- Log rotation

---

## Roadmap

Planned improvements:

- Improve prediction accuracy
- Better feature engineering
- Hyperparameter optimization
- Docker support
- PostgreSQL support
- Distributed model training
- Live prediction API
- Additional machine learning models
- Better SHAP explainability
- Performance monitoring dashboard
- Public documentation improvements

---

## Contributing

Contributions are welcome.

If you plan to add a new feature or make significant changes, please open an Issue before submitting a Pull Request.

### Development workflow

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

## License

This project is released under the **MIT License**.

---

## Disclaimer

This project is intended for research and educational purposes.

Horse racing predictions are probabilistic estimates based on historical data and machine learning models. They should **not** be interpreted as guaranteed outcomes, betting advice, or financial advice.

---

## Technical Documentation

Detailed technical documentation is being organized and will be expanded over time.

Current documentation areas:

- Public API mode
- Endpoint catalog
- Automatic horse discovery
- Data lifecycle
- Reliability
- Test plan
- VPS deployment
- Backup system
- Cleanup
- Production deployment

---

## Project Documents

- [Contributing Guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Security Policy](SECURITY.md)
- [License](LICENSE)
