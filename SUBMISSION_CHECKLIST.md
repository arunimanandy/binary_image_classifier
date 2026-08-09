# Assignment Compliance Checklist

Source: MLOps Assignment 2 requirements.

## M1: Model Development and Experiment Tracking
- Git-ready source layout: included.
- DVC data versioning: `dvc.yaml`, `params.yaml`, README commands included.
- Baseline model: `src/ml/model.py` and `src/ml/train.py` included.
- Serialized trained artifact: produced as `models/catdog_cnn.pt` after running training on Kaggle dataset.
- MLflow tracking: `src/ml/train.py` logs params, metrics, confusion matrix, loss curve, and model artifact.

## M2: Packaging and Containerization
- REST service: `src/api/main.py`.
- Endpoints: `/health`, `/predict`, `/metrics`.
- Dependencies: `requirements.txt` with pinned versions.
- Containerization: `Dockerfile`.

## M3: CI Pipeline
- Unit tests: `tests/test_preprocess.py`, `tests/test_inference.py`.
- GitHub Actions CI: `.github/workflows/ci.yml`.
- Docker image build and GHCR publish: included in CI workflow.

## M4: CD Pipeline and Deployment
- Deployment target selected: Docker Compose.
- Compose manifest: `deployment/docker-compose/docker-compose.yml`.
- CD workflow: `.github/workflows/cd.yml`.
- Smoke checks: `scripts/smoke_test.py` and workflow health check.

## M5: Monitoring, Logs and Final Submission
- Logging: request, predicted label, latency, and failures logged in `src/api/main.py`.
- Metrics: `/metrics` exposes request count, prediction count, total and average latency.
- Simulated request support: `assets/smoke_test_image.jpg` for smoke testing API mechanics.
