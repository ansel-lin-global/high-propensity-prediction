# 🎯 High Propensity Purchase Prediction System

An end-to-end MLOps system to identify high-intent users for e-commerce campaigns — built with Vertex AI Pipelines, automated drift detection, retraining, and integrated CI/CD deployment.

🔗 **Full project write-up**: [Medium Article](https://medium.com/@ansel-lin/from-model-to-deployment-building-an-automated-high-propensity-purchase-prediction-system-2aed17de9412)

---

## 🧠 Problem Statement

Traditional EDM segmentation often targets users based on simple heuristics or hot products. This leads to low conversion rates and wasted campaign budget.

This project solves that by:
- Predicting high-intent users using rich behavioral signals (GA4)
- Automating model lifecycle: training, prediction, drift detection, and retraining
- Enabling real-time EDM segmentation at global scale

---

## ⚙️ System Architecture

```mermaid
flowchart TD
    A[GA4 Event Data in BigQuery] --> B[Rolling Feature Generation\n(10d window → 3d label)]
    B --> C[Model Training\n(LightGBM + SMOTE + Undersampling)]
    C --> D[Daily Prediction Pipeline\n(Vertex AI Pipelines)]
    D --> E[Drift Detection\n(Data Drift & Concept Drift)]
    E --> F[Trigger Retraining + CI/CD\n(GitHub Actions)]
    F --> G[Export Top-K Users to GCS\nfor EDM Targeting]
```
