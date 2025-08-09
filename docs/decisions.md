# 📌 Key Design Decisions

> This document explains **why** each major decision was made in the High Propensity Purchase Prediction System.  
> Focus: business impact, maintainability, and scalability in a real e-commerce environment.

---

## 1. Business-Driven Problem Framing
**Decision:**  
Frame the model objective as *"maximize campaign recall at fixed budget"* rather than pure accuracy.

**Reasoning:**  
- EDM campaigns have limited daily budget — recall at top-K is more valuable than generic accuracy.  
- Aligning model outputs with **marketing ROI** ensures stakeholder buy-in.  

**Impact:**  
- Business metrics (Recall@TopK, ROI) became the primary success criteria.
- Model directly influenced **global campaign targeting**.

---

## 2. Rolling Window Labeling
**Decision:**  
Use a 14-day observation window → 3-day prediction horizon.

**Reasoning:**  
- Captures recent intent signals without being overly reactive to short-term spikes.  
- Prevents information leakage and supports **temporal generalization**.

**Impact:**  
- Stable performance across **40+ global markets** with varying user behaviors.  

---

## 3. Multi-Model Pipeline with Vertex AI
**Decision:**  
Train LightGBM, XGBoost, and CatBoost in parallel.

**Reasoning:**  
- Ensures robustness by comparing multiple algorithms.  
- Vertex AI Pipelines orchestrates **parallel training and evaluation** for reproducibility.

**Impact:**  
- Reduced model selection time from days to hours.  
- Consistent deployment process via **templated pipeline compilation**.

---

## 4. Drift-Aware Retraining
**Decision:**  
Trigger retraining only when **data or concept drift** exceeds threshold.

**Reasoning:**  
- Avoids unnecessary retraining costs.  
- Keeps model performance stable in dynamic e-commerce traffic patterns.

**Impact:**  
- Reduced training frequency by ~40% while maintaining prediction quality.  

---

## 5. CI/CD Integration with GitHub Actions
**Decision:**  
Use GitHub Actions for **pipeline template deployment** and version control.

**Reasoning:**  
- Separates **code updates** from **daily operations**.  
- Ensures reproducibility and easy rollback if a pipeline fails.

**Impact:**  
- Zero downtime during updates.  
- Clear change history for audit and debugging.

---

## 6. Modular Component Design
**Decision:**  
Split pipelines into independent components for data fetching, training, prediction, and monitoring.

**Reasoning:**  
- Enables component reusability for other predictive projects.  
- Easier to debug and extend without breaking the whole pipeline.

**Impact:**  
- Accelerated development of other ML use cases by reusing core components.

---

## 7. Public Repository as a Showcase
**Decision:**  
Strip sensitive SQL, proprietary feature logic, and internal metrics before publishing.

**Reasoning:**  
- Complies with corporate data policies.  
- Allows external recruiters to see **MLOps expertise** without exposing confidential assets.

**Impact:**  
- Functions as a **portfolio project** demonstrating end-to-end ML system capability.

---

## 📍 Summary
Every technical choice in this system was made with **product alignment** in mind.  
This is not just a model — it’s a **living ML product** that integrates with business processes, scales globally, and operates with minimal manual intervention.
