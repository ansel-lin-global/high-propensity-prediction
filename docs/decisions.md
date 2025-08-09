# Key Design Decisions

This document explains the major decisions made while building the **High-Propensity Purchase Prediction System**, focusing on the business context behind each choice.

---

## 1. Business Problem Framing
The original EDM targeting was based on popular products, without personalization.  
We reframed the problem as **"identify which users are most likely to purchase within the next few days"** so marketing efforts could be focused where they have the highest impact.

---

## 2. Time-Series Sliding Window
**Why:** User behavior patterns shift quickly in e-commerce.  
**Decision:** Train models using a **sliding window** (10-day observation, 3-day prediction) to capture the most recent behavioral signals while keeping enough history for robust learning.  
**Impact:** Improved model recall on active buyers without increasing false positives.

---

## 3. Recall@Top-K as Primary Metric
**Why:** In real campaigns, we send EDMs to a limited audience due to budget and email fatigue constraints.  
**Decision:** Optimize for **recall at the target audience size (K)** instead of global AUC/accuracy.  
**Impact:** Ensures that the top-ranked predictions are truly high-quality leads.

---

## 4. Drift Detection for Model Reliability
**Why:** User behavior changes with seasons, promotions, and product launches.  
**Decision:** Implement both **data drift** (feature distribution shifts) and **concept drift** (target label relationship changes) monitoring.  
**Impact:** Allows proactive retraining, avoiding performance drops during key campaigns.

---

## 5. Vertex AI Pipelines for Scalability
**Why:** Manual notebook runs were error-prone and hard to maintain.  
**Decision:** Move to **Vertex AI Pipelines** for automated, repeatable workflows covering training, daily scoring, drift detection, and retraining.  
**Impact:** Reduced operational overhead and ensured consistent execution across markets.

---

## 6. Product-Centric Evaluation
**Why:** Model success is defined by business impact, not just technical metrics.  
**Decision:** Link model predictions to **actual EDM campaign results** (CTR, conversion rate, incremental revenue).  
**Impact:** Created a closed feedback loop, guiding both model improvements and marketing strategy.

---

## Summary
Every decision in this system balances **technical feasibility** with **business impact**.  
The result is an ML-powered targeting system that scales globally, adapts to market changes, and delivers measurable revenue growth.
