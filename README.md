# Amazon Hybrid Recommendation Engine & Performance Audit

This repository contains a production-engineered, three-tier hybrid recommendation framework built using the Amazon Electronics reviews dataset from Kaggle. Instead of relying on a single algorithm, this system implements a modular pipeline that dynamically routes users to different recommendation engines based on their historical data and active storefront browsing state. 

The codebase focuses heavily on data hygiene, memory efficiency, and real-world deployment readiness. By enforcing a strict split-before-filter architecture, the pipeline completely eliminates data leakage, ensuring all performance metrics accurately reflect true performance on unseen customer distributions.

---

## System Architecture & Fallback Flow

The application handles different customer states on an e-commerce platform using three distinct layers:

1. **Tier 3 — Personalised Discovery Core (SVD Matrix Factorisation):** This layer handles logged-in, returning power users. It uses a Singular Value Decomposition algorithm to map deep, hidden purchasing habits and guess explicit star ratings for unbought products.
2. **Tier 2 — Real-Time Cross-Selling Engine (Cosine Similarity):** This block activates on specific product detail pages. It calculates angular similarity vectors between product profiles to instantly render "Frequently Bought Together" modules without needing any personal user data.
3. **Tier 1 — Global Cold-Start Fallback (Bayesian Weighted Popularity):** This serves as the baseline safety net for brand-new or anonymous web visitors. It runs a manipulation-proof IMDb Bayesian formula that anchors low-volume items to the platform average, ensuring users only see reliable trending items.

---

## Tech Stack

* **Language & Environment:** Python, Jupyter Notebook
* **Data Processing:** Pandas, NumPy, SciPy (Sparse Matrices)
* **Machine Learning Core:** Surprise (SVD, Reader, Dataset, Dump)
* **Model Optimisation:** Scikit-Learn (GridSearchCV, Train_Test_Split)
* **File Serialization:** JSON, Pickle

---

## Pipeline Layout & Execution Sections

The project code is organised into six functional sections to mirror an enterprise machine learning pipeline:

### Environment Setup & Data Pipelines
This section loads the raw data and instantly drops timestamps to minimise RAM overhead. To maintain total data integrity, the data is split into training (80%) and testing (20%) sets before running any filters or statistics. Once the split is complete, a simultaneous bitwise filter isolates a dense core of active users and items with at least 50 reviews to ensure stable matrix operations.

### Tier 1 Popularity Engine
This section builds the global fallback asset. It calculates the global rating mean and cuts off products below the 90th percentile of review counts. It then applies the Bayesian Weighted Rating formula to smooth out the scores, protecting the platform from showing products that randomly received a single five-star review.

### Memory-Optimised Item Similarity
This structural block configures the cross-selling matrix. Instead of using standard Pandas pivot tables—which fill system RAM with millions of useless empty zeros and crash servers—this code maps alphanumeric Amazon IDs into sequential codes and passes a Compressed Sparse Row (CSR) matrix to Scikit-Learn to calculate Cosine Similarity efficiently.

### Tuning Engine & Model Selection
This part handles hyperparameter optimisation. Instead of guessing SVD settings blindly, it deploys an automated `GridSearchCV` framework with 3-Fold Cross-Validation. It executes 24 mini-tests across a grid of latent dimensions, learning speeds, and safety penalties to isolate the configuration with the lowest error before training it on the full training pool.

### System Evaluation
This section acts as the model's independent final exam using the untouched out-of-sample test split. It passes the hidden validation entries into the finalised, tuned SVD model and scores the predictions using the native `accuracy.rmse` toolkit from the Surprise library, outputting a stable and realistic final **RMSE of 1.4049**.

### Production Artifact Export
The final component handles file serialization to prepare the system for deployment. Because a live web server cannot afford to re-clean data or re-train an SVD model during a live page refresh, this section freezes and saves our work into three specific deployment assets: a lightweight text JSON array for cold-start homepages, a pickled binary DataFrame for fast item-page lookups, and a natively frozen SVD model object for real-time personalised predictions.

---

## Exported Production Assets

The training script automatically outputs the following deployment artifacts to disk:
* `cold_start_recommendations.json`: A tiny, plain-text array containing the top 10 globally popular product IDs for instant homepage rendering.
* `item_similarity.pkl`: A serialized binary lookup table containing item-to-item cosine similarity vectors.
* `svd_model.pkl`: A securely frozen, optimised SVD model object saved via native Surprise serialization, ready for live API inference hooks (`model.predict()`).
