# Stock Market Machine Learning 

This repository contains a series of Task exploring **machine learning and trading strategies** on stock market data (using NVIDIA NVDA stock and others).  
The work spans from simple labeling strategies to advanced classifiers, ensemble models, and clustering.

---

## 📂 Overview

### 1. Weekly Labeling Strategy
- **Task:** Assign weekly labels (`Green` = invest, `Red` = no-invest) for NVDA stock (2020–2024).
- **Approach:** Created labeling rules based on weekly return and volatility.
- **Output:**
  - `Labeled Weekly Stock Data.csv`
  - Basis for further strategy simulations.

---

### 2. Trading Strategy Simulation
- **Task:** Implement a trading strategy using labels.
- **Rules:**
  - Buy $100 at first `Green` week opening.
  - Stay invested during `Green`, exit during `Red`.
- **Comparison:** Evaluated vs **Buy-and-Hold** strategy.
- **Output:**
  - `Trading Strategy Performance.csv`
  - Higher returns than buy-and-hold in many years.

---

### 3. Buy-and-Hold vs Label-Based Strategy
- **Task:** Compare 5-year performance of:
  - Label-based trading
  - Buy-and-hold strategy
- **Result:** Label-based trading often outperformed, depending on volatility.

---

### 4. Linear Models
- **Task:** Build predictive models using features `(µ, σ)`.
- **Models:**
  - Linear Regression
  - Polynomial Regression
  - Generalized Linear Model (GLM)
- **Evaluation:** SSE, accuracy comparison.
- **Output:** `Yearly Strategy Comparison.csv`

---

### 5. Naive Bayes with Student-t Distribution
- **Task:** Implement Student-t Naive Bayes with different degrees of freedom (df = 0.5, 1, 5).
- **Outputs:**
  - Confusion matrices
  - Accuracy, TPR, TNR
- **Comparison:** Student-t vs Gaussian NB
- **Trading Strategy:** Simulated based on best df.

---

### 6. LDA vs QDA
- **Task:** Implement Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA).
- **Evaluation:**
  - Accuracy
  - Confusion matrices
  - TPR, TNR
- **Trading Strategies:** Compared both to Buy-and-Hold.
- **Conclusion:** One classifier generalized better depending on test years.

---

### 7. Decision Tree Classifier
- **Task:** Implement Decision Tree with both **entropy** and **gini** criteria.
- **Outputs:**
  - Accuracy
  - Confusion Matrix
  - TPR & TNR
- **Trading Strategy:** Compared to Buy-and-Hold.
- **Finding:** Both achieved high accuracy, but differences in sensitivity/specificity highlighted pros/cons.

---

### 8. Random Forest Classifier
- **Task:** Implement Random Forest and evaluate.
- **Outputs:**
  - Accuracy, Confusion Matrix, TPR, TNR
- **Trading Strategy:** Simulated for last two years vs Buy-and-Hold.
- **Conclusion:** Random Forest performed more robustly than a single decision tree.

---

### 9. Hamming Distance Analysis
- **Task:** Compute **Hamming distance** between predicted label trajectories of different models/stocks.
- **Usage:** Measure similarity/dissimilarity in investment signals.
- **Outputs:** Distance matrices, largest/smallest pairs, average distances.

---

### 10. AdaBoost
- **Task:** Implement AdaBoost with decision stumps.
- **Outputs:**
  - Accuracy
  - Confusion matrix
  - TPR & TNR
- **Trading Strategy:** Compared AdaBoost vs Buy-and-Hold.
- **Finding:** Boosting improved generalization compared to standalone classifiers.

---

### 11. Clustering of Dow Jones Stocks
- **Task:** Cluster residuals of regression (stock returns vs SPY).
- **Steps:**
  - Regression residuals (60 months × 5 stocks)
  - K-means clustering with k=3–7
  - Elbow method → chose k=4
  - Time-cluster trajectories
  - Hamming distance between stock trajectories
- **Insights:**
  - AAPL & MSFT showed similar behavior.
  - AAPL & JPM most divergent.
  - Average Hamming distance ≈ 0.54.

---

## 🛠️ Tools & Libraries
- Python (Colab/Jupyter)
- Pandas, NumPy
- Scikit-learn (SVM, Decision Tree, Random Forest, LDA, QDA, Naive Bayes, AdaBoost)
- Matplotlib / Seaborn (visualizations)
- yfinance (stock data)

---

## 📑 Deliverables
Each Task includes:
- **Colab-ready Python code** (`.ipynb` / `.py`)
- **Word/PDF report** with:
  - Problem statement
  - Methodology
  - Results (tables, confusion matrices, plots)
  - Conclusions

---

## 🎯 Conclusion
Across these Tasks, we explored a full pipeline of **quantitative finance + ML**:
1. Label creation  
2. Strategy backtesting  
3. Classical ML models (NB, LDA/QDA, SVM)  
4. Tree-based methods (Decision Tree, Random Forest, AdaBoost)  
5. Unsupervised clustering with trajectory similarity (Hamming Distance)  

These projects provide a comprehensive foundation for applying **machine learning in stock trading analysis**.

---
