# Stock Market Machine Learning 

This repository contains a series of Tasks exploring **machine learning and trading strategies** on stock market data (using NVIDIA NVDA stock and others).  
The work spans from simple labeling strategies to advanced classifiers, ensemble models, and clustering.

---

## 📂 Overview

### 1. Label Creation
- **Weekly Labeling Strategy**  
  - Assigned weekly labels (`Green` = invest, `Red` = no-invest) for NVDA stock (2020–2024).  
  - Rules based on weekly return and volatility.  
  - Output: `Labeled Weekly Stock Data.csv`.

---

### 2. Strategy Backtesting
- **Trading Strategy Simulation**  
  - Implemented trading based on labels.  
  - Buy $100 at first `Green` week, exit on `Red`.  
  - Compared against **Buy-and-Hold**.  
- **Buy-and-Hold vs Label-Based Strategy**  
  - Evaluated both strategies over 5 years.  
  - Label-based strategy often outperformed during volatile periods.  
- **Linear Models Assignment**  
  - Used `(µ, σ)` as features to predict labels.  
  - Models: Linear Regression, Polynomial Regression, GLM.  
  - Evaluated by SSE and accuracy.  
  - Output: `Yearly Strategy Comparison.csv`.

---

### 3. Classical ML Models (NB, LDA/QDA, SVM)
- **Naive Bayes with Student-t**  
  - Implemented Student-t Naive Bayes with df = 0.5, 1, 5.  
  - Computed confusion matrices, accuracy, TPR, TNR.  
  - Compared to Gaussian NB.  
  - Best df used in trading simulation.  

- **LDA vs QDA**  
  - Implemented Linear and Quadratic Discriminant Analysis.  
  - Computed accuracy, confusion matrices, TPR, TNR.  
  - Trading strategies compared to Buy-and-Hold.  

- **Support Vector Machines (SVM)**  
  - Implemented Linear SVM, Gaussian (RBF) SVM, Polynomial SVM (degree=2).  
  - Computed accuracy, confusion matrices, TPR, TNR.  
  - Trading strategy based on Linear SVM predictions significantly outperformed Buy-and-Hold.
    
- **k-Nearest Neighbors (k-NN)**  
  - Trained k-NN using Years 1–3, tested on Years 4–5.  
  - Evaluated multiple k values, selected best k.  
  - Computed accuracy, confusion matrix, sensitivity (TPR), and specificity (TNR).  
  - Trading strategy with k-NN labels compared to Buy-and-Hold.  

---

### 4. Tree-Based Methods (Decision Tree, Random Forest, AdaBoost)
- **Decision Tree**  
  - Implemented with both **entropy** and **gini** criteria.  
  - Computed accuracy, confusion matrices, TPR, TNR.  
  - Trading strategy vs Buy-and-Hold.  
  - Compared criteria performance.  

- **Random Forest**  
  - Implemented ensemble method.  
  - Evaluated accuracy, confusion matrices, TPR, TNR.  
  - Trading performance compared to Buy-and-Hold.  

- **AdaBoost**  
  - Implemented AdaBoost with decision stumps as weak learners.  
  - Measured accuracy, confusion matrices, TPR, TNR.  
  - Trading strategy outperformed Buy-and-Hold, showing benefits of boosting.  

---

### 5. Unsupervised Clustering with Trajectory Similarity (Hamming Distance)
- **Hamming Distance Analysis**  
  - Computed Hamming distances between label trajectories of models/stocks.  
  - Found pairs of stocks with largest/smallest similarity.  
  - Calculated average distance.  

- **Clustering Dow Jones Stocks**  
  - Selected 5 Dow Jones stocks (AAPL, MSFT, JNJ, JPM, V) + SPY.  
  - Monthly regression (Stock ~ SPY) residuals collected (60 months).  
  - Applied K-means clustering (k = 3–7), elbow method → best k=4.  
  - Built time-cluster trajectories for each stock.  
  - Computed Hamming distances to compare stock similarity.  
  - Found: AAPL & MSFT most similar, AAPL & JPM most divergent.  

---

## 🛠️ Tools & Libraries
- Python (Colab/Jupyter)
- Pandas, NumPy
- Scikit-learn (SVM, Decision Tree, Random Forest, LDA, QDA, Naive Bayes, AdaBoost, KMeans)
- Matplotlib / Seaborn
- yfinance (stock data)

---

## 📑 Deliverables
Each assignment includes:
- **Colab-ready Python code** (`.ipynb` / `.py`)
- **Word/PDF report** with:
  - Problem statement
  - Methodology
  - Results (tables, confusion matrices, plots)
  - Conclusions

---

## 🎯 Conclusion
Across these assignments, we explored a full pipeline of **quantitative finance + ML**:
1. **Label creation** → setting up invest/no-invest signals  
2. **Backtesting** → comparing trading vs buy-and-hold  
3. **Classical ML models** → Naive Bayes, LDA/QDA, SVM  
4. **Tree-based models** → Decision Tree, Random Forest, AdaBoost  
5. **Unsupervised clustering** → K-means on stock residuals with Hamming distance similarity  

This progression demonstrates the application of both **supervised and unsupervised ML** methods in stock market analysis.

