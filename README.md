# 💳 CardGuard AI  
**Anomaly Detection in Credit Card Transactions**  
_A machine learning approach to detect fraudulent credit card transactions using Isolation Forest and Local Outlier Factor (LOF)._

## 🚀 About the Project
Credit card fraud detection is a critical challenge in the financial sector.  
**CardGuard AI** leverages advanced anomaly detection techniques to identify potentially fraudulent transactions from highly imbalanced datasets.  
This project implements unsupervised learning methods — Isolation Forest and Local Outlier Factor — to effectively flag outliers without relying on labeled fraudulent data.

## 🛠️ Technologies Used
- **Python 3.x**
- **Pandas** for data processing
- **NumPy** for numerical operations
- **Matplotlib / Seaborn** for visualization
- **Scikit-Learn (sklearn)** for machine learning models (Isolation Forest, LOF)


## 📂 Dataset
- Dataset used: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Note**: The dataset is not uploaded in this repository due to Kaggle's licensing. Download it directly from the link above.

## 📈 Project Workflow
1. **Data Exploration**: Analyze data distribution, correlations, and imbalance.
2. **Preprocessing**: Clean data, handle missing values (if any).
3. **Modeling**:
    - **Isolation Forest**
    - **Local Outlier Factor (LOF)**
4. **Evaluation**: Confusion Matrix, Precision, Recall, ROC AUC.

## 📝 Results
| Model             | Precision | Recall | AUC Score |
|-------------------|-----------|--------|-----------|
| Isolation Forest  | High      | Moderate | Good      |
| Local Outlier Factor | Moderate  | High    | Good      |

Both methods effectively capture anomalies despite extreme class imbalance ---

## ⚙️ Installation  
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/BHAVYAPIPLANI/CardGuard-AI.git
cd CardGuard-AI
