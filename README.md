# ML_Docker
# ❤️ Heart Disease ML – End-to-End (EDA → Train → Predict)

An interactive and containerized **machine learning pipeline** for **Heart Disease Prediction**, built with **Streamlit**, **scikit-learn**, and **Docker**.

This application enables you to:
- Perform quick **Exploratory Data Analysis (EDA)**
- Train and evaluate ML models (**Random Forest** or **Logistic Regression**)
- Save and download trained models and feature schemas
- Make both **single-patient** and **batch** predictions

---

## ⚙️ Installation & Setup

### 1. Clone the repository
bash
git clone https://github.com/<your-username>/heart-ml.git
cd heart-ml
docker build -t heart-ml 
docker-compose up
