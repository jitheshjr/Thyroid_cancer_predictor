# 🧠 Thyroid Cancer Risk & Diagnosis Prediction (Streamlit App)

This repository contains a full machine learning pipeline and an interactive **Streamlit application** to predict:

- 🩺 **Diagnosis** of Thyroid Cancer – Benign or Malignant (Binary Classification)
- 🎯 **Risk Level** – Low, Medium, or High (Multiclass Classification)

The project is built using clinical and medical data, and designed for both educational and practical demonstration purposes.

---

## 📂 Project Structure
├── app.py # Streamlit application
├── artifacts/ # Contains all model and preprocessing files
│ ├── binning_edges_diagnosis.pkl
│ ├── binning_edges_risk.pkl
│ ├── diagnosis_predictor.pkl
│ ├── risk_predictor.pkl
│ ├── feature_columns_diagnosis.pkl
│ ├── feature_columns_risk.pkl
│ ├── scaler_diagnosis.pkl
│ └── scaler_risk.pkl
├── preprocessing/
│ └── binary_classification.ipynb # Notebook containing full preprocessing and training
│ └── multiclass_classification.ipynb # Notebook containing full preprocessing and training
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md # Project documentation

## 🧠 Models Overview

| Task                 | Model File                | Type             | Classes                          |
|----------------------|---------------------------|------------------|----------------------------------|
| Diagnosis Prediction | `diagnosis_predictor.pkl` | Binary Classifier| Benign, Malignant                |
| Risk Prediction      | `risk_predictor.pkl`      | Multiclass Classifier | Low, Medium, High           |

## 📁 Model & Preprocessing Artifacts (`/models/`)

These `.pkl` files are **required to run the Streamlit app** and are included in this repository:

### 🧪 Diagnosis Prediction Files:
| File                            | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| `diagnosis_predictor.pkl`       | Trained binary classification model to predict benign/malignant diagnosis |
| `scaler_diagnosis.pkl`          | `StandardScaler` used to scale numeric input features for diagnosis model |
| `feature_columns_diagnosis.pkl` | List of feature names used for diagnosis prediction inputs in the app      |
| `binning_edges_diagnosis.pkl`   | Binning thresholds for numerical features used in Streamlit select boxes   |

### 📊 Risk Prediction Files:
| File                         | Description                                                                |
|------------------------------|----------------------------------------------------------------------------|
| `risk_predictor.pkl`         | Trained multiclass model to predict thyroid cancer risk (Low/Medium/High) |
| `scaler_risk.pkl`            | `StandardScaler` used for numeric inputs of the risk model                |
| `feature_columns_risk.pkl`   | List of features used in risk prediction input                            |
| `binning_edges_risk.pkl`     | Binning thresholds used for categorical-like dropdowns in the app         |

## 📊 Dataset

The models were trained on a clinical dataset containing patient health parameters and diagnostic labels.  
If you'd like to train the models yourself, you can download the dataset from:

🔗 **[]Kaggle Dataset Link :] (https://www.kaggle.com/datasets/bhargavchirumamilla/thyroid-cancer-risk-dataset)**

## 🛠 Technologies Used

- **Python 3.10+**
- **Streamlit** – for building interactive web apps
- **Scikit-learn** – machine learning models and preprocessing
- **Pandas, NumPy** – data analysis and manipulation
- **Joblib** – efficient model serialization
- **Google Colab** – preprocessing and model training

## 📜 License

This project is open-source and free to use for non-commercial and academic purposes.  
If you use this work in research or development, please provide attribution.

## 🙌 Acknowledgments

- [Kaggle: Thyroid Disease Dataset](https://www.kaggle.com/datasets/bhargavchirumamilla/thyroid-cancer-risk-dataset) – for providing the publicly available clinical data
- The open-source contributors to **Streamlit**, **Scikit-learn**, and **XGBoost**
- The broader **Python** and **machine learning** communities for open knowledge sharing


