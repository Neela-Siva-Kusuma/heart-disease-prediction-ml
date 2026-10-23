\# 🫀 Heart Disease Prediction ML Pipeline \& Streamlit App



A end-to-end, production-ready Machine Learning application that predicts cardiovascular disease risk using clinical patient parameters. Built with \*\*Scikit-Learn\*\*, \*\*Pandas\*\*, and \*\*Streamlit\*\*, this project transitions a traditional Jupyter research notebook into a modular, containerized machine learning pipeline deployed to the cloud.



\---



\## 📌 Features \& Engineering Highlights



\- \*\*Modular Architecture:\*\* Clean separation of concern with `train.py` handling data preprocessing/training and `app.py` managing web UI inference.

\- \*\*Robust Preprocessing Pipeline:\*\* Integrated `SimpleImputer` (median imputation for string `'?'` missing values) and `StandardScaler` to process raw clinical features smoothly.

\- \*\*Clinical Edge Case Handling:\*\* Tuned model classification threshold (`0.20` decision boundary) to ensure young patients with severe asymptomatic markers (e.g., high ST depression, elevated cholesterol) are accurately flagged.

\- \*\*Zero Warnings:\*\* Fully compatible with Pandas DataFrames during inference to avoid Scikit-Learn feature name mismatch warnings.



\---



\## 🛠️ Tech Stack \& Dependencies



\- \*\*Language:\*\* Python 3.13

\- \*\*Libraries:\*\* Scikit-Learn, Pandas, NumPy, Joblib

\- \*\*Web Framework:\*\* Streamlit

\- \*\*Version Control:\*\* Git \& GitHub



\---



\## 📂 Repository Structure



heart-disease-prediction-ml/

│

├── train.py          # Preprocessing pipeline, model training \& artifact serialization

├── app.py            # Streamlit interactive web application for real-time inference

├── heart.csv         # UCI Cleveland Heart Disease dataset

├── model.pkl         # Serialized Random Forest Classifier

├── scaler.pkl        # Serialized StandardScaler artifact

├── imputer.pkl       # Serialized SimpleImputer artifact

├── requirements.txt  # Project dependency specifications

└── README.md         # Project documentation







\## 📊 Dataset Overview



This project uses the \*\*UCI Cleveland Heart Disease Dataset\*\* comprising 13 clinical features:



| Feature Name | Description | Type / Range |

| :--- | :--- | :--- |

| `age` | Age in years | Numeric (1 - 120) |

| `sex` | Gender | Categorical (0 = Female, 1 = Male) |

| `cp` | Chest Pain Type | Categorical (0 - 3) |

| `trestbps` | Resting Blood Pressure | Numeric (mm Hg) |

| `chol` | Serum Cholesterol | Numeric (mg/dl) |

| `fbs` | Fasting Blood Sugar > 120 mg/dl | Categorical (0 = False, 1 = True) |

| `restecg` | Resting ECG Results | Categorical (0 - 2) |

| `thalach` | Maximum Heart Rate Achieved | Numeric |

| `exang` | Exercise-Induced Angina | Categorical (0 = No, 1 = Yes) |

| `oldpeak` | ST Depression Induced by Exercise | Numeric |

| `slope` | Slope of Peak Exercise ST Segment | Categorical (0 - 2) |

| `ca` | Number of Major Vessels Colored by Fluoroscopy | Categorical (0 - 3) |

| `thal` | Thalassemia Status | Categorical (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect) |



\---



\## 🚀 Local Installation \& Setup



\### 1. Clone the Repository

```bash

git clone \[https://github.com/Neela-Siva-Kusuma/heart-disease-prediction-ml.git](https://github.com/Neela-Siva-Kusuma/heart-disease-prediction-ml.git)

cd heart-disease-prediction-ml

