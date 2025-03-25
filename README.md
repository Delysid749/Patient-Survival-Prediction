

🏥 Patient Survival Prediction System
=====================================

This project aims to predict in-hospital mortality risk for ICU patients using machine learning models. It provides a full pipeline from preprocessing, correlation analysis, model training, ensemble learning, visualization, and a user-friendly GUI for real-time clinical prediction.

* * *

🎯 Project Objectives
---------------------

*   Build an ensemble model to predict hospital mortality risk
    
*   Extract critical features from patient records and ICU monitoring
    
*   Design a usable prediction interface for clinical settings
    
*   Provide structured results and insights to support decision-making
    
* * *

📁 Project Structure
--------------------

```
Patient–Survival–Prediction/
│
├── dataSet/
│   └── dataset.csv                # Raw patient-level data (CSV)
│
├── Model/                         # Saved models
│   ├── base_models.pkl            # Trained base models (XGB, DNN, LR)
│   └── meta_model.pkl            # Trained stacking meta-model
│
├── report/
│   └── logs/
│       ├── categorical_correlation.log
│       └── numerical_correlation.log
│
├── src/
│   ├── common/
│   │   └── variable_array.py     # Predefined categorical/numerical feature lists
│
│   ├── config/
│   │   └── path_config.py        # Path manager for logs, data, and models
│
│   ├── correlation_analysis/
│   │   ├── categorical_analysis.py   # Categorical feature correlation (e.g., Cramer’s V)
│   │   └── numerical_analysis.py     # Numerical feature correlation (e.g., Pearson)
│
│   ├── model/
│   │   ├── DNN.py                # Keras-based deep neural network
│   │   ├── Ensemble.py           # Stacking & Soft Voting ensemble logic
│   │   ├── LogisticRegression.py # Scikit-learn logistic regression
│   │   └── Xgboost.py            # XGBoost tree model
│
│   ├── notebook/
│   │   └── EDA.ipynb             # Exploratory Data Analysis notebook
│
│   └── visualization/
│       ├── DNN/
│       ├── Ensemble/
│       ├── LogisticRegression/
│       ├── XgBoost/
│       └── plot_utils.py         # Utility functions for plotting
│
├── patient_risk_interface.py     # GUI for real-time mortality risk prediction
│
├── col_description.md            # Complete data dictionary (in Chinese)
├── requirements.txt              # Python dependency list
├── README.md                     # Project documentation (this file)
├── .gitignore                    # Git ignore config
└── .venv/                        # Local Python virtual environment
```

* * *

📊 Dataset Description
----------------------

The dataset originates from a Kaggle ICU mortality prediction dataset. Full feature descriptions are in `col_description.md`. Key data types:

*   **Demographics**: Age, gender, ethnicity, height, weight, BMI
*   **ICU Info**: Admission source, ICU type, stay type
*   **Vitals**: Heart rate, blood pressure, temperature, oxygen saturation
*   **Scores**: GCS (eye, motor, verbal), APACHE II & III scores
*   **Comorbidities**: Diabetes, liver failure, AIDS, leukemia, cancer
*   **Target variable**: `hospital_death` (1=death, 0=survival)

💡 Models Implemented
---------------------

| Model | File | Description |
| --- | --- | --- |
| Logistic Regression | `LogisticRegression.py` | Linear baseline classifier |
| XGBoost | `Xgboost.py` | Gradient boosting decision trees |
| DNN | `DNN.py` | Multi-layer perceptron (MLP) |
| Stacking Ensemble | `Ensemble.py` | Combines DNN, LR, XGB via meta-learner |
| Soft Voting | `Ensemble.py` | Weighted average of predicted probabilities |

* * *

⚙️ Setup & Installation
-----------------------

### Step 1: Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

* * *

🚀 Running Instructions
-----------------------

### 📌 1. Run correlation analysis

```bash
python src/correlation_analysis/categorical_analysis.py
python src/correlation_analysis/numerical_analysis.py
```

### 📌 2. Explore data (optional)

Open `notebook/EDA.ipynb` to explore missing values, distributions, etc.

* * *

### 📌 3. Train individual models

```bash
python src/model/Xgboost.py
python src/model/LogisticRegression.py
python src/model/DNN.py
```

Outputs:

*   Model files saved in `Model/`
*   Visualization saved in `visualization/`
    

### 📌 4. Train ensemble model

```bash
python src/model/Ensemble.py
```

This will:

*   Perform 5-fold cross-validation
*   Save base and meta models
*   Output ensemble metrics
    

### 📌 5. Run GUI interface

```bash
streamlit patient_risk_interface.py
```

GUI Features:

*   Input patient features
*   Click “Predict”
*   View mortality risk (0–1 probability)
    

📈 Visualizations
-----------------

Generated plots are saved under `visualization/`:

*   Confusion Matrix
*   ROC Curves
*   F1 Score comparison
*   Feature Importance bar charts
    

🧰 Requirements
---------------

```txt
numpy
pandas
scikit-learn
xgboost
tensorflow
keras
matplotlib
seaborn
joblib
tkinter
```

Install all via:

```bash
pip install -r requirements.txt
```

* * *

🧠 Highlights
-------------

*   ✅ Modular, clean architecture
*   ✅ Reusable scripts for training & testing
*   ✅ Real-time risk prediction UI
*   ✅ Transparent feature documentation
*   ✅ Log files to trace analysis steps
    

🧑‍💻 Authors & Credits
-----------------------

> Group Project from **\[LingNan University\]**  
> Group Members: Yao HaoYang,Fang Zhou,Chen ZheHan,Liu ZhenTao,Yang Chuang,Xie Jie
