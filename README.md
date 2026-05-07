# aLZfusion: A Neuro-Fusion Stacked Ensemble Framework for Alzheimer's Disease Risk Prediction

## Overview
This repository contains the full implementation of the aLZfusion 
framework proposed in:

aLZfusion is a two-layer stacked ensemble model that integrates 
multimodal biomarkers: MRI, PET, DTI, CSF, cognitive scores, 
genetics, and demographics — from the ADNI dataset to classify 
subjects across five diagnostic categories: CN, SMC, EMCI, LMCI, 
and AD, with secondary prediction of MCI-to-AD conversion.

---

## Key Results

| Model | CV | Accuracy (%) | Std (±) | AUC | AUC Std (±) |
|---|---|---|---|---|---|
| Standard Model | 10-fold | 96.70 | 1.06 | 0.9941 | 0.0049 |
| Proposed Model | 10-fold | 96.84 | 1.02 | 0.9928 | 0.0055 |
| Standard Model | 5-fold | 95.72 | 0.81 | 0.9890 | 0.0022 |
| Proposed Model | 5-fold | 96.14 | 1.02 | 0.9877 | 0.0028 |

Optimal classifier combination: **RF + XGBoost + Extra Trees + LDA**  
Selected via exhaustive power-set search over all 57 possible 
combinations of 6 base classifiers.

---

## Requirements
python >= 3.8
scikit-learn
xgboost
lightgbm
shap
pandas
numpy
matplotlib
joblib

Install all dependencies:
```bash
pip install scikit-learn xgboost lightgbm shap pandas numpy 
matplotlib joblib
```

---

## Data

Data used in this study were obtained from the Alzheimer's Disease 
Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). 
Researchers wishing to use the raw ADNI data must apply for access 
at the ADNI website. 

---

## How to Reproduce Results

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/aLZfusion-AD-Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. To retrain models, run:
notebooks/Modeling Fusion.ipynb

4. To load saved models and evaluate directly:
```python
import joblib
import pandas as pd

df = pd.read_csv('data/df_model.csv')
X = df.drop('converted', axis=1)
y = df['converted']

model = joblib.load('models/proposed_model_cv10.pkl')
predictions = model.predict(X)
```

---

## Contact
For questions regarding the dataset or methodology, please open 
a GitHub Issue or contact the author.
