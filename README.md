# Breast-Cancer-Prediction-Model
Developed ML models (classification/regression) to analyze patient data, optimizing treatment planning. Tools, Python, pandas, scikit-learn.


This project analyzes breast cancer patient data to predict mortality status using multiple machine learning models.

## Project Overview
This notebook implements three machine learning approaches to predict breast cancer patient mortality status:
1. Linear Regression
2. k-Nearest Neighbors (kNN)
3. Gaussian Naive Bayes

Key steps include:
- Comprehensive data cleaning and preprocessing
- Feature engineering
- Model training and evaluation
- Hyperparameter tuning

## Dataset
**Source**: Coursework Dataset (5DATA002W.2)  
**Records**: Original 4,000+ patient records  
**Features**: 14 clinical attributes including:
- Demographic (Age, Month_of_Birth)
- Clinical (T_Stage, N_Stage, 6th_Stage)
- Pathological (Tumor_Size, Differentiated)
- Hormonal (Estrogen_Status, Progesterone_Status)

Target variable: `Mortality_Status` (0=Dead, 1=Alive)

## Data Preprocessing
```python
# Key preprocessing steps:
df.drop(['Patient_ID', 'Sex', 'Occupation'], axis=1)  # Remove identifiers
df.dropna()  # Remove missing values

# Feature encoding:
df['Mortality_Status'] = df['Mortality_Status'].map({'dead':0, 'alive':1})
df['T_Stage'] = df['T_Stage'].map({'T1':0, 'T2':1, 'T3':2, 'T4':3})

# Outlier removal using IQR method:
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
df = df[(df['Age'] >= Q1-1.5*IQR) & (df['Age'] <= Q3+1.5*IQR)]