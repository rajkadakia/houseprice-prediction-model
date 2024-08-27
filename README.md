# houseprice-prediction-model
a ml model that predicts the price of a house based on provided data
## Installation

### Requirements

- Python 3.x
- pandas
- scikit-learn
- numpy
- joblib
- matplotlib

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/boston-housing-prediction.git
   pip install -r requirements.txt
boston-housing-prediction/
│
├── data.csv                    # Dataset used for training and testing
├── boston_housing.ipynb        # Jupyter Notebook with the full workflow
├── main.py                     # Python script with the core implementation
├── Dragon.joblib               # Saved RandomForest model
└── README.md                   # This readme file

Data Preprocessing
Initial Exploration: The dataset is loaded and basic exploratory analysis is performed using pandas. This includes viewing the first few rows, checking for null values, and summarizing statistics.
Train-Test Split: The data is split into training and testing sets. Stratified sampling is used based on the CHAS variable to ensure that the train and test sets have similar distributions of this categorical attribute.
Feature Engineering: A new feature TAXRM is created by dividing TAX by RM to explore potential relationships with MEDV.
Data Imputation: Missing values are imputed using the median of each feature.
Pipeline: A data processing pipeline is created using scikit-learn to handle imputation and feature scaling.

Three regression models were considered:

Linear Regression
Decision Tree Regression
Random Forest Regression (chosen model)
The RandomForestRegressor was chosen based on its performance during cross-validation.
Predicting with the Model
To make predictions with the trained model:

python
Copy code
from joblib import load
import numpy as np

model = load('Dragon.joblib')
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
                      -0.24641278, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
                      -0.97491834,  0.41164221, -0.86091034]])
predicted_value = model.predict(features)
print("Predicted value:", predicted_value)
Replace features with your data to get predictions.

