import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from joblib import dump, load
import numpy as np

# Load dataset
housing = pd.read_csv("data.csv")

# Initial exploration
housing.head()
housing.info()
housing.describe()

# Train-test split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(len(train_set))
print(len(test_set))

# Stratified Shuffle Split based on 'CHAS'
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS ']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing = strat_train_set.copy()

# Visualizing correlations
attributes = ["MEDV", "RM", "ZN ", "LSTAT"]
scatter_matrix(housing[attributes], figsize=(12, 8))
housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)

# Feature engineering
housing["TAXRM"] = housing["TAX"]/housing["RM"]
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)

# Prepare data for machine learning algorithms
housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

# Impute missing values
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)
X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns=housing.columns)

# Pipeline for numerical attributes
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])
housing_num_tr = my_pipeline.fit_transform(housing)

# Select and train a model
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

# Model evaluation using cross-validation
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

def print_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

print_scores(rmse_scores)

# Save the model
dump(model, 'Dragon.joblib')

# Testing the model on test data
X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(Y_test))
print("Final RMSE:", final_rmse)

# Using the saved model for prediction
model = load('Dragon.joblib')
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
                      -0.24641278, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
                      -0.97491834,  0.41164221, -0.86091034]])
predicted_value = model.predict(features)
print("Predicted value:", predicted_value)
