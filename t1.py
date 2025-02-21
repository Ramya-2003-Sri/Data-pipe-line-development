import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load Data
df = pd.read_csv("data.csv")

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Define preprocessing steps
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),  # Fill missing values
    ("scaler", StandardScaler())  # Scale numerical data
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values
    ("encoder", OneHotEncoder(handle_unknown="ignore"))  # One-hot encoding
])

# Combine both pipelines
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Apply transformations
X = df.drop(columns=["target"])  # Assuming 'target' is the label column
y = df["target"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the preprocessor on training data and transform
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Save processed data
pd.DataFrame(X_train_transformed).to_csv("X_train_preprocessed.csv", index=False)
pd.DataFrame(X_test_transformed).to_csv("X_test_preprocessed.csv", index=False)
pd.DataFrame(y_train).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

print("Data preprocessing pipeline completed successfully.")
