
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("fault_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print("\nTarget Variable Distribution (Fault Type):")
print(df["Fault Type"].value_counts())
print("\nProportion of each Fault Type:")
print(df["Fault Type"].value_counts(normalize=True))
df.drop(columns=["Fault ID"], inplace=True)

# ============================================================
# TARGET VARIABLE
# ============================================================

TARGET_COLUMN = "Fault Type"

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# STEP 3: Convert location into Latitude & Longitude
df["Latitude"] = df["Fault Location (Latitude, Longitude)"].apply(
    lambda x: float(x.strip("()").split(",")[0])
)

df["Longitude"] = df["Fault Location (Latitude, Longitude)"].apply(
    lambda x: float(x.strip("()").split(",")[1])
)

df.drop(
    columns=["Fault Location (Latitude, Longitude)"],
    inplace=True
)

# ============================================================
# LABEL ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Voltage/Current ratio
if ("Voltage" in X.columns) and ("Current" in X.columns):
    X["V_I_Ratio"] = X["Voltage (V)"] / (X["Current (A)"] + 1e-6)


if ("Temperature" in X.columns) and ("Power Load" in X.columns):
    X["Temp_Load_Ratio"] = (
        X["Temperature (°C)"] /
        (X["Power Load (MW)"] + 1e-6)
    )

if (Load_Current := "Load_Current") in X.columns:
    X["Load_Current"] = (
        X["Power Load (MW)"] /
        (X["Current (A)"] + 1e-6)
    )
# Power efficiency
if (
    "Power Load (MW)" in X.columns
    and "Voltage (V)" in X.columns
    and "Current (A)" in X.columns
):
    X["Power_Factor_Approx"] = (
        X["Power Load (MW)"] /
        ((X["Voltage (V)"] * X["Current (A)"]) + 1e-6)
    )
# Fault duration impact
if (
    "Duration of Fault (hrs)" in X.columns
    and "Down time (hrs)" in X.columns
):
    X["Downtime_Ratio"] = (
        X["Down time (hrs)"] /
        (X["Duration of Fault (hrs)"] + 1e-6)
    )

# ============================================================
# IDENTIFY COLUMN TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumeric Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)

# ============================================================
# PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ============================================================
# MODEL
# ============================================================

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(np.unique(y_encoded)),
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ============================================================
# TRAIN
# ============================================================

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Training completed!")

# ============================================================
# PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)

# EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

joblib.dump(
    pipeline,
    "fault_detection_model.pkl"
)

joblib.dump(
    label_encoder,
    "fault_label_encoder.pkl"
)

print("\nModel saved successfully!")

import pandas as pd
import joblib

model = joblib.load("fault_detection_model.pkl")
label_encoder = joblib.load("fault_label_encoder.pkl")

new_sample = pd.DataFrame({
    "Voltage (V)": [228],
    "Current (A)": [18],
    "Power Load (MW)": [4100],
    "Temperature (°C)": [42],
    "Wind Speed (km/h)": [10],
    "Weather Condition": ["Rainy"],
    "Maintenance Status": ["Pending"],
    "Component Health": ["Poor"],
    "Fault Location (Latitude, Longitude)": ["(17.3850, 78.4867)"],
    "Duration of Fault (hrs)": [45],
    "Down time (hrs)": [30]
})

# Re-apply feature engineering to new_sample
# Power efficiency
if (
    "Power Load (MW)" in new_sample.columns
    and "Voltage (V)" in new_sample.columns
    and "Current (A)" in new_sample.columns
):
    new_sample["Power_Factor_Approx"] = (
        new_sample["Power Load (MW)"] /
        ((new_sample["Voltage (V)"] * new_sample["Current (A)"]) + 1e-6)
    )
# Fault duration impact
if (
    "Duration of Fault (hrs)" in new_sample.columns
    and "Down time (hrs)" in new_sample.columns
):
    new_sample["Downtime_Ratio"] = (
        new_sample["Down time (hrs)"] /
        (new_sample["Duration of Fault (hrs)"] + 1e-6)
    )


prediction = model.predict(new_sample)

fault_type = label_encoder.inverse_transform(prediction)

print("Predicted Fault:", fault_type[0])

