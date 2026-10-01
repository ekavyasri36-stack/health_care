import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# HEALTHCARE DISEASE PREDICTION USING ML
# ==========================================

print("=" * 60)
print("🏥 HEALTHCARE DISEASE PREDICTION USING ML")
print("=" * 60)

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("healthcare_dataset.csv")

print("\n✅ Dataset loaded successfully!")
print("Dataset Shape:", df.shape)

# ==========================================
# 2. REMOVE UNNECESSARY COLUMNS
# ==========================================

columns_to_drop = [
    "Name",
    "Date of Admission",
    "Doctor",
    "Hospital",
    "Insurance Provider",
    "Room Number",
    "Discharge Date"
]

# Remove only columns that exist
df = df.drop(
    columns=[col for col in columns_to_drop if col in df.columns]
)

print("\n✅ Unnecessary columns removed.")

# ==========================================
# 3. TARGET COLUMN
# ==========================================

target_column = "Medical Condition"

if target_column not in df.columns:
    print(f"\n❌ '{target_column}' column not found!")
    print("\nAvailable columns:")
    print(df.columns.tolist())
    exit()

# Remove rows where target is missing
df = df.dropna(subset=[target_column])

# ==========================================
# 4. FEATURES AND TARGET
# ==========================================

X = df.drop(columns=[target_column])
y = df[target_column]

print("\n🎯 We are predicting:")
print(target_column)

print("\nPossible Disease Categories:")
print(y.unique())

# ==========================================
# 5. IDENTIFY COLUMN TYPES
# ==========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\n📊 Numerical Features:")
print(numeric_features)

print("\n📋 Categorical Features:")
print(categorical_features)

# ==========================================
# 6. PREPROCESSING
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ==========================================
# 7. CREATE ML MODEL PIPELINE
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ==========================================
# 8. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n📊 Dataset Split Completed!")
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# ==========================================
# 9. TRAIN MODEL
# ==========================================

print("\n🤖 Training Random Forest Model...")

pipeline.fit(X_train, y_train)

print("✅ Model training completed!")

# ==========================================
# 10. MAKE PREDICTIONS
# ==========================================

y_pred = pipeline.predict(X_test)

# ==========================================
# 11. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("📊 MODEL PERFORMANCE")
print("=" * 60)

print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%")

print("\n📋 Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\n🔢 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# 12. SAMPLE PREDICTIONS
# ==========================================

print("\n" + "=" * 60)
print("🔮 SAMPLE DISEASE PREDICTIONS")
print("=" * 60)

sample_data = X_test.head(5)
sample_predictions = pipeline.predict(sample_data)

for i, (actual, predicted) in enumerate(
    zip(y_test.head(5), sample_predictions),
    start=1
):
    print(f"\nPatient {i}")
    print("Actual Medical Condition:   ", actual)
    print("Predicted Medical Condition:", predicted)

print("\n" + "=" * 60)
print("🎉 HEALTHCARE DISEASE PREDICTION COMPLETED!")
print("=" * 60)

print("\n⚠️ Note: This project is for educational purposes only.")
print("It should not be used for real medical diagnosis.")