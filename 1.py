nbimport pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

# ==========================================
# 1️⃣ LOAD DATA
# ==========================================

df = pd.read_csv(r"C:\Users\koran\bharath\codes\vara\train\datasets\fake_social_media.csv")   # change filename

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================================
# 2️⃣ SEPARATE FEATURES & TARGET
# ==========================================

X = df.drop("is_fake", axis=1)
y = df["is_fake"]

# Encode platform if categorical
if X["platform"].dtype == "object":
    le_platform = LabelEncoder()
    X["platform"] = le_platform.fit_transform(X["platform"])
    joblib.dump(le_platform, "platform_encoder.pkl")

# ==========================================
# 3️⃣ SCALE FEATURES
# ==========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

# ==========================================
# 4️⃣ TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 5️⃣ TRAIN MULTIPLE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# ==========================================
# 6️⃣ SAVE BEST MODEL
# ==========================================

joblib.dump(best_model, "best_fake_profile_model.pkl")

print("\n===================================")
print("Best Model:", best_model_name)
print("Best Accuracy:", best_accuracy)
print("Model saved as best_fake_profile_model.pkl")


import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# ==========================================
# 1️⃣ LOAD DATA
# ==========================================

df = pd.read_csv(r"C:\Users\koran\bharath\codes\vara\train\datasets\inst_train.csv")  # change filename

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================================
# 2️⃣ RENAME COLUMNS (REMOVE SPECIAL CHARS)
# ==========================================

df.columns = [
    "profile_pic",
    "nums_length_username",
    "fullname_words",
    "nums_length_fullname",
    "name_equals_username",
    "description_length",
    "external_url",
    "private",
    "posts",
    "followers",
    "follows",
    "fake"
]

# ==========================================
# 3️⃣ FEATURES & TARGET
# ==========================================

X = df.drop("fake", axis=1)
y = df["fake"]

# ==========================================
# 4️⃣ SCALE DATA
# ==========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

# ==========================================
# 5️⃣ TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 6️⃣ TRAIN MULTIPLE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# ==========================================
# 7️⃣ SAVE BEST MODEL
# ==========================================

joblib.dump(best_model, "instagram_fake_detector.pkl")

print("\n===================================")
print("Best Model:", best_model_name)
print("Best Accuracy:", best_accuracy)
print("Model saved as instagram_fake_detector.pkl")

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1️⃣ LOAD DATA
# ==========================================

df = pd.read_csv(r"C:\Users\koran\bharath\codes\vara\train\datasets\twitter_bot_detection_data.csv")  # change filename

print("Dataset Shape:", df.shape)

# ==========================================
# 2️⃣ FEATURE ENGINEERING
# ==========================================

# Convert created_at to account age (days)
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df["account_age_days"] = (datetime.now() - df["created_at"]).dt.days

# Description length
df["description_length"] = df["description"].fillna("").apply(len)

# Follower/Following ratio
df["follower_friend_ratio"] = df["followers_count"] / (df["friends_count"] + 1)

# ==========================================
# 3️⃣ SELECT IMPORTANT NUMERIC FEATURES
# ==========================================

selected_features = [
    "statuses_count",
    "followers_count",
    "friends_count",
    "favourites_count",
    "listed_count",
    "default_profile",
    "default_profile_image",
    "geo_enabled",
    "protected",
    "verified",
    "account_age_days",
    "description_length",
    "follower_friend_ratio"
]

X = df[selected_features]

# ==========================================
# 4️⃣ TARGET COLUMN
# ==========================================

y = df["dataset"]   # real / bot

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

joblib.dump(le, "twitter_label_encoder.pkl")

# ==========================================
# 5️⃣ HANDLE MISSING VALUES
# ==========================================

X = X.fillna(0)

# ==========================================
# 6️⃣ SCALE FEATURES
# ==========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "twitter_scaler.pkl")

# ==========================================
# 7️⃣ TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 8️⃣ TRAIN MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# 9️⃣ EVALUATE
# ==========================================

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================================
# 🔟 SAVE MODEL
# ==========================================

joblib.dump(model, "twitter_fake_detector.pkl")

print("\n✅ Model saved as twitter_fake_detector.pkl")



import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1️⃣ LOAD DATA
# ==========================================

df = pd.read_csv(r"C:\Users\koran\bharath\codes\vara\train\datasets\twitter_profiles_original.csv")   # change filename
print("Dataset Shape:", df.shape)

# ==========================================
# 2️⃣ FEATURE ENGINEERING
# ==========================================

# Convert created_at to datetime
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# Account age in days
df["account_age_days"] = (datetime.now() - df["created_at"]).dt.days

# Description length
df["description_length"] = df["description"].fillna("").apply(len)

# Follower/Following ratio
df["follower_friend_ratio"] = df["followers_count"] / (df["friends_count"] + 1)

# ==========================================
# 3️⃣ SELECT IMPORTANT FEATURES
# ==========================================

features = [
    "followers_count",
    "friends_count",
    "post_count",
    "default_profile_image",
    "profile_use_background_image",
    "verified",
    "account_age_days",
    "description_length",
    "follower_friend_ratio"
]

X = df[features]
y = df["label"]

# ==========================================
# 4️⃣ HANDLE MISSING VALUES
# ==========================================

X = X.fillna(0)

# Encode label if string
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)
    joblib.dump(le, "label_encoder.pkl")

# ==========================================
# 5️⃣ SCALE FEATURES
# ==========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

# ==========================================
# 6️⃣ TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 7️⃣ TRAIN MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# 8️⃣ EVALUATE
# ==========================================

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================================
# 9️⃣ SAVE MODEL
# ==========================================

joblib.dump(model, "twitter_fake_detector1.pkl")

print("\n✅ Model saved as twitter_fake_detector1.pkl")
print("Training completed successfully!")