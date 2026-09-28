"""
Retrain Twitter model using twitter_profiles.csv (10K rows, label column).
Features map exactly to what 2.py expects in 13-element input vector.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

print("Loading twitter_profiles.csv...")
tw = pd.read_csv("datasets/twitter_profiles.csv")
print(f"Rows: {len(tw)} | Fake(1): {tw['label'].sum()} | Real(0): {(tw['label']==0).sum()}")

now = datetime.now()

def account_age_days(created_at):
    try:
        d = pd.to_datetime(created_at)
        return (now - d.to_pydatetime().replace(tzinfo=None)).days
    except:
        return 365

def screen_name_randomness(name):
    """Ratio of digits and underscores in screen_name -> higher = more bot-like"""
    if not isinstance(name, str) or len(name) == 0:
        return 0.0
    return len(re.findall(r'[\d_]', name)) / len(name)

tw['age_days'] = tw['created_at'].apply(account_age_days)
tw['statuses']  = tw['post_count'].fillna(0).astype(int)
tw['followers'] = tw['followers_count'].fillna(0).astype(int)
tw['friends']   = tw['friends_count'].fillna(0).astype(int)
tw['bio_len']   = tw['description'].fillna('').apply(len)
tw['ratio']     = tw['followers'] / (tw['friends'] + 1)
tw['posts_age'] = tw['statuses'] / (tw['age_days'] + 1)
tw['default_img'] = tw['default_profile_image'].fillna(0).astype(int)
tw['verified_int'] = tw['verified'].fillna(0).astype(int)
tw['sn_random'] = tw['screen_name'].apply(screen_name_randomness)
tw['geo'] = 0  # placeholder
tw['favs'] = 0
tw['listed'] = 0

# 13 features matching 2.py's twitter prediction input
feature_cols = [
    'statuses',    # 0 total posts/tweets
    'followers',   # 1
    'friends',     # 2
    'favs',        # 3 (0 proxy)
    'listed',      # 4 (0 proxy)
    'default_img', # 5
    'sn_random',   # 6 (not default_img exactly but close proxy)
    'geo',         # 7
    'verified_int',# 8
    'verified_int',# 9 (double slot as in original)
    'age_days',    # 10
    'bio_len',     # 11
    'ratio',       # 12
]

X = tw[feature_cols].fillna(0).values
y = tw['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Try Random Forest
rf = RandomForestClassifier(
    n_estimators=300, max_depth=15, random_state=42,
    class_weight='balanced', min_samples_leaf=5
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nRandom Forest Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# Try Gradient Boosting too
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)
print(f"\nGradient Boosting Accuracy: {acc_gb:.4f}")
print(classification_report(y_test, y_pred_gb, target_names=["Real", "Fake"]))

# Pick best model
best = rf if acc >= acc_gb else gb
best_acc = max(acc, acc_gb)
print(f"\nSaving best Twitter model (acc={best_acc:.4f})...")
joblib.dump(best, "twitter_fake_detector1.pkl")
print("Saved: twitter_fake_detector1.pkl")
print(f"n_features: {best.n_features_in_}")
