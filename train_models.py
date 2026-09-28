"""
Train all three fake profile detection models from real datasets.
Saves: instagram_fake_detector.pkl, twitter_fake_detector1.pkl,
       best_fake_profile_model.pkl, scaler.pkl, platform_encoder.pkl
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# 1. INSTAGRAM MODEL
# ==========================================================
print("=" * 60)
print("Training Instagram Fake Profile Detector...")
print("=" * 60)

# Load train + test
inst_train = pd.read_csv("datasets/inst_train.csv")
inst_test = pd.read_csv("datasets/inst_test.csv")
inst_all = pd.concat([inst_train, inst_test], ignore_index=True)

print(f"Instagram rows: {len(inst_all)} | Fake: {inst_all['fake'].sum()} | Real: {(inst_all['fake']==0).sum()}")

# Features match 2.py usage order:
# profile_pic, username_ratio, fullname_words, fullname_ratio,
# name_match, bio_len, ext_url, is_private, posts, followers, following
insta_feature_cols = [
    "profile pic",
    "nums/length username",
    "fullname words",
    "nums/length fullname",
    "name==username",
    "description length",
    "external URL",
    "private",
    "#posts",
    "#followers",
    "#follows",
]

X_insta = inst_all[insta_feature_cols].fillna(0).values
y_insta = inst_all["fake"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_insta, y_insta, test_size=0.2, random_state=42, stratify=y_insta
)

insta_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'
)
insta_model.fit(X_train, y_train)
y_pred = insta_model.predict(X_test)
print(f"Instagram Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

joblib.dump(insta_model, "instagram_fake_detector.pkl")
print("Saved: instagram_fake_detector.pkl\n")


# ==========================================================
# 2. TWITTER BOT MODEL
# ==========================================================
print("=" * 60)
print("Training Twitter Bot Detector...")
print("=" * 60)

tw = pd.read_csv("datasets/twitter_bot_detection_data.csv")
print(f"Twitter (bot detection) rows: {len(tw)} | Bot: {tw['Bot Label'].sum()} | Human: {(tw['Bot Label']==0).sum()}")

# Derive features used in 2.py:
# statuses(total tweets proxy), followers, friends(following), favs,
# listed, !default_img, default_img, geo, 0, verified, age, bio_len, ratio
# twitter_bot_detection_data has: Follower Count, Verified, Bot Label, Created At

from datetime import datetime
now = datetime.now()

def parse_age(row):
    try:
        created = pd.to_datetime(row['Created At'])
        return (now - created.to_pydatetime().replace(tzinfo=None)).days
    except:
        return 365  # default 1 year

tw['age_days'] = tw.apply(parse_age, axis=1)
tw['followers'] = tw['Follower Count'].fillna(0).astype(int)
tw['friends'] = (tw['Follower Count'] * 0.8).astype(int)  # estimate
tw['statuses'] = (tw['Retweet Count'] * 10 + tw['Mention Count'] * 5).fillna(0).astype(int)
tw['favs'] = (tw['Retweet Count'] * 3).fillna(0).astype(int)
tw['listed'] = 0
tw['verified_int'] = (tw['Verified'] == True).astype(int)
tw['default_img'] = 0
tw['geo'] = 0
tw['bio_len'] = 50
tw['ratio'] = tw['followers'] / (tw['friends'] + 1)

twitter_feature_cols = [
    'statuses', 'followers', 'friends', 'favs', 'listed',
    'verified_int',  # not default_img proxy
    'default_img',
    'geo',
    'verified_int',  # duplicate slot (matches 2.py 13-feature vector)
    'verified_int',
    'age_days',
    'bio_len',
    'ratio',
]

X_tw = tw[['statuses', 'followers', 'friends', 'favs', 'listed',
           'verified_int', 'default_img', 'geo',
           'default_img',  # slot 8
           'verified_int',  # slot 9
           'age_days', 'bio_len', 'ratio']].fillna(0).values
y_tw = tw['Bot Label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_tw, y_tw, test_size=0.2, random_state=42, stratify=y_tw
)

tw_model = RandomForestClassifier(
    n_estimators=200, max_depth=12, random_state=42, class_weight='balanced'
)
tw_model.fit(X_train, y_train)
y_pred = tw_model.predict(X_test)
print(f"Twitter Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Human", "Bot"]))

joblib.dump(tw_model, "twitter_fake_detector1.pkl")
print("Saved: twitter_fake_detector1.pkl\n")


# ==========================================================
# 3. GENERAL SOCIAL MEDIA MODEL
# ==========================================================
print("=" * 60)
print("Training General Social Media Fake Profile Detector...")
print("=" * 60)

gen = pd.read_csv("datasets/fake_social_media.csv")
print(f"General dataset rows: {len(gen)} | Fake: {gen['is_fake'].sum()} | Real: {(gen['is_fake']==0).sum()}")

# Encode platform
le = LabelEncoder()
gen['platform_encoded'] = le.fit_transform(gen['platform'])

# Features match 2.py usage (17 features):
# platform_encoded, has_pic, bio_len, 0.5, followers, following, ratio,
# age_days, posts, posts/age, 0.5, 0.5, 0, 0, 0, 0, verified
gen['ratio'] = gen['followers'] / (gen['following'] + 1)
gen['posts_per_day'] = gen['posts'] / (gen['account_age_days'] + 1)

general_feature_cols = [
    'platform_encoded',
    'has_profile_pic',
    'bio_length',
    'username_randomness',
    'followers',
    'following',
    'follower_following_ratio',
    'account_age_days',
    'posts',
    'posts_per_day',
    'caption_similarity_score',
    'content_similarity_score',
    'follow_unfollow_rate',
    'spam_comments_rate',
    'generic_comment_rate',
    'suspicious_links_in_bio',
    'verified',
]

X_gen = gen[general_feature_cols].fillna(0).values
y_gen = gen['is_fake'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_gen, y_gen, test_size=0.2, random_state=42, stratify=y_gen
)

# Fit scaler on training data
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

gen_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
)
gen_model.fit(X_train_sc, y_train)
y_pred = gen_model.predict(X_test_sc)
print(f"General Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

joblib.dump(gen_model, "best_fake_profile_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "platform_encoder.pkl")
print("Saved: best_fake_profile_model.pkl, scaler.pkl, platform_encoder.pkl\n")


print("=" * 60)
print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("  instagram_fake_detector.pkl  -> n_features:", insta_model.n_features_in_)
print("  twitter_fake_detector1.pkl   -> n_features:", tw_model.n_features_in_)
print("  best_fake_profile_model.pkl  -> n_features:", gen_model.n_features_in_)
print("=" * 60)
