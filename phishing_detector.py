"""
phishing_detector.py
Phishing Email Detection Model using Scikit-learn.

Pipeline:
1. Load dataset of labeled emails (phishing / safe).
2. Extract features:
   - TF-IDF on the raw email text (captures keywords, phrasing).
   - Hand-engineered features (URL count, IP-based URLs, suspicious
     keywords, HTTPS presence, exclamation marks, urgency words, etc.)
3. Combine features and train a classifier (Logistic Regression).
4. Evaluate: accuracy, classification report, confusion matrix.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
df = pd.read_csv("emails_dataset.csv")
print(f"Loaded {len(df)} emails.")
print(df["label"].value_counts(), "\n")

SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "suspend", "account", "click", "password",
    "confirm", "limited", "act now", "winner", "prize", "bank",
    "security alert", "update your", "claim", "expire", "immediately",
    "restricted", "unusual activity",
]

URL_REGEX = re.compile(r"https?://[^\s]+")
IP_URL_REGEX = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")


def extract_features(text: str) -> dict:
    text_lower = text.lower()
    urls = URL_REGEX.findall(text)
    return {
        "num_urls": len(urls),
        "has_ip_url": int(bool(IP_URL_REGEX.search(text))),
        "has_https": int(any(u.startswith("https") for u in urls)),
        "num_exclaim": text.count("!"),
        "num_suspicious_kw": sum(kw in text_lower for kw in SUSPICIOUS_KEYWORDS),
        "text_length": len(text),
        "num_digits": sum(c.isdigit() for c in text),
        "has_dollar": int("$" in text),
    }


# ----------------------------------------------------------------------
# 2. Feature extraction
# ----------------------------------------------------------------------
engineered = pd.DataFrame([extract_features(t) for t in df["text"]])
X_train_text, X_test_text, X_train_eng, X_test_eng, y_train, y_test = train_test_split(
    df["text"], engineered, df["label"],
    test_size=0.25, random_state=42, stratify=df["label"]
)

tfidf = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

scaler = StandardScaler()
X_train_eng_scaled = scaler.fit_transform(X_train_eng)
X_test_eng_scaled = scaler.transform(X_test_eng)

X_train = hstack([X_train_tfidf, csr_matrix(X_train_eng_scaled)])
X_test = hstack([X_test_tfidf, csr_matrix(X_test_eng_scaled)])

# ----------------------------------------------------------------------
# 3. Train model
# ----------------------------------------------------------------------
model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 4. Evaluate
# ----------------------------------------------------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

labels = ["phishing", "safe"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nSaved confusion matrix plot to confusion_matrix.png")

# ----------------------------------------------------------------------
# 5. Quick manual test on new example emails
# ----------------------------------------------------------------------
def predict_email(text: str) -> str:
    tfidf_vec = tfidf.transform([text])
    eng_vec = scaler.transform(pd.DataFrame([extract_features(text)]))
    combined = hstack([tfidf_vec, csr_matrix(eng_vec)])
    return model.predict(combined)[0]


examples = [
    "URGENT: Your account has been suspended. Verify now at http://192.168.1.5/login to restore access!",
    "Hi Alex, attached is the agenda for tomorrow's meeting. See you then. Best, Sarah",
]

print("\n--- Manual test predictions ---")
for e in examples:
    print(f"'{e[:60]}...' -> {predict_email(e)}")
