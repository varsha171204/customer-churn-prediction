import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# ================================
# 2. LOAD DATA
# ================================
df = pd.read_excel(r"C:\Users\varsh\Desktop\churn\Telco_customer_churn.xlsx")

print("Dataset Loaded ✅")


# ================================
# 3. DATA CLEANING
# ================================
df.rename(columns={'Churn Value': 'Churn'}, inplace=True)

drop_cols = [
    'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
    'Lat Long', 'Latitude', 'Longitude',
    'Churn Label', 'Churn Score', 'CLTV', 'Churn Reason',
    'Total Charges'
]

df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

df.dropna(inplace=True)


# ================================
# 🔥 SAVE CLEANED DATA (FIXED)
# ================================
save_path = r"C:\Users\varsh\Desktop\churn\cleaned_churn_data.csv"

df.to_csv(save_path, index=False)

print("✅ Cleaned dataset saved at:")
print(save_path)

# Confirm file exists
print("File exists:", os.path.exists(save_path))


# ================================
# 4. ENCODING
# ================================
df = pd.get_dummies(df, drop_first=True)


# ================================
# 5. SPLIT DATA
# ================================
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ================================
# 6. SMOTE
# ================================
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)


# ================================
# 7. SCALING
# ================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ================================
# 8. MODELS
# ================================

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:,1]

print("\n🔹 Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_test, y_prob_lr))
print(confusion_matrix(y_test, y_pred_lr))


# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]

print("\n🔹 Random Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_prob_rf))
print(confusion_matrix(y_test, y_pred_rf))


# XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    n_estimators=200,
    max_depth=5
)

xgb.fit(X_train, y_train)

y_prob_xgb = xgb.predict_proba(X_test)[:,1]

# Custom threshold
threshold = 0.4
y_pred_xgb = (y_prob_xgb > threshold).astype(int)

print("\n🔥 XGBoost")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, y_prob_xgb))
print(confusion_matrix(y_test, y_pred_xgb))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_xgb))


# ================================
# 9. ROC CURVE
# ================================
fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)

plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.show()


# ================================
# DONE
# ================================
print("\n🎉 PROJECT COMPLETED")