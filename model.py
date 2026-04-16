# model.py

import pandas as pd
import numpy as np
import joblib
import os
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")


def train_and_save():
    print("=" * 55)
    print("  FraudShield — Model Training Pipeline")
    print("=" * 55)

    if not os.path.exists("creditcard.csv"):
        raise FileNotFoundError(
            "\n❌ creditcard.csv not found!\n"
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "Place it in the same folder as model.py"
        )

    print("\n📂 Loading dataset...")


    with open(r"C:\Credit Card Fraud Detection\archive (4)\creditcard.csv", 'rb') as f:
        content = f.read()
    df = pd.read_csv(io.BytesIO(content))
    print(f"   Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    print(f"   Fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    print(f"   Legit: {(df['Class']==0).sum():,}")

    print("\n⚙️  Engineering features...")

    df["Hour"] = (df["Time"] / 3600 % 24).astype(int)

    df["Log_Amount"] = np.log1p(df["Amount"])

    df["Amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

    feature_cols = (
        [f"V{i}" for i in range(1, 29)]   
        + ["Log_Amount", "Amount_zscore", "Hour"]
    )

    X = df[feature_cols]
    y = df["Class"]

    print("\n✂️  Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("📏 Scaling features (RobustScaler)...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print("⚖️  Applying SMOTE to balance classes...")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
    print(f"   After SMOTE — Fraud: {y_res.sum():,}  Legit: {(y_res==0).sum():,}")

    print("\n🤖 Training Ensemble (XGBoost + RandomForest + LR)...")

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        C=0.01
    )

    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf), ("lr", lr)],
        voting="soft",
        weights=[3, 2, 1]        
    )

    ensemble.fit(X_res, y_res)
    print("   ✅ Training complete!")

    print("\n📊 Evaluating on test set...")
    y_pred  = ensemble.predict(X_test_scaled)
    y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

    roc    = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    prec   = precision_score(y_test, y_pred)
    rec    = recall_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred)

    print(f"\n   ROC-AUC  : {roc:.4f}")
    print(f"   PR-AUC   : {pr_auc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   F1 Score : {f1:.4f}")
    print("\n" + classification_report(y_test, y_pred,
          target_names=["Legit", "Fraud"]))

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "roc_auc":   round(roc, 4),
        "pr_auc":    round(pr_auc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "feature_cols": feature_cols,
        "total_train": len(X_train),
        "total_test":  len(X_test),
        "fraud_train": int(y_train.sum()),
        "fraud_test":  int(y_test.sum()),
    }

    print("\n💾 Saving model, scaler, metrics...")
    joblib.dump(ensemble, "fraud_model.pkl")
    joblib.dump(scaler,   "scaler.pkl")
    joblib.dump(metrics,  "metrics.pkl")

    print("\n✅ Done! Files saved:")
    print("   fraud_model.pkl")
    print("   scaler.pkl")
    print("   metrics.pkl")
    print("\n🚀 Now run: streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    train_and_save()