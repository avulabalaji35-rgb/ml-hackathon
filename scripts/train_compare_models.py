"""
Model Training, Cross-Validation, and Benchmark Comparison across 5 distinct Algorithm Families
for the PathSnap Hackathon.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

FEATURE_COLS = [
    'dist_to_segment', 'heading_diff', 'dist_to_u', 'dist_to_v',
    'segment_length', 'proj_ratio', 'gps_speed_mps', 'speed_length_ratio',
    'node_in_degree_u', 'node_out_degree_v', 'spatial_rank'
]
TARGET_COL = 'is_match'
GROUP_COL = 'trip_id'


def compute_top1_accuracy(df_test, y_pred_probs):
    """
    Computes observation-level Top-1 accuracy:
    For each (trip_id, obs_idx), picks candidate with highest predicted probability,
    and checks if that candidate has is_match == 1.
    """
    df_eval = df_test[['trip_id', 'obs_idx', 'arc_id', 'is_match']].copy()
    df_eval['prob'] = y_pred_probs
    
    # Identify candidate with max prob for each observation
    idx_max = df_eval.groupby(['trip_id', 'obs_idx'])['prob'].idxmax()
    top1_picks = df_eval.loc[idx_max]
    
    # Top 1 accuracy is fraction of top picks that are the true match
    top1_acc = top1_picks['is_match'].mean()
    return float(top1_acc)


def main():
    print("Loading extracted feature sets...")
    df_train = pd.read_csv("data/train_features.csv")
    df_test = pd.read_csv("data/test_features.csv")

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train[TARGET_COL].values
    groups_train = df_train[GROUP_COL].values

    X_test = df_test[FEATURE_COLS].values
    y_test = df_test[TARGET_COL].values

    print(f"Train data: {X_train.shape}, Test data: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.3f}, Test positive rate: {y_test.mean():.3f}")

    # Define at least 5 models from distinct algorithm families
    models = {
        'Logistic Regression': {
            'family': 'Linear & Probabilistic',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
            ]),
            'config': 'C=1.0, penalty=l2, solver=lbfgs'
        },
        'Decision Tree': {
            'family': 'Tree-based',
            'pipeline': Pipeline([
                ('clf', DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=42))
            ]),
            'config': 'max_depth=8, min_samples_split=10'
        },
        'Random Forest': {
            'family': 'Ensemble',
            'pipeline': Pipeline([
                ('clf', RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=6, random_state=42, n_jobs=-1))
            ]),
            'config': 'n_estimators=100, max_depth=12'
        },
        'HistGradientBoosting': {
            'family': 'Boosting',
            'pipeline': Pipeline([
                ('clf', HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_depth=8, random_state=42))
            ]),
            'config': 'max_iter=100, lr=0.08, max_depth=8'
        },
        'MLP Neural Network': {
            'family': 'Neural Network',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=250, alpha=0.01, random_state=42, early_stopping=True))
            ]),
            'config': 'hidden=(64, 32), alpha=0.01, early_stopping'
        }
    }

    results = []
    trained_estimators = {}
    gkf = GroupKFold(n_splits=5)

    print("\n" + "="*80)
    print("Beginning 5-Fold Group-Aware Cross Validation & Benchmark Evaluation")
    print("="*80)

    for name, item in models.items():
        print(f"\nEvaluating: {name} (Family: {item['family']})")
        pipe = item['pipeline']

        # 1. Group-aware 5-fold cross validation
        cv_f1s = []
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups_train)):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_va, y_va = X_train[val_idx], y_train[val_idx]

            pipe_clone = joblib.load(joblib.dump(pipe, f"models/temp_{name.replace(' ', '_')}.pkl")[0])
            pipe_clone.fit(X_tr, y_tr)
            y_va_pred = pipe_clone.predict(X_va)
            cv_f1s.append(f1_score(y_va, y_va_pred, average='macro'))

        cv_f1_mean = np.mean(cv_f1s)
        cv_f1_std = np.std(cv_f1s)
        print(f"  CV Macro F1: {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")

        # 2. Fit on full training set
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_time = time.time() - t0
        trained_estimators[name] = pipe
        joblib.dump(pipe, f"models/{name.lower().replace(' ', '_')}_model.joblib")

        # 3. Test Set Evaluation
        t_infer0 = time.time()
        y_test_pred = pipe.predict(X_test)
        y_test_probs = pipe.predict_proba(X_test)[:, 1]
        infer_time = time.time() - t_infer0

        acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred, zero_division=0)
        rec = recall_score(y_test, y_test_pred, zero_division=0)
        macro_f1 = f1_score(y_test, y_test_pred, average='macro')
        loss = log_loss(y_test, y_test_probs)

        # Observation-level Top-1 accuracy
        top1_acc = compute_top1_accuracy(df_test, y_test_probs)

        print(f"  Test Accuracy:     {acc:.4f}")
        print(f"  Test Macro F1:     {macro_f1:.4f}")
        print(f"  Observation Top-1: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
        print(f"  Test Log Loss:     {loss:.4f}")
        print(f"  Train Time:        {fit_time:.2f} s | Infer Time: {infer_time*1000:.2f} ms")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Match', 'Match'])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f"Confusion Matrix: {name}")
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/cm_{name.lower().replace(' ', '_')}.png", dpi=200)
        plt.close()

        results.append({
            'Model': name,
            'Algorithm Family': item['family'],
            'Key Configuration': item['config'],
            'CV Macro F1': f"{cv_f1_mean:.4f} ± {cv_f1_std:.4f}",
            'CV Macro F1 Mean': cv_f1_mean,
            'Student-Test Macro F1': macro_f1,
            'Top-1 Accuracy': top1_acc,
            'Precision': prec,
            'Recall': rec,
            'Test Accuracy': acc,
            'Log Loss': loss,
            'Training Time (s)': fit_time,
            'Inference Time (ms)': infer_time * 1000.0
        })

    # Cleanup temp models
    for f in os.listdir("models"):
        if f.startswith("temp_"):
            try: os.remove(os.path.join("models", f))
            except: pass

    df_res = pd.DataFrame(results)
    res_path = "reports/model_comparison_results.csv"
    df_res.to_csv(res_path, index=False)
    print(f"\nSaved consolidated model comparison to: {res_path}")

    # Display comparison table
    print("\n=== CONSOLIDATED MODEL COMPARISON TABLE ===")
    display_cols = ['Model', 'Algorithm Family', 'CV Macro F1', 'Top-1 Accuracy', 'Student-Test Macro F1', 'Log Loss', 'Inference Time (ms)']
    print(df_res[display_cols].to_string(index=False))

    # Generate Model Comparison Plots
    # 1. Macro F1 & Top-1 Accuracy comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_res.plot.bar(x='Model', y=['CV Macro F1 Mean', 'Student-Test Macro F1'], ax=axes[0], color=['#1f77b4', '#aec7e8'], edgecolor='black', width=0.7)
    axes[0].set_title("Cross-Validation vs Student-Test Macro F1")
    axes[0].set_ylabel("Macro F1-Score")
    axes[0].set_xticklabels(df_res['Model'], rotation=25, ha='right')
    axes[0].set_ylim(0.7, 1.0)

    df_res.plot.bar(x='Model', y='Top-1 Accuracy', ax=axes[1], color='#2ca02c', edgecolor='black', width=0.5)
    axes[1].set_title("Observation-level Top-1 Accuracy (Test Set)")
    axes[1].set_ylabel("Top-1 Accuracy")
    axes[1].set_xticklabels(df_res['Model'], rotation=25, ha='right')
    axes[1].set_ylim(0.7, 1.0)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 2), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_comparison_performance.png", dpi=300)
    plt.close()

    # 2. Log Loss & Latency
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_res.plot.bar(x='Model', y='Log Loss', ax=axes[0], color='#ff7f0e', edgecolor='black', width=0.5)
    axes[0].set_title("Candidate Scoring Log Loss (Lower is Better)")
    axes[0].set_ylabel("Log Loss")
    axes[0].set_xticklabels(df_res['Model'], rotation=25, ha='right')

    df_res.plot.bar(x='Model', y='Inference Time (ms)', ax=axes[1], color='#9467bd', edgecolor='black', width=0.5)
    axes[1].set_title("Inference Latency on Test Set (ms)")
    axes[1].set_ylabel("Time (ms)")
    axes[1].set_xticklabels(df_res['Model'], rotation=25, ha='right')

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_comparison_loss_latency.png", dpi=300)
    plt.close()
    print("Saved comparison plots to reports/figures/.")

    # Identify best model based on Top-1 Accuracy and Macro F1
    best_model_name = df_res.sort_values(by=['Top-1 Accuracy', 'Student-Test Macro F1'], ascending=False).iloc[0]['Model']
    print(f"\nWinning Model Selected for Bayesian Optimization: {best_model_name}")


if __name__ == '__main__':
    main()
