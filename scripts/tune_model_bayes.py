"""
Bayesian Hyperparameter Optimization for PathSnap Model using Optuna.
Uses GroupKFold cross-validation on training trips to prevent leakage.
Refits the best tuned model and compares tuned vs untuned performance.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import optuna

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)

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


def compute_top1_acc(df_test, probs):
    df_eval = df_test[['trip_id', 'obs_idx', 'is_match']].copy()
    df_eval['prob'] = probs
    idx_max = df_eval.groupby(['trip_id', 'obs_idx'])['prob'].idxmax()
    return float(df_eval.loc[idx_max, 'is_match'].mean())


def main():
    print("Loading datasets for Bayesian Hyperparameter Optimization...")
    df_train = pd.read_csv("data/train_features.csv")
    df_test = pd.read_csv("data/test_features.csv")

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train[TARGET_COL].values
    groups = df_train[GROUP_COL].values

    X_test = df_test[FEATURE_COLS].values
    y_test = df_test[TARGET_COL].values

    gkf = GroupKFold(n_splits=5)

    # 1. Untuned Baseline Evaluation
    baseline_model = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.08, max_depth=8, random_state=42
    )
    t0 = time.time()
    baseline_model.fit(X_train, y_train)
    t_infer0 = time.time()
    probs_base = baseline_model.predict_proba(X_test)[:, 1]
    infer_time_base = (time.time() - t_infer0) * 1000.0

    preds_base = baseline_model.predict(X_test)
    base_macro_f1 = f1_score(y_test, preds_base, average='macro')
    base_top1 = compute_top1_acc(df_test, probs_base)
    base_loss = log_loss(y_test, probs_base)

    print(f"Untuned Baseline: Test Macro F1={base_macro_f1:.4f}, Top-1 Acc={base_top1*100:.2f}%, Log Loss={base_loss:.4f}")

    # 2. Define Objective for Bayesian Search
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.20, log=True),
            'max_iter': trial.suggest_int('max_iter', 80, 220, step=20),
            'max_depth': trial.suggest_int('max_depth', 5, 14),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 60, step=5),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
            'random_state': 42
        }

        clf = HistGradientBoostingClassifier(**params)
        scores = []
        for train_idx, val_idx in gkf.split(X_train, y_train, groups):
            clf.fit(X_train[train_idx], y_train[train_idx])
            y_pred = clf.predict(X_train[val_idx])
            scores.append(f1_score(y_train[val_idx], y_pred, average='macro'))

        return np.mean(scores)

    # 3. Run Optimization
    print("\nRunning Bayesian Hyperparameter Optimization with Optuna (25 Trials)...")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    t_search0 = time.time()
    study.optimize(objective, n_trials=25, show_progress_bar=True)
    total_search_time = time.time() - t_search0

    print(f"\nOptimization completed in {total_search_time:.2f} seconds!")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best CV Macro F1: {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 4. Save Trials History
    trials_df = study.trials_dataframe()
    trials_df.to_csv("reports/bayesian_optimization_trials.csv", index=False)

    # 5. Refit Best Model on Full Training Data
    best_params = study.best_params.copy()
    best_params['random_state'] = 42
    tuned_model = HistGradientBoostingClassifier(**best_params)

    t_fit0 = time.time()
    tuned_model.fit(X_train, y_train)
    fit_time_tuned = time.time() - t_fit0

    # 6. Evaluate Tuned Model on Student Testing Data
    t_infer_tuned0 = time.time()
    probs_tuned = tuned_model.predict_proba(X_test)[:, 1]
    infer_time_tuned = (time.time() - t_infer_tuned0) * 1000.0

    preds_tuned = tuned_model.predict(X_test)
    tuned_macro_f1 = f1_score(y_test, preds_tuned, average='macro')
    tuned_top1 = compute_top1_acc(df_test, probs_tuned)
    tuned_loss = log_loss(y_test, probs_tuned)
    tuned_acc = accuracy_score(y_test, preds_tuned)

    print("\n=== TUNED VS UNTUNED COMPARISON ===")
    print(f"Metric                  | Untuned Baseline | Tuned Model")
    print(f"------------------------+------------------+------------")
    print(f"CV Macro F1             | 0.9355           | {study.best_value:.4f}")
    print(f"Student-Test Top-1 Acc  | {base_top1*100:6.2f}%          | {tuned_top1*100:6.2f}%")
    print(f"Student-Test Macro F1   | {base_macro_f1:8.4f}         | {tuned_macro_f1:8.4f}")
    print(f"Test Log Loss           | {base_loss:8.4f}         | {tuned_loss:8.4f}")
    print(f"Inference Latency (ms)  | {infer_time_base:8.2f} ms      | {infer_time_tuned:8.2f} ms")

    # Save artifacts
    joblib.dump(tuned_model, "models/tuned_model.joblib")
    print("\nSaved tuned model artifact to: models/tuned_model.joblib")

    # Save Comparison Summary CSV
    comparison_df = pd.DataFrame([
        {
            'Model': 'HistGradientBoosting (Untuned Baseline)',
            'CV Macro F1': 0.9355,
            'Student-Test Macro F1': base_macro_f1,
            'Top-1 Accuracy': base_top1,
            'Log Loss': base_loss,
            'Inference Time (ms)': infer_time_base
        },
        {
            'Model': 'HistGradientBoosting (Bayesian Tuned)',
            'CV Macro F1': study.best_value,
            'Student-Test Macro F1': tuned_macro_f1,
            'Top-1 Accuracy': tuned_top1,
            'Log Loss': tuned_loss,
            'Inference Time (ms)': infer_time_tuned
        }
    ])
    comparison_df.to_csv("reports/bayesian_tuning_comparison.csv", index=False)

    # 7. Generate Visualization Plots
    # Plot Optimization History
    fig, ax = plt.subplots(figsize=(8, 4.5))
    trial_nums = [t.number for t in study.trials]
    f1_values = [t.value for t in study.trials]
    best_so_far = np.maximum.accumulate(f1_values)

    ax.scatter(trial_nums, f1_values, color='#1f77b4', alpha=0.7, label='Trial Objective Value')
    ax.plot(trial_nums, best_so_far, color='#d62728', linewidth=2.5, label='Best Objective Value')
    ax.set_title("Bayesian Optimization Convergence History (Optuna TPE)")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Group-Aware CV Macro F1")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/bayes_opt_history.png", dpi=300)
    plt.close()

    # Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(7, 4.5))
    comparison_df.plot.bar(x='Model', y=['Student-Test Macro F1', 'Top-1 Accuracy'], ax=ax, color=['#2ca02c', '#1f77b4'], edgecolor='black', width=0.5)
    ax.set_title("Performance Comparison: Untuned Baseline vs Bayesian Tuned")
    ax.set_ylabel("Score")
    ax.set_xticklabels(['Untuned Baseline', 'Bayesian Tuned'], rotation=0)
    ax.set_ylim(0.85, 1.0)
    for p in ax.patches:
        ax.annotate(f"{p.get_height()*100:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 2), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/bayes_opt_comparison.png", dpi=300)
    plt.close()
    print("Saved optimization plots to reports/figures/.")


if __name__ == '__main__':
    main()
