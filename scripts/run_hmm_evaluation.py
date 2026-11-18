"""
Evaluation of HMM and Viterbi Sequential Route Reconstruction across Student-Testing Trips.
Compares Classifier-Only predictions against Classifier-plus-HMM decoded routes
for pointwise accuracy and topological sequence continuity.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath("."))
from src.data.dataset import PathSnapDataset
from src.hmm.decoder import MapMatchingHMM

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    print("Initializing MapMatchingHMM with tuned model...")
    hmm = MapMatchingHMM(model_or_path="models/tuned_model.joblib", beta=6.0, search_radius=35.0)

    test_ds = PathSnapDataset(".", partition="student_test")
    test_trips = test_ds.get_trip_ids()[:5] # Test on first 5 test trips
    print(f"Evaluating {len(test_trips)} student testing trips: {test_trips}")

    records = []
    for t_id in test_trips:
        print(f"\nDecoding Trip {t_id}...")
        trip = test_ds[t_id]
        res = hmm.decode_trip(trip, sample_step=4)
        if res is None:
            continue

        m = res['metrics']
        clf_acc = m.get('classifier_only_accuracy', 0.0)
        hmm_acc = m.get('hmm_viterbi_accuracy', 0.0)
        clf_cont = m.get('classifier_continuity', 0.0)
        hmm_cont = m.get('hmm_continuity', 0.0)
        rec_cont = m.get('reconstructed_route_continuity', 0.0)

        print(f"  Classifier Pointwise Acc: {clf_acc*100:.2f}% | Continuity: {clf_cont*100:.2f}%")
        print(f"  HMM Viterbi Pointwise Acc:{hmm_acc*100:.2f}% | Continuity: {hmm_cont*100:.2f}%")
        print(f"  Reconstructed Path Continuity: {rec_cont*100:.2f}% (Length: {m.get('reconstructed_route_length')} arcs)")

        records.append({
            'trip_id': t_id,
            'classifier_acc': clf_acc,
            'hmm_acc': hmm_acc,
            'acc_improvement': hmm_acc - clf_acc,
            'classifier_continuity': clf_cont,
            'hmm_continuity': hmm_cont,
            'reconstructed_continuity': rec_cont,
            'gt_route_length': len(res['ground_truth_route']),
            'reconstructed_length': m.get('reconstructed_route_length')
        })

    df_eval = pd.DataFrame(records)
    eval_csv = "reports/hmm_viterbi_evaluation.csv"
    df_eval.to_csv(eval_csv, index=False)
    print(f"\nSaved HMM evaluation results to: {eval_csv}")

    # Summary Table
    print("\n=== CLASSIFIER VS HMM ROUTE RECONSTRUCTION BENCHMARK ===")
    print(df_eval[['trip_id', 'classifier_acc', 'hmm_acc', 'classifier_continuity', 'hmm_continuity', 'reconstructed_continuity']].to_string(index=False))

    mean_clf_acc = df_eval['classifier_acc'].mean()
    mean_hmm_acc = df_eval['hmm_acc'].mean()
    mean_clf_cont = df_eval['classifier_continuity'].mean()
    mean_hmm_cont = df_eval['hmm_continuity'].mean()
    mean_rec_cont = df_eval['reconstructed_continuity'].mean()

    print(f"\nOverall Mean Accuracy:   Classifier = {mean_clf_acc*100:.2f}% | HMM Viterbi = {mean_hmm_acc*100:.2f}% (Δ = +{(mean_hmm_acc-mean_clf_acc)*100:.2f}%)")
    print(f"Overall Mean Continuity: Classifier = {mean_clf_cont*100:.2f}% | HMM Viterbi = {mean_hmm_cont*100:.2f}% | Reconstructed = {mean_rec_cont*100:.2f}%")

    # Generate Figures
    # 1. Accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    df_eval.plot.bar(x='trip_id', y=['classifier_acc', 'hmm_acc'], ax=ax, color=['#ff7f0e', '#2ca02c'], edgecolor='black', width=0.6)
    ax.set_title("Pointwise Observation Accuracy: Classifier vs Classifier + HMM")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Trip ID")
    ax.set_xticklabels(df_eval['trip_id'], rotation=0)
    ax.set_ylim(0.7, 1.05)
    ax.legend(['Classifier Only (Top-1)', 'Classifier + HMM Viterbi'])
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/hmm_vs_classifier_accuracy.png", dpi=300)
    plt.close()

    # 2. Route continuity comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    df_eval.plot.bar(x='trip_id', y=['classifier_continuity', 'hmm_continuity', 'reconstructed_continuity'],
                     ax=ax, color=['#d62728', '#1f77b4', '#2ca02c'], edgecolor='black', width=0.7)
    ax.set_title("Route Topological Continuity Comparison")
    ax.set_ylabel("Continuity (% consecutive connecting segments)")
    ax.set_xlabel("Trip ID")
    ax.set_xticklabels(df_eval['trip_id'], rotation=0)
    ax.set_ylim(0.4, 1.05)
    ax.legend(['Classifier Only', 'HMM Viterbi Raw', 'Full Reconstructed Route'])
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/hmm_vs_classifier_continuity.png", dpi=300)
    plt.close()
    print("Saved HMM evaluation figures to reports/figures/.")


if __name__ == '__main__':
    main()
