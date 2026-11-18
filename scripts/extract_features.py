"""
Script to extract feature datasets across representative training trips and student-testing trips.
Saves extracted feature tables to data/train_features.csv and data/test_features.csv.
"""

import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath("."))
from src.data.dataset import PathSnapDataset
from src.features import CandidateExtractor


def main():
    extractor = CandidateExtractor(search_radius=35.0)

    # 1. Training trips: 8 diverse training trips
    train_ds = PathSnapDataset(".", partition="train")
    all_train = train_ds.get_trip_ids()
    # Select 8 trips with varied sizes
    selected_train = [all_train[i] for i in [0, 4, 9, 14, 19, 24, 29, 34] if i < len(all_train)]
    print(f"Extracting features from {len(selected_train)} training trips: {selected_train}")

    train_dfs = []
    for t_id in tqdm(selected_train, desc="Train trips"):
        trip = train_ds[t_id]
        df_trip = extractor.extract_for_trip(trip, sample_step=4, max_candidates_per_obs=8)
        if len(df_trip) > 0:
            train_dfs.append(df_trip)

    df_train = pd.concat(train_dfs, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    train_path = "data/train_features.csv"
    df_train.to_csv(train_path, index=False)
    print(f"\nTrain features saved: {df_train.shape} -> {train_path}")
    print(f"Train class balance: {df_train['is_match'].value_counts().to_dict()}")

    # 2. Student testing trips: 5 test trips
    test_ds = PathSnapDataset(".", partition="student_test")
    test_trips = test_ds.get_trip_ids()[:5]
    print(f"\nExtracting features from {len(test_trips)} student testing trips: {test_trips}")

    test_dfs = []
    for t_id in tqdm(test_trips, desc="Test trips"):
        trip = test_ds[t_id]
        df_trip = extractor.extract_for_trip(trip, sample_step=4, max_candidates_per_obs=8)
        if len(df_trip) > 0:
            test_dfs.append(df_trip)

    df_test = pd.concat(test_dfs, ignore_index=True)
    test_path = "data/test_features.csv"
    df_test.to_csv(test_path, index=False)
    print(f"\nTest features saved: {df_test.shape} -> {test_path}")
    print(f"Test class balance: {df_test['is_match'].value_counts().to_dict()}")


if __name__ == '__main__':
    main()
