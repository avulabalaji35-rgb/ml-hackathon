"""
PathSnapDataset Abstraction
Provides high-level dataset iteration, item indexing, and candidate queries.
"""

import os
import json
import pandas as pd
from .loader import discover_trips, load_trip, TripData


class PathSnapDataset:
    """
    High-level Dataset interface for PathSnap trajectory and map data.
    
    Usage:
        ds = PathSnapDataset(base_dir=".", partition="train")
        trip = ds[0] # Returns TripData instance
        print(trip.trip_id, len(trip.track))
    """
    def __init__(self, base_dir=".", partition="train"):
        self.base_dir = base_dir
        self.partition = partition
        self.trips_info = discover_trips(base_dir)

        if partition == "train":
            self.trips = self.trips_info['train_trips']
            self.root_path = self.trips_info['train_path']
        elif partition in ("test", "student_test"):
            self.trips = self.trips_info['test_trips']
            self.root_path = self.trips_info['test_path']
        elif partition == "sample":
            sample_dir = os.path.join(base_dir, "data", "sample_data")
            if os.path.exists(sample_dir):
                self.trips = sorted(os.listdir(sample_dir))
                self.root_path = sample_dir
            else:
                self.trips = []
                self.root_path = None
        else:
            raise ValueError(f"Unknown partition: {partition}. Expected 'train', 'student_test', or 'sample'.")

        # Load catalog if available
        catalog_path = os.path.join(base_dir, "data", "dataset_catalog.csv")
        if os.path.exists(catalog_path):
            self.catalog = pd.read_csv(catalog_path)
            self.catalog = self.catalog[self.catalog['partition'] == (
                'train' if partition == 'train' else 'student_test'
            )].set_index('trip_id')
        else:
            self.catalog = None

    def __len__(self):
        return len(self.trips)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            trip_id = self.trips[idx]
        else:
            trip_id = str(idx).zfill(8)

        trip_dir = os.path.join(self.root_path, trip_id)
        return load_trip(trip_dir, partition=self.partition, trip_id=trip_id)

    def get_trip_ids(self):
        return list(self.trips)

    def get_metadata(self, trip_id):
        if self.catalog is not None and trip_id in self.catalog.index:
            return self.catalog.loc[trip_id].to_dict()
        return None

    def __repr__(self):
        return f"PathSnapDataset(partition='{self.partition}', trips={len(self.trips)}, root='{self.root_path}')"
