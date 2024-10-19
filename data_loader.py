# data_loader.py

import xarray
import dataclasses
from graphcast import data_utils

def load_dataset(gcs_bucket, dataset_file):
    with gcs_bucket.blob(dataset_file).open("rb") as f:
        dataset = xarray.load_dataset(f).compute()
    return dataset

def load_normalization_data(gcs_bucket, diffs_stddev_file, mean_file, stddev_file):
    with gcs_bucket.blob(diffs_stddev_file).open("rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(mean_file).open("rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(stddev_file).open("rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()
    return diffs_stddev_by_level, mean_by_level, stddev_by_level

def extract_inputs_targets_forcings(example_batch, target_lead_times, task_config):
    return data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=target_lead_times,
        **dataclasses.asdict(task_config)
    )