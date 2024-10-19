# main.py

import functools
import jax
import haiku as hk
import numpy as np
import xarray
import yaml  # For reading config.yaml
from google.cloud import storage
from graphcast import checkpoint, rollout, graphcast  # Import graphcast module
# Ensure you have imported graphcast

from data_loader import load_dataset, load_normalization_data, extract_inputs_targets_forcings
from model import construct_wrapped_graphcast, create_jitted_functions
from utils import parse_file_parts

def main():
    # Load configurations from config.yaml
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Authentication for Google Cloud
    gcs_client = storage.Client.create_anonymous_client()
    gcs_bucket = gcs_client.get_bucket(config['gcs_bucket_name'])

    # List all files in the "params" directory of the bucket
    blobs = gcs_bucket.list_blobs(prefix="params/")
    print("Files in params directory:")
    for blob in blobs:
        print(blob.name)

    # Load model parameters
    model_file_name = config['model_file_name']
    blob = gcs_bucket.blob(model_file_name)
    with blob.open("rb") as f:
        # Corrected line: Pass graphcast.CheckPoint as the type argument
        ckpt = checkpoint.load(f, graphcast.CheckPoint)

    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

    # Print model information
    print("Model description:\n", ckpt.description, "\n")
    print("Model license:\n", ckpt.license, "\n")

    # Load dataset
    dataset_file = config['dataset_file']
    example_batch = load_dataset(gcs_bucket, dataset_file)

    # Define steps
    train_steps = config['train_steps']
    eval_steps = example_batch.sizes["time"] - 2

    # Extract training and evaluation data
    train_inputs, train_targets, train_forcings = extract_inputs_targets_forcings(
        example_batch, slice("6h", f"{train_steps*6}h"), task_config)

    eval_inputs, eval_targets, eval_forcings = extract_inputs_targets_forcings(
        example_batch, slice("6h", f"{eval_steps*6}h"), task_config)

    # Print dimensions
    print("All Examples:  ", example_batch.dims.mapping)
    print("Train Inputs:  ", train_inputs.dims.mapping)
    print("Train Targets: ", train_targets.dims.mapping)
    print("Train Forcings:", train_forcings.dims.mapping)
    print("Eval Inputs:   ", eval_inputs.dims.mapping)
    print("Eval Targets:  ", eval_targets.dims.mapping)
    print("Eval Forcings: ", eval_forcings.dims.mapping)

    # Load normalization data
    normalization_files = config['normalization_files']
    diffs_stddev_by_level, mean_by_level, stddev_by_level = load_normalization_data(
        gcs_bucket,
        normalization_files['diffs_stddev_by_level'],
        normalization_files['mean_by_level'],
        normalization_files['stddev_by_level']
    )

    # Build jitted functions and initialize weights
    predictor = construct_wrapped_graphcast(
        model_config, task_config,
        diffs_stddev_by_level, mean_by_level, stddev_by_level
    )

    run_forward, loss_fn, grads_fn = create_jitted_functions(
        predictor, model_config, task_config
    )

    init_jitted = jax.jit(run_forward.init)

    if params is None:
        params, state = init_jitted(
            rng=jax.random.PRNGKey(config['random_seed']),
            inputs=train_inputs,
            targets_template=train_targets,
            forcings=train_forcings
        )

    # Prepare functions
    def run_forward_jitted(inputs, targets_template, forcings):
        predictions, _ = run_forward.apply(
            params, state, jax.random.PRNGKey(config['random_seed']),
            inputs, targets_template, forcings
        )
        return predictions

    # Autoregressive rollout
    assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
        "Model resolution doesn't match the data resolution."
    )

    print("Inputs:  ", eval_inputs.dims.mapping)
    print("Targets: ", eval_targets.dims.mapping)
    print("Forcings:", eval_forcings.dims.mapping)

    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(config['random_seed']),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings
    )
    print(predictions)

if __name__ == "__main__":
    main()
