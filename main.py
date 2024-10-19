import dataclasses
import functools
import yaml
from typing import Tuple

import haiku as hk
import jax
import numpy as np
import xarray
from graphcast import (
    autoregressive,
    casting,
    data_utils,
    graphcast,
    normalization,
    rollout,
    xarray_jax,
    xarray_tree,
)
from utils import (
    load_blob_to_xarray,
    load_checkpoint,
    print_model_info,
    setup_gcs_client,
)


def main():
    # Load configuration from config.yaml
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Step 1: Set up Google Cloud Storage client
    gcs_client = setup_gcs_client()
    gcs_bucket = gcs_client.bucket(config["gcs_bucket_name"])

    # Step 2: Load the model checkpoint
    ckpt = load_checkpoint(gcs_bucket, config["model_file_name"])
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

    # Print model information
    print_model_info(ckpt)

    # Step 3: Load the dataset
    dataset = load_blob_to_xarray(gcs_bucket, config["dataset_file_name"])

    # Step 4: Extract inputs, targets, and forcings
    train_inputs, train_targets, train_forcings, eval_inputs, eval_targets, eval_forcings = extract_data(
        dataset, task_config, config
    )

    # Step 5: Load normalization data
    normalization_data = load_normalization_data(
        gcs_bucket, config["normalization_files"]
    )

    # Step 6: Build the model
    run_forward, params, state = build_model(
        model_config, task_config, normalization_data, params, state, eval_inputs, eval_targets, eval_forcings
    )

    # Step 7: Run predictions
    predictions = run_predictions(
        run_forward,
        params,
        state,
        model_config,
        task_config,
        eval_inputs,
        eval_targets,
        eval_forcings,
        config,
    )

    # Step 8: Process or save predictions
    process_predictions(predictions)


def extract_data(dataset, task_config, config) -> Tuple:
    """Extracts inputs, targets, and forcings for training and evaluation."""
    train_steps = config["train_steps"]
    eval_steps = dataset.sizes["time"] - config["eval_steps_offset"]

    train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
        dataset,
        target_lead_times=slice("6h", f"{train_steps * 6}h"),
        **dataclasses.asdict(task_config)
    )

    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        dataset,
        target_lead_times=slice("6h", f"{eval_steps * 6}h"),
        **dataclasses.asdict(task_config)
    )

    # Print dimensions of the data
    print("All Examples:  ", dataset.dims.mapping)
    print("Train Inputs:  ", train_inputs.dims.mapping)
    print("Train Targets: ", train_targets.dims.mapping)
    print("Train Forcings:", train_forcings.dims.mapping)
    print("Eval Inputs:   ", eval_inputs.dims.mapping)
    print("Eval Targets:  ", eval_targets.dims.mapping)
    print("Eval Forcings: ", eval_forcings.dims.mapping)

    return train_inputs, train_targets, train_forcings, eval_inputs, eval_targets, eval_forcings


def load_normalization_data(gcs_bucket, normalization_files: dict) -> dict:
    """Loads normalization data from Google Cloud Storage."""
    normalization_data = {}
    for key, file_name in normalization_files.items():
        normalization_data[key] = load_blob_to_xarray(gcs_bucket, file_name)
    return normalization_data


def build_model(
    model_config,
    task_config,
    normalization_data,
    params,
    state,
    inputs,
    targets_template,
    forcings,
):
    """Builds and initializes the model."""
    # Construct the model
    def construct_wrapped_graphcast(model_config, task_config):
        """Constructs and wraps the GraphCast Predictor."""
        predictor = graphcast.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=normalization_data["diffs_stddev_by_level"],
            mean_by_level=normalization_data["mean_by_level"],
            stddev_by_level=normalization_data["stddev_by_level"],
        )
        predictor = autoregressive.Predictor(
            predictor, gradient_checkpointing=True
        )
        return predictor

    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        return predictor(
            inputs, targets_template=targets_template, forcings=forcings
        )

    # Initialize the model if parameters are not loaded
    if params is None:
        rng = jax.random.PRNGKey(0)
        params, state = run_forward.init(
            rng,
            model_config,
            task_config,
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
        )

    return run_forward, params, state


def run_predictions(
    run_forward,
    params,
    state,
    model_config,
    task_config,
    eval_inputs,
    eval_targets,
    eval_forcings,
    config,
):
    """Runs predictions using autoregressive rollout."""
    # Define a function to apply the forward pass with the loaded parameters
    def run_forward_apply(inputs, targets_template, forcings):
        return run_forward.apply(
            params,
            state,
            jax.random.PRNGKey(config["random_seed"]),
            model_config,
            task_config,
            inputs,
            targets_template,
            forcings,
        )[0]  # Extract predictions from the output tuple

    # JIT compile the forward application function for efficiency
    run_forward_jitted = jax.jit(run_forward_apply)

    # Ensure model resolution matches the data resolution
    assert model_config.resolution in (
        0,
        360.0 / eval_inputs.sizes["lon"],
    ), "Model resolution doesn't match the data resolution."

    # Print dimensions for verification
    print("Inputs:  ", eval_inputs.dims.mapping)
    print("Targets: ", eval_targets.dims.mapping)
    print("Forcings:", eval_forcings.dims.mapping)

    # Run the autoregressive prediction
    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(config["random_seed"]),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
    )

    return predictions


def process_predictions(predictions):
    """Processes or saves the predictions."""
    # For demonstration, we'll print the predictions
    print(predictions)
    # You can save the predictions to a file or perform further analysis here


if __name__ == "__main__":
    main()
