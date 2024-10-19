# model.py

import haiku as hk
import jax
import functools
from graphcast import graphcast, casting, normalization, autoregressive
from graphcast import xarray_jax, xarray_tree

def construct_wrapped_graphcast(
    model_config, task_config,
    diffs_stddev_by_level, mean_by_level, stddev_by_level
):
    """Constructs and wraps the GraphCast Predictor."""
    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level
    )
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor

def create_jitted_functions(predictor, model_config, task_config):
    """Creates jitted functions for forward pass, loss, and gradients."""

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    @hk.transform_with_state
    def loss_fn(inputs, targets, forcings):
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics)
        )

    def grads_fn(params, state, inputs, targets, forcings):
        def _aux(params, state, i, t, f):
            (loss, diagnostics), next_state = loss_fn.apply(
                params, state, jax.random.PRNGKey(0), i, t, f
            )
            return loss, (diagnostics, next_state)
        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
            _aux, has_aux=True
        )(params, state, inputs, targets, forcings)
        return loss, diagnostics, next_state, grads

    return run_forward, loss_fn, grads_fn