"""Single-forward PiStar validation; metrics match its existing C51 definitions."""
import jax
import jax.numpy as jnp


def make_value_evaluator(model_def, nnx, observation_module, value_module):
    @jax.jit
    def evaluate(params, rng, observation, target):
        model = nnx.merge(model_def, params)
        model.eval()
        observation = observation_module.preprocess_observation(
            rng, observation, train=False, image_keys=list(observation.images.keys())
        )
        logits = model(observation, train=False).astype(jnp.float32)
        target = value_module.normalize_value_target_shape(target)
        target_dist = value_module.target_to_twohot(target)
        loss = -jnp.sum(target_dist * jax.nn.log_softmax(logits, axis=-1), axis=-1)
        prediction = value_module.dist_to_value(logits)
        # Return sums so a smaller final batch has its exact frame weight.
        return jnp.sum(loss), jnp.sum(jnp.abs(prediction - target))

    return evaluate
