import equinox as eqx
import jax

import equinox as eqx
import jax

class Plate2DNeuralNetwork(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))
        self.layers = []

    def __call__(self, x):
        return self.weight @ x + self.bias