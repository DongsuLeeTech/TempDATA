"""From https://raw.githubusercontent.com/google/flax/main/examples/ppo/models.py"""

from flax import linen as nn
import jax.numpy as jnp

class VisualEncoder(nn.Module):
  """Class defining the actor-critic model."""

  # @nn.compact
  # def __call__(self, x):
  #   dtype = jnp.float32
  #   x = x.astype(dtype) / 255.
  #   x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name='conv1',
  #               dtype=dtype)(x)
  #   x = nn.relu(x)
  #   x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name='conv2',
  #               dtype=dtype)(x)
  #   x = nn.relu(x)
  #   x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name='conv3',
  #               dtype=dtype)(x)
  #   x = nn.relu(x)
  #   x = x.reshape((x.shape[0], -1))  # flatten
  #   return x
  @nn.compact
  def __call__(self, x):
      x = x.astype(jnp.float32) / 255.0
      x = nn.Conv(features=32, kernel_size=(3, 3), strides=2)(x)  # 32x32 => 16x16
      x = nn.gelu(x)
      # x = nn.Conv(features=64, kernel_size=(3, 3))(x)
      # x = nn.gelu(x)
      x = nn.Conv(features=64, kernel_size=(3, 3), strides=2)(x)  # 16x16 => 8x8
      x = nn.gelu(x)
#       x = nn.Conv(features=2 * 64, kernel_size=(3, 3))(x)
#       x = nn.gelu(x)
      x = nn.Conv(features=2 * 64, kernel_size=(3, 3), strides=2)(x)  # 8x8 => 4x4
      x = nn.gelu(x)
      x = x.reshape(x.shape[0], -1)  # Image grid to single feature vector
      x = nn.Dense(features=1024)(x)
      return x


class VisualDecoder(nn.Module):
  @nn.compact
  def __call__(self, x):
      x = x.astype(jnp.float32)
      x = nn.Dense(features=2 * 16 * 64)(x)
      x = nn.gelu(x)
      x = x.reshape(x.shape[0], 4, 4, -1)
      x = nn.ConvTranspose(features=2 * 64, kernel_size=(3, 3), strides=(2, 2))(x)
      x = nn.gelu(x)
      # x = nn.Conv(features=2 * 64, kernel_size=(3, 3))(x)
      # x = nn.gelu(x)
      x = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
      x = nn.gelu(x)
      x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
      x = nn.gelu(x)
#       x = nn.Conv(features=64, kernel_size=(3, 3))(x)
#       x = nn.gelu(x)
      x = nn.ConvTranspose(features=3, kernel_size=(3, 3), strides=(2, 2))(x)
      x = nn.tanh(x)
      return x


visual_dec_configs = {
    'VisualDecoder': VisualDecoder,
}

visual_enc_configs = {
    'VisualEncoder': VisualEncoder,
}