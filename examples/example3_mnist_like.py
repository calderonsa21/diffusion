import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from tensorflow.keras.datasets import mnist

from diffusion_tf.diffusion_utils_2 import GaussianDiffusion2, get_beta_schedule
from diffusion_tf import nn


tf.disable_v2_behavior()
tf.set_random_seed(42)
np.random.seed(42)


def load_mnist():
  (x_train, _), _ = mnist.load_data()
  x = x_train.astype(np.float32) / 127.5 - 1.0  # [-1, 1]
  x = x[..., None]  # [N, 28, 28, 1]
  return x


def make_diffusion(num_steps: int) -> GaussianDiffusion2:
  betas = get_beta_schedule(
      'linear', beta_start=1e-4, beta_end=0.02, num_diffusion_timesteps=num_steps)
  return GaussianDiffusion2(
      betas=betas,
      model_mean_type='eps',
      model_var_type='fixedsmall',
      loss_type='mse')


def denoise_conv(x, t, channels=64):
  """Lightweight convolutional denoiser for MNIST-like data."""
  with tf.variable_scope('denoiser', reuse=tf.AUTO_REUSE):
    t_emb = nn.get_timestep_embedding(t, embedding_dim=64)
    t_proj = tf.layers.dense(t_emb, channels, activation=tf.nn.relu)
    t_proj = t_proj[:, None, None, :]  # [B, 1, 1, C]

    h = tf.layers.conv2d(x, 32, 3, padding='same', activation=tf.nn.relu)
    h = tf.layers.conv2d(h, channels, 3, strides=2, padding='same', activation=tf.nn.relu)
    h = tf.layers.conv2d(h, channels, 3, padding='same', activation=tf.nn.relu)
    h = h + t_proj
    h = tf.layers.conv2d(h, channels, 3, padding='same', activation=tf.nn.relu)
    h = tf.image.resize(h, [28, 28], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h = tf.layers.conv2d(h, 32, 3, padding='same', activation=tf.nn.relu)
    out = tf.layers.conv2d(h, 1, 3, padding='same', activation=None)
    return out


def train_and_sample(
    steps=4000,
    batch_size=128,
    log_every=400,
    sample_size=16,
    num_steps=400,
    out_dir='examples_outputs'):
  data = load_mnist()
  diffusion = make_diffusion(num_steps)
  x_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
  t_ph = tf.placeholder(tf.int32, [None])

  losses = diffusion.training_losses(denoise_conv, x_ph, t_ph)
  loss = tf.reduce_mean(losses)
  train_op = tf.train.AdamOptimizer(2e-4).minimize(loss)

  samples = diffusion.p_sample_loop(denoise_conv, shape=[sample_size, 28, 28, 1])

  os.makedirs(out_dir, exist_ok=True)
  loss_history = []
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_data = data.shape[0]
    for step in range(1, steps + 1):
      idx = np.random.randint(0, num_data, size=batch_size)
      batch = data[idx]
      t_batch = np.random.randint(0, num_steps, size=batch_size, dtype=np.int32)
      _, loss_v = sess.run([train_op, loss], feed_dict={x_ph: batch, t_ph: t_batch})
      if step % log_every == 0:
        print(f'Step {step:05d} loss={loss_v:.4f}')
      loss_history.append(loss_v)

    gen = sess.run(samples)

  plt.figure(figsize=(10, 4))
  plt.subplot(1, 2, 1)
  plt.plot(loss_history)
  plt.xlabel('step')
  plt.ylabel('loss')
  plt.title('Training loss')

  plt.subplot(1, 2, 2)
  grid = int(np.ceil(np.sqrt(sample_size)))
  for i in range(sample_size):
    plt.subplot(grid, grid, i + 1)
    plt.axis('off')
    plt.imshow((gen[i, ..., 0] + 1) * 0.5, cmap='gray')
  plt.tight_layout()

  out_path = os.path.join(out_dir, 'example3_mnist.png')
  plt.savefig(out_path, dpi=200)
  print(f'Saved plot to {out_path}')


if __name__ == '__main__':
  train_and_sample()

