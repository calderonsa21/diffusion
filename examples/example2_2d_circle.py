import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

from diffusion_tf.diffusion_utils import GaussianDiffusion, get_beta_schedule
from diffusion_tf import nn


tf.disable_v2_behavior()


def sample_circle(batch_size: int, radius=3.0, noise_std=0.1) -> np.ndarray:
  angles = np.random.uniform(0, 2 * np.pi, size=batch_size).astype(np.float32)
  x = radius * np.cos(angles)
  y = radius * np.sin(angles)
  pts = np.stack([x, y], axis=-1) + np.random.randn(batch_size, 2).astype(np.float32) * noise_std
  return pts[:, None, None, :]  # [B, 1, 1, 2]


def make_diffusion(num_steps: int) -> GaussianDiffusion:
  betas = get_beta_schedule(
      'linear', beta_start=1e-4, beta_end=0.02, num_diffusion_timesteps=num_steps)
  return GaussianDiffusion(betas=betas, loss_type='noisepred')


def denoise_mlp(x, t, hidden_size=256):
  with tf.variable_scope('denoiser', reuse=tf.AUTO_REUSE):
    flat_x = tf.reshape(x, [tf.shape(x)[0], -1])
    t_emb = nn.get_timestep_embedding(t, embedding_dim=64)
    h = tf.concat([flat_x, t_emb], axis=1)
    h = tf.nn.relu(nn.dense(h, name='dense1', num_units=hidden_size))
    h = tf.nn.relu(nn.dense(h, name='dense2', num_units=hidden_size))
    out = nn.dense(h, name='out', num_units=2, init_scale=0.1)
    return tf.reshape(out, tf.shape(x))


def train_and_sample(
    steps=3000,
    batch_size=256,
    log_every=300,
    sample_size=1000,
    num_steps=200,
    out_dir='examples_outputs'):
  diffusion = make_diffusion(num_steps)
  x_ph = tf.placeholder(tf.float32, [None, 1, 1, 2])
  t_ph = tf.placeholder(tf.int32, [None])

  losses = diffusion.p_losses(denoise_mlp, x_ph, t_ph)
  loss = tf.reduce_mean(losses)
  train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

  samples = diffusion.p_sample_loop(denoise_mlp, shape=[sample_size, 1, 1, 2])

  os.makedirs(out_dir, exist_ok=True)
  loss_history = []
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1, steps + 1):
      batch = sample_circle(batch_size)
      t_batch = np.random.randint(0, num_steps, size=batch_size, dtype=np.int32)
      _, loss_v = sess.run([train_op, loss], feed_dict={x_ph: batch, t_ph: t_batch})
      if step % log_every == 0:
        print(f'Step {step:04d} loss={loss_v:.4f}')
      loss_history.append(loss_v)

    gen = sess.run(samples)

  gen_pts = gen.reshape(-1, 2)

  plt.figure(figsize=(10, 4))
  plt.subplot(1, 2, 1)
  plt.plot(loss_history)
  plt.xlabel('step')
  plt.ylabel('loss')
  plt.title('Training loss')

  plt.subplot(1, 2, 2)
  plt.scatter(gen_pts[:, 0], gen_pts[:, 1], s=4, alpha=0.5, label='model')
  data_pts = sample_circle(sample_size).reshape(-1, 2)
  plt.scatter(data_pts[:, 0], data_pts[:, 1], s=4, alpha=0.4, label='data')
  plt.axis('equal')
  plt.legend()
  plt.title('2D circle samples')

  out_path = os.path.join(out_dir, 'example2_circle.png')
  plt.tight_layout()
  plt.savefig(out_path, dpi=200)
  print(f'Saved plot to {out_path}')


if __name__ == '__main__':
  train_and_sample()


