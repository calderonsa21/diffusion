import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# Import Ho's DDPM utilities from the local package
from diffusion_tf.diffusion_utils import get_beta_schedule, GaussianDiffusion

# ---------- 1. Data: 1D mixture of Gaussians ----------

def sample_1d_mog(n, seed=0):
    """
    Mixture of 3 Gaussians in 1D, with means at -4, 0, 4.
    Returns array of shape [n, 1, 1, 1] so it looks like a '1x1 image' to the DDPM.
    """
    rng = np.random.RandomState(seed)
    means = np.array([-4.0, 0.0, 4.0])
    stds = np.array([0.5, 0.7, 0.5])
    mixing = np.array([0.3, 0.4, 0.3])

    comps = rng.choice(len(means), size=n, p=mixing)
    x = rng.normal(loc=means[comps], scale=stds[comps])
    x = x.astype(np.float32)
    # Shape [B, H, W, C] with H=W=C=1
    return x.reshape(-1, 1, 1, 1)


# ---------- 2. Small denoiser network ε_θ(x_t, t) ----------

def build_denoise_net(x, t, hidden_dim=64):
    """
    x: [B, 1, 1, 1]
    t: [B] (int32)
    Returns predicted noise with same shape as x.
    """
    B = tf.shape(x)[0]

    # Flatten x: [B, 1]
    x_flat = tf.reshape(x, [B, 1])

    # Simple sinusoidal timestep embedding (you could also import nn.get_timestep_embedding)
    t = tf.cast(t, tf.float32)
    t_embed = tf.stack([tf.sin(t / 1000.0), tf.cos(t / 1000.0)], axis=1)  # [B, 2]

    inp = tf.concat([x_flat, t_embed], axis=1)  # [B, 3]

    h = tf.layers.dense(inp, hidden_dim, activation=tf.nn.swish)
    h = tf.layers.dense(h, hidden_dim, activation=tf.nn.swish)
    out = tf.layers.dense(h, 1)  # [B, 1]

    return tf.reshape(out, [B, 1, 1, 1])


# ---------- 3. DDPM graph: loss & sampling ----------

def build_graph(T=1000, beta_start=1e-4, beta_end=0.02, batch_size=256):
    # Placeholders
    x0_ph = tf.placeholder(tf.float32, [None, 1, 1, 1], name="x0")

    # Beta schedule & diffusion helper
    betas = get_beta_schedule(
        beta_schedule='linear',
        beta_start=beta_start,
        beta_end=beta_end,
        num_diffusion_timesteps=T,
    )
    diffusion = GaussianDiffusion(betas=betas, loss_type='noisepred')

    # Sample timesteps uniformly
    B = tf.shape(x0_ph)[0]
    t_ph = tf.random_uniform([B], minval=0, maxval=diffusion.num_timesteps, dtype=tf.int32)

    # Denoiser wrapper (to match diffusion.p_losses interface)
    def denoise_fn(x_t, t):
        return build_denoise_net(x_t, t)

    # Loss over x0_ph and t_ph
    losses = diffusion.p_losses(denoise_fn, x_start=x0_ph, t=t_ph)
    loss = tf.reduce_mean(losses)

    # Optimizer
    opt = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_op = opt.minimize(loss)

    # Sampling
    # Start from pure noise, then run reverse process
    sample_shape = (batch_size, 1, 1, 1)
    samples = diffusion.p_sample_loop(
        denoise_fn=denoise_fn,
        shape=sample_shape,
        noise_fn=tf.random_normal,
    )

    return {
        'x0_ph': x0_ph,
        'loss': loss,
        'train_op': train_op,
        'samples': samples,
    }


# ---------- 4. Training loop & plotting ----------

def main():
    tf.reset_default_graph()
    batch_size = 256
    n_steps = 20000

    g = build_graph(batch_size=batch_size)

    out_dir = "example1_outputs"
    os.makedirs(out_dir, exist_ok=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Training...")
        for step in range(1, n_steps + 1):
            x_batch = sample_1d_mog(batch_size)
            _, loss_val = sess.run(
                [g['train_op'], g['loss']],
                feed_dict={g['x0_ph']: x_batch}
            )
            if step % 500 == 0:
                print(f"Step {step}: loss = {loss_val:.4f}")

        print("Sampling...")
        samples_val = sess.run(g['samples'])  # [B,1,1,1]
        samples_1d = samples_val.reshape(-1)

        # For comparison, draw ground-truth samples
        gt = sample_1d_mog(5000)

        # Plot
        plt.figure(figsize=(6, 4))
        plt.hist(gt.reshape(-1), bins=100, density=True, alpha=0.5, label="Ground truth")
        plt.hist(samples_1d, bins=100, density=True, alpha=0.5, label="DDPM samples")
        plt.xlabel("x")
        plt.ylabel("density")
        plt.title("Example 1: 1D Mixture of Gaussians")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "example1_hist.png"))
        plt.close()

        print(f"Saved plot to {os.path.join(out_dir, 'example1_hist.png')}")

if __name__ == "__main__":
    main()