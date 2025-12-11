"""
Example 1 (part 2) – 1D Gaussian Mixture with DDPM (PyTorch, modern stack)

What it shows:
- DDPM can learn a multimodal 1D mixture (3 Gaussians at -4, 0, +4).
- Outputs hist overlay (data vs model) and training loss curve.
- Time-lapsed visualization showing evolution of probability distribution during
  reverse diffusion process (from noise to final multimodal distribution).

Outputs:
- example1_loss_curve.png: Training loss over iterations
- example1_hist_data_vs_model.png: Final histogram comparison
- example1_timelapse_histograms.png: Grid showing distribution evolution at
  different timesteps during reverse diffusion

Run:
  python pytorch_examples/example1_1d_gaussian_part2.py
"""

import math
import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Positional / timestep embed
# -----------------------------
def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
  """Standard sinusoidal embedding."""
  half = dim // 2
  freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / (half - 1))
  args = timesteps.float()[:, None] * freqs[None, :]
  emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
  if dim % 2 == 1:
    emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
  return emb


# -----------------------------
# Schedules
# -----------------------------
def make_beta_schedule(num_steps: int, kind: Literal["linear", "cosine"] = "linear",
                       beta_start=1e-4, beta_end=0.02) -> torch.Tensor:
  if kind == "linear":
    return torch.linspace(beta_start, beta_end, num_steps)
  if kind == "cosine":
    # Cosine schedule from Nichol & Dhariwal (2021)
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps)
    alphas_cumprod = torch.cos(((x / num_steps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(max=0.999)
  raise ValueError(f"Unknown schedule: {kind}")


# -----------------------------
# Model
# -----------------------------
class MLPDenoiser(nn.Module):
  """Predicts noise epsilon(x_t, t)."""

  def __init__(self, hidden=128, t_dim=64):
    super().__init__()
    self.t_dim = t_dim
    self.fc1 = nn.Linear(1 + t_dim, hidden)
    self.fc2 = nn.Linear(hidden, hidden)
    self.fc3 = nn.Linear(hidden, hidden)
    self.out = nn.Linear(hidden, 1)

  def forward(self, x, t):
    t_emb = sinusoidal_embedding(t, self.t_dim)
    h = torch.cat([x, t_emb], dim=-1)
    h = F.relu(self.fc1(h))
    h = F.relu(self.fc2(h))
    h = F.relu(self.fc3(h))
    return self.out(h)


# -----------------------------
# DDPM core
# -----------------------------
class DDPM:
  def __init__(self, num_steps: int, schedule: Literal["linear", "cosine"] = "linear"):
    betas = make_beta_schedule(num_steps, kind=schedule).to(DEVICE)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    self.num_steps = num_steps
    self.betas = betas
    self.alphas = alphas
    self.alphas_cumprod = alphas_cumprod
    self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

  def q_sample(self, x0, t, noise=None):
    if noise is None:
      noise = torch.randn_like(x0)
    sqrt_ac = self.sqrt_alphas_cumprod[t][:, None]
    sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None]
    return sqrt_ac * x0 + sqrt_om * noise

  def p_losses(self, model, x0, t):
    noise = torch.randn_like(x0)
    xt = self.q_sample(x0, t, noise)
    pred = model(xt, t)
    return F.mse_loss(pred, noise)

  @torch.no_grad()
  def p_sample(self, model, xt, t):
    beta_t = self.betas[t][:, None]
    sqrt_one_minus_at = self.sqrt_one_minus_alphas_cumprod[t][:, None]
    sqrt_recip_alphas_t = 1.0 / torch.sqrt(self.alphas[t])[:, None]

    pred_noise = model(xt, t)
    mean = sqrt_recip_alphas_t * (xt - beta_t / sqrt_one_minus_at * pred_noise)
    if (t == 0).all():
      return mean
    noise = torch.randn_like(xt)
    sigma = torch.sqrt(beta_t)
    return mean + sigma * noise

  @torch.no_grad()
  def sample_loop(self, model, batch_size):
    xt = torch.randn(batch_size, 1, device=DEVICE)
    for t in reversed(range(self.num_steps)):
      t_batch = torch.full((batch_size,), t, device=DEVICE, dtype=torch.long)
      xt = self.p_sample(model, xt, t_batch)
    return xt

  @torch.no_grad()
  def sample_loop_with_history(self, model, batch_size, save_every=50):
    """Sample and save intermediate states for visualization."""
    xt = torch.randn(batch_size, 1, device=DEVICE)
    history = [(self.num_steps - 1, xt.cpu().clone())]
    
    for t in reversed(range(self.num_steps)):
      t_batch = torch.full((batch_size,), t, device=DEVICE, dtype=torch.long)
      xt = self.p_sample(model, xt, t_batch)
      
      # Save at regular intervals and at the end
      if t % save_every == 0 or t == 0:
        history.append((t, xt.cpu().clone()))
    
    return xt, history


# -----------------------------
# Data
# -----------------------------
def sample_data(n: int, means=(-4.0, 0.0, 4.0), std=1.0) -> torch.Tensor:
  means_t = torch.tensor(means, device=DEVICE)
  choices = torch.randint(0, len(means), (n,), device=DEVICE)
  base = means_t[choices]
  noise = torch.randn(n, device=DEVICE) * std
  x = base + noise
  return x[:, None]  # [n, 1]


def gaussian_pdf(x, mean, std):
  """Compute Gaussian PDF manually."""
  return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def true_density(x, means=(-4.0, 0.0, 4.0), std=1.0):
  """
  Compute the true probability density of the 3-mode Gaussian mixture.
  
  Args:
    x: Array of x values
    means: Tuple of means for each mode
    std: Standard deviation (same for all modes)
  
  Returns:
    Array of density values
  """
  x = np.asarray(x)
  density = np.zeros_like(x, dtype=float)
  for mean in means:
    density += gaussian_pdf(x, mean, std)
  density /= len(means)  # Normalize to make it a proper mixture
  return density


# -----------------------------
# Training / plotting
# -----------------------------
def train(
    steps=8000,
    batch_size=256,
    num_steps=1000,
    lr=1e-3,
    log_every=500,
    sample_size=5000,
    schedule="linear",
    out_dir="pytorch_examples_outputs"):
  os.makedirs(out_dir, exist_ok=True)

  # Prepare dataset once to match the spec (50k points)
  data = sample_data(50_000)
  dataset = torch.utils.data.TensorDataset(data)
  loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
  data_iter = iter(loader)

  ddpm = DDPM(num_steps, schedule=schedule)
  model = MLPDenoiser().to(DEVICE)
  opt = torch.optim.Adam(model.parameters(), lr=lr)
  losses = []

  for step in range(1, steps + 1):
    try:
      (x0,) = next(data_iter)
    except StopIteration:
      data_iter = iter(loader)
      (x0,) = next(data_iter)
    x0 = x0.to(DEVICE)
    t = torch.randint(0, num_steps, (batch_size,), device=DEVICE)
    loss = ddpm.p_losses(model, x0, t)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if step % log_every == 0:
      print(f"step {step:05d} loss={loss.item():.4f}")

  # Sample with history for time-lapsed visualization
  print("Sampling with intermediate states...")
  with torch.no_grad():
    samples, history = ddpm.sample_loop_with_history(model, sample_size, save_every=num_steps // 10)
    samples = samples.cpu().numpy().reshape(-1)

  # Plots
  plt.figure(figsize=(10, 4))
  plt.subplot(1, 2, 1)
  plt.plot(losses)
  plt.title("Training loss")
  plt.xlabel("step")
  plt.ylabel("loss")
  loss_path = os.path.join(out_dir, "example1_loss_curve.png")
  plt.tight_layout()
  plt.savefig(loss_path, dpi=200)

  plt.figure(figsize=(6, 4))
  plt.hist(samples, bins=80, density=True, alpha=0.75, label="model")
  plt.hist(sample_data(sample_size).cpu().numpy().reshape(-1),
           bins=80, density=True, alpha=0.45, label="data")
  plt.legend()
  plt.title("1D Gaussian mixture: data vs model")
  out_path = os.path.join(out_dir, "example1_hist_data_vs_model.png")
  plt.tight_layout()
  plt.savefig(out_path, dpi=200)
  print(f"Saved loss curve to {loss_path}")
  print(f"Saved histogram to {out_path}")

  # Time-lapsed visualization: evolution of probability distribution
  print("Creating time-lapsed visualization...")
  n_steps = len(history)
  n_cols = min(5, n_steps)
  n_rows = (n_steps + n_cols - 1) // n_cols
  
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))
  if n_rows == 1:
    axes = axes[None, :] if n_cols > 1 else axes
  elif n_cols == 1:
    axes = axes[:, None]
  
  # Generate true distribution samples for histogram overlay
  true_samples = sample_data(sample_size).cpu().numpy().reshape(-1)
  
  for idx, (t_step, xt_state) in enumerate(history):
    row, col = idx // n_cols, idx % n_cols
    ax = axes[row, col] if n_rows > 1 else axes[col]
    
    values = xt_state.numpy().reshape(-1)
    # Histogram of true distribution (background)
    ax.hist(true_samples, bins=60, density=True, alpha=0.4, color='red', 
            edgecolor='darkred', linewidth=0.5, label='true dist' if idx == 0 else '')
    # Histogram of generated samples (foreground)
    ax.hist(values, bins=60, density=True, alpha=0.7, color='steelblue', 
            edgecolor='black', linewidth=0.5, label='generated' if idx == 0 else '')
    ax.set_title(f't = {t_step}', fontsize=10)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('density', fontsize=8)
    ax.set_xlim(-8, 8)
    ax.grid(True, alpha=0.3)
    ax.axvline(-4, color='orange', linestyle=':', alpha=0.4, linewidth=1)
    ax.axvline(0, color='orange', linestyle=':', alpha=0.4, linewidth=1)
    ax.axvline(4, color='orange', linestyle=':', alpha=0.4, linewidth=1)
    if idx == 0:
      ax.legend(fontsize=7, loc='upper right')
  
  # Hide unused subplots
  for idx in range(n_steps, n_rows * n_cols):
    row, col = idx // n_cols, idx % n_cols
    ax = axes[row, col] if n_rows > 1 else axes[col]
    ax.axis('off')
  
  plt.suptitle('Reverse Diffusion Process: Evolution of Probability Distribution', fontsize=12, y=1.02)
  plt.tight_layout()
  timeline_path = os.path.join(out_dir, "example1_timelapse_histograms.png")
  plt.savefig(timeline_path, dpi=200, bbox_inches='tight')
  print(f"Saved time-lapsed visualization to {timeline_path}")


if __name__ == "__main__":
  train()

