"""
Minimal DDPM example in PyTorch: 2D noisy circle.
Run: python pytorch_examples/example2_2d_circle.py
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
  half = dim // 2
  freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / (half - 1))
  args = timesteps.float()[:, None] * freqs[None, :]
  emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
  if dim % 2 == 1:
    emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
  return emb


def make_beta_schedule(num_steps: int, beta_start=1e-4, beta_end=0.02):
  return torch.linspace(beta_start, beta_end, num_steps)


class MLPDenoiser(nn.Module):
  def __init__(self, hidden=256, t_dim=64):
    super().__init__()
    self.t_dim = t_dim
    self.fc1 = nn.Linear(2 + t_dim, hidden)
    self.fc2 = nn.Linear(hidden, hidden)
    self.out = nn.Linear(hidden, 2)

  def forward(self, x, t):
    t_emb = sinusoidal_embedding(t, self.t_dim)
    h = torch.cat([x, t_emb], dim=-1)
    h = F.relu(self.fc1(h))
    h = F.relu(self.fc2(h))
    return self.out(h)


class DDPM:
  def __init__(self, num_steps: int):
    betas = make_beta_schedule(num_steps).to(DEVICE)
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
    xt = torch.randn(batch_size, 2, device=DEVICE)
    for t in reversed(range(self.num_steps)):
      t_batch = torch.full((batch_size,), t, device=DEVICE, dtype=torch.long)
      xt = self.p_sample(model, xt, t_batch)
    return xt

  @torch.no_grad()
  def sample_loop_with_history(self, model, batch_size, save_every=50):
    """
    Sample with intermediate saves for visualization (reverse diffusion trajectory).
    """
    xt = torch.randn(batch_size, 2, device=DEVICE)
    history = [(self.num_steps - 1, xt.cpu().clone())]
    for t in reversed(range(self.num_steps)):
      t_batch = torch.full((batch_size,), t, device=DEVICE, dtype=torch.long)
      xt = self.p_sample(model, xt, t_batch)
      if t % save_every == 0 or t == 0:
        history.append((t, xt.cpu().clone()))
    return xt, history


def sample_data(batch_size: int, radius=3.0, noise_std=0.1):
  angles = torch.rand(batch_size, device=DEVICE) * 2 * math.pi
  x = radius * torch.cos(angles)
  y = radius * torch.sin(angles)
  pts = torch.stack([x, y], dim=-1)
  pts = pts + torch.randn_like(pts) * noise_std
  return pts  # [B, 2]


def train(
    steps=3000,
    batch_size=256,
    num_steps=1000,
    lr=1e-3,
    log_every=300,
    sample_size=1000,
    out_dir="pytorch_examples_outputs"):
  os.makedirs(out_dir, exist_ok=True)
  ddpm = DDPM(num_steps)
  model = MLPDenoiser().to(DEVICE)
  opt = torch.optim.Adam(model.parameters(), lr=lr)
  losses = []
  for step in range(1, steps + 1):
    x0 = sample_data(batch_size)
    t = torch.randint(0, num_steps, (batch_size,), device=DEVICE)
    loss = ddpm.p_losses(model, x0, t)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if step % log_every == 0:
      print(f"step {step:04d} loss={loss.item():.4f}")

  with torch.no_grad():
    samples, history = ddpm.sample_loop_with_history(model, sample_size, save_every=max(1, num_steps // 10))
    samples = samples.cpu().numpy()
    data = sample_data(sample_size).cpu().numpy()

  plt.figure(figsize=(10, 4))
  plt.subplot(1, 2, 1)
  plt.plot(losses)
  plt.title("Training loss")
  plt.xlabel("step")
  plt.ylabel("loss")

  plt.subplot(1, 2, 2)
  plt.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.6, label="model")
  plt.scatter(data[:, 0], data[:, 1], s=4, alpha=0.4, label="data")
  plt.axis("equal")
  plt.legend()
  plt.title("2D circle")

  out_path = os.path.join(out_dir, "example2_circle.png")
  plt.tight_layout()
  plt.savefig(out_path, dpi=200)
  print(f"Saved plot to {out_path}")

  # Time-lapse scatter of reverse diffusion
  print("Creating time-lapse scatter...")
  n_steps = len(history)
  n_cols = min(5, n_steps)
  n_rows = (n_steps + n_cols - 1) // n_cols
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
  if n_rows == 1:
    axes = axes[None, :] if n_cols > 1 else axes
  elif n_cols == 1:
    axes = axes[:, None]

  # Generate true distribution samples for overlay
  true_samples = sample_data(sample_size).cpu().numpy()

  for idx, (t_step, xt_state) in enumerate(history):
    row, col = idx // n_cols, idx % n_cols
    ax = axes[row, col] if n_rows > 1 else axes[col]
    pts = xt_state.numpy()
    
    # Overlay true distribution scatter (background)
    ax.scatter(true_samples[:, 0], true_samples[:, 1], s=2, alpha=0.3, 
              color="red", label="true dist" if idx == 0 else "")
    # Generated samples scatter (foreground)
    ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.6, color="steelblue", 
              label="generated" if idx == 0 else "")
    
    ax.set_title(f"t = {t_step}", fontsize=10)
    ax.set_xlabel("x₁", fontsize=8)
    ax.set_ylabel("x₂", fontsize=8)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    # reference circle
    circle = plt.Circle((0, 0), 3.0, fill=False, color="orange", linestyle=":", linewidth=1, alpha=0.5)
    ax.add_patch(circle)
    if idx == 0:
      ax.legend(fontsize=7, loc="upper right")

  # Hide unused subplots
  for idx in range(n_steps, n_rows * n_cols):
    row, col = idx // n_cols, idx % n_cols
    ax = axes[row, col] if n_rows > 1 else axes[col]
    ax.axis("off")

  plt.suptitle("Reverse Diffusion Process: 2D Circle Evolution", fontsize=12, y=1.02)
  plt.tight_layout()
  timeline_path = os.path.join(out_dir, "example2_timelapse_scatter.png")
  plt.savefig(timeline_path, dpi=200, bbox_inches="tight")
  print(f"Saved time-lapse scatter to {timeline_path}")


if __name__ == "__main__":
  train()

