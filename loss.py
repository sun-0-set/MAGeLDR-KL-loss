import torch
import torch.nn as nn
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
import torch.distributed as dist

try:
  import wandb
  WANDB_AVAILABLE = True
except Exception:
  wandb = None
  WANDB_AVAILABLE = False
    
from math import log, sqrt, ceil

class JAGeRLoss(nn.Module):

  
  def __init__ (
    self, 
    Y: torch.Tensor, # true label columns
    K: int, # number of levels
    def_batch_size: int,
    steps_per_epoch: int | None = None,  # per-rank micro-batch steps per epoch (len(dl_train)); if None, computed from N/def_batch_size
    joint: bool = True,
    mixture: bool = True,
    conf_gating: bool = True,
    reassignment: bool = True,
    level_offset: int = 0, 
    λ0: float = 1, # initial value for λ
    α: float = 2, # initial value for α
    C: float = 1, # initial value for C
    softplus: bool = False, # whether to use softplus activation
    debug: bool = False,
    log_to_wandb: bool = False
    ):
      
      
      def _trunc_disc_norm_grid(K):
        SD_φ: tuple = (0.25541281188299525, 0.4479398673070137, 0.4887300185352654, 0.4978993024239318, 0.4996875664596105, 0.49996397161691486) # precomputed .5/φ(K)^2 for K=4..10; for K>10 approximate with .5
        _kk = torch.arange(K, device=dev, dtype=torch.float64).tile((K,1))
        return (
          (_kk - _kk.T).tile(K,1,1).permute(2,1,0) * (
            (_kk + _kk.T).tile(K,1,1).permute(2,1,0) * (SD_φ[K-4] if K<=10 else .5) -\
            _kk.tile(K,1,1)
          )
        ).exp_().sum(1).reciprocal_().T
          
          
      def _level_comb_counts(Y, H, K):
        base = torch.full((H,), K, dtype=torch.long, device=dev)
        exp = torch.arange(H, dtype=torch.long, device=dev)
        factors = torch.pow(base, exp)                  # [H] integer powers
        flat_idx = (Y * factors).sum(dim=1)
        counts_flat = torch.bincount(flat_idx, minlength=K**H)
        return counts_flat.view([K]*H)

        
      super().__init__()
      
      # logging toggle (safe default = off)
      self.log_to_wandb = bool(log_to_wandb)
      self._wandb_run_active = WANDB_AVAILABLE and getattr(wandb, "run", None) is not None

      import os
      self._debug = debug or os.environ.get("JAGeR_DEBUG", "0") == "1"
      self._eps = 1e-12

      if α <= 1:
        raise ValueError("α must be greater than 1.")
      if λ0 <= 0:
        raise ValueError("λ0 must be positive.")
      if K < 4:
        raise ValueError("K must be at least 4.")
      if (reassignment or conf_gating) and not mixture:
        raise ValueError("Competitor reassignment and confidence gating require 'mixture' to be True.")
      
      self.joint = joint
      self.mixture = mixture
      self.reassignment = reassignment
      self.conf_gating = conf_gating
      
      self.Y = (Y - level_offset).long().to(dev)
      self.N, self.H = self.Y.shape
      
      self.K = K
      self.logK = log(K)
      
      self.λ0 = λ0
      self.register_buffer(
        'λ',
        torch.full((self.N,) if joint else (self.N, self.H), λ0, dtype=torch.float64, device=dev)
      )
      self.α = α
      self.softplus = softplus
      
      
      if joint:
        self.KpowH = K**self.H
        self.HlogK = self.H * self.logK
        
        
      if mixture:
        π = _trunc_disc_norm_grid(K)
        self.Kπ_1 = K * π - 1
        self.register_buffer(
          'ρ',
          torch.zeros((self.N, self.H), dtype=torch.float64, device=dev)
        )
        
        #--- Setup for ρ estimation

        # Numerical constants
        self.sqrt3 = sqrt(3)
        _13rd = self._13rd = 1/3

        Mn, Vn, Sn, Kn = self._cent_moments(π)
        
        Mu_raw = torch.tensor([
          1,
          (K-1)/2,
          (K-1)*(2*K-1)/6,
          (K-1)**2*(K)/4,
          (K-1)*(2*K-1)*(3*(K-1)**2+3*K-4)/30
        ], dtype=torch.float64, device=dev)

        ### Mean ###
        δm = self.δm = Mn - Mu_raw[1]
        δm_sq = self.δm_sq = self.δm.square()
        δm_cu2 = self.δm_cu2 = δm_sq*self.δm*2
        δm_qu = self.δm_qu = δm_sq.square()

        eq_m1 = self.eq_m1 = torch.isclose(δm, torch.zeros_like(δm))

        ### Variance ###
        Vu = self.Vu = Mu_raw[2] - Mu_raw[1]**2
        δv = self.δv = Vn - Vu
        # Depressing shift
        self.ρ0 = (δv / δm_sq + 1)*.5
        self.ρ0[eq_m1] = Vu/δv[eq_m1]
        ρ0 = self.ρ0
        r0 = self.r0 = Vu/δm_sq + ρ0.square()

        ### Skewness ###
        Sn /= δm_cu2

        ### Kurtosis ###
        self.Ku = (Mu_raw[4] - 4*Mu_raw[3]*Mu_raw[1] + 6*Mu_raw[2]*Mu_raw[1].square() - 3*Mu_raw[1].square().square()).tile(K)
        self.δK = Kn - self.Ku
        self.Ku[~eq_m1] /= 3*δm_qu[~eq_m1]
        self.δK[~eq_m1] /= 3*δm_qu[~eq_m1]
        Ku = self.Ku
        δK = self.δK
        # Numerator coefficients (coef3 = 0, coef4 = 1)
        S = 8*_13rd*Sn + 2*r0
        self.coef_K = torch.stack((
          -Ku - ρ0*(δK - (ρ0-1)*S + 5*((ρ0-2*_13rd)**3 - 1/27)),
          2*(ρ0-.5)*(S - 6*(ρ0-.5).square() - .5*_13rd) - δK,
          S - 8*(ρ0-.5).square() - 2*_13rd
        ))
        
        
      if conf_gating:
        # Mean ρ setup
        self.def_batch_size = def_batch_size
        self.steps_per_epoch = int(steps_per_epoch) if steps_per_epoch is not None else ceil(self.N / self.def_batch_size)
        self.level_counts = self._level_counts(self.Y, self.H, K)
        if dist.is_available() and dist.is_initialized():
          self.level_counts = self.level_counts.to(torch.long)
          dist.all_reduce(self.level_counts, op=dist.ReduceOp.SUM)
        self.register_buffer(
          'cumul_ρ',
          torch.zeros_like(self.level_counts, dtype=torch.float64, device=dev)
        )
      
      
      # Margins according to Cao et al. "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss", 2019
      _base = (
        (_level_comb_counts(self.Y, self.H, K).add(1) if joint else self._level_counts(self.Y, self.H, K))
          .to(torch.float64)
          .rsqrt().sqrt()
          .mul(C)
      )
      self.thresholds = (
        _base
          .unsqueeze(0)
          .expand(self.N, *_base.shape)
          .contiguous()
          .to(dev)
      )


  def _level_counts(self, Y, H, K):
    counts = torch.zeros((H, K), dtype=torch.long, device=Y.device)
    for h in range(H):
      counts[h] = torch.bincount(Y[:, h], minlength=K)
    return counts  


  def _outer_sum(self, x: torch.Tensor, flat: bool = True) -> torch.Tensor:
    B, H, K = x.shape
    joint = x[:, 0, :] # (B, K)
    for h in range(1, H):
      joint = (joint.unsqueeze(-1) + x[:, h, :].unsqueeze(1)).reshape(B, -1)
    if flat:
      return joint
    # reshape back to (B, K, .., K)
    return joint.view(B, *([K] * H))


  def _cent_moments(self, probs):
    # Compute raw moments
    k = torch.arange(self.K, dtype=probs.dtype, device=probs.device)
    kk = torch.stack([k**i for i in range(5)], dim=0) 
    M_raw = torch.tensordot(probs, kk, dims=([-1], [1])) 
    M_raw = M_raw.movedim(-1, 0)
    var = M_raw[2] - M_raw[1]**2
    skew = M_raw[3] - 3*M_raw[1]*var - M_raw[1]**3
    kurt = M_raw[4] - 4*M_raw[3]*M_raw[1] + 6*M_raw[2]*M_raw[1]**2 - 3*M_raw[1]**4
    return (M_raw[1], var, skew, kurt)
      
  def _cbrt(self, x: torch.Tensor):
    return x.sign() * x.abs().pow(self._13rd)

  def _marginals_from_log_joint(self, pred_log_prob: torch.Tensor) -> torch.Tensor:
    H = pred_log_prob.dim() - 1  # number of heads
    marginals_log = [
      pred_log_prob.logsumexp(dim=tuple(d for d in range(1, H+1) if d != h+1))  # (B, K)
      for h in range(H)
    ]
    return torch.stack(marginals_log, dim=1).exp()  # (B, H, K)

  def _ce(self, p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    return -(p * log_q).sum(dim=-1)

  def _estimate_ρ(self, ν, β2, probs):

    ν = ν.long()

    # Gather per-ν constants (broadcast to (B, H))
    msk = self.eq_m1[ν]
    δv = self.δv[ν]  # (B, H)
    Ku = self.Ku[ν]
    δK = self.δK[ν]
    ρ0 = self.ρ0[ν]
    r0 = self.r0[ν]
    Kπ_1 = self.Kπ_1[ν]
    logK = self.logK

    ρ = torch.empty_like(ν, dtype=torch.float64)  # Initialize ρ tensor

    # Case 1: equal mean (quadratic)
    b = δK[msk] / (2 * δv[msk].square() * β2[msk].clamp_min(1e-12))
    D = (b.square() + (Ku[msk] - δK[msk]*ρ0[msk]) / (δv[msk].square() * β2[msk].clamp_min(1e-12))).clamp_min(0).sqrt()
    kurt_quad_roots = (b - ρ0[msk]).unsqueeze(-1) + torch.stack((-D, D), dim=1)  # (M,2)
    kurt_quad_roots = kurt_quad_roots.clamp(0, 1)
    
    log_υ_roots = (kurt_quad_roots.unsqueeze(-1) * Kπ_1[msk].unsqueeze(1)).log1p() - logK  # (M,2,K)
    
    ce = self._ce(probs[msk].unsqueeze(1), log_υ_roots)  # (M,2)
    idx = ce.argmin(dim=1)  # (M,)
    
    ρ[msk] = kurt_quad_roots.gather(1, idx.unsqueeze(-1)).squeeze(-1)


    # Case 2: general (quartic)
    coef_K0 = self.coef_K[0][ν]
    coef_K1 = self.coef_K[1][ν]
    coef_K2 = self.coef_K[2][ν]
    _13rd = self._13rd
    _cbrt = self._cbrt
    
    den = 1 + β2.clamp_min(1e-12) * _13rd
    p = -(coef_K2 - 2 * β2.clamp_min(1e-12) * r0 * _13rd) * 0.5 / den
    q = (coef_K1 * 0.5) / den
    r = -(coef_K0 + β2.clamp_min(1e-12) * r0.square() * _13rd) / den

    p13 = p * _13rd
    p2 = p13 * p13
    f = r * _13rd - p2
    g = p13 * (2 * p2 - r) + p * r - 0.5 * q.square()
    h = 0.25 * g.square() + f * f * f

    mask_h = h <= 0
    # h <= 0 branch
    l = f.abs().sqrt()
    acos_arg = torch.clamp(0.5 * g / (f * l), -1.0, 1.0)
    m_res = (acos_arg.acos() * _13rd).cos()
    cr1 = 2 * l * m_res - p13
    # h > 0 branch
    sqrt_h = h.clamp_min(0).sqrt()
    cr2 = _cbrt(-0.5 * g + sqrt_h) + _cbrt(-0.5 * g - sqrt_h) - p13
    cr = torch.where(mask_h, cr1, cr2)

    s = (2 * (p + cr)).clamp_min(0).sqrt()
    s_nz = s != 0
    t = torch.empty_like(s)
    t[s_nz] = -q[s_nz] / s[s_nz]
    t[~s_nz] = cr[~s_nz] * cr[~s_nz] + r[~s_nz]
    s = s*.5

    s2 = s.square()

    kurt_quart_roots = torch.stack([
      -s - (s2 - cr - t).clamp_min(0).sqrt(),
      -s + (s2 - cr - t).clamp_min(0).sqrt(),
      s - (s2 - cr + t).clamp_min(0).sqrt(),
      s + (s2 - cr + t).clamp_min(0).sqrt()
    ], dim=-1) + ρ0.unsqueeze(-1) 
    kurt_quart_roots = kurt_quart_roots.clamp(0, 1)

    log_υ_roots = (kurt_quart_roots.unsqueeze(-1) * Kπ_1.unsqueeze(-2)).log1p() - logK  
    ce = self._ce(probs.unsqueeze(-2), log_υ_roots)
    idx = ce.argmin(dim=-1)  
    ρ[~msk] = kurt_quart_roots.gather(-1, idx.unsqueeze(-1)).squeeze(-1)[~msk]

    return ρ


  def forward(self, y_pred, ids, update_state: bool = True, global_step: int | None = None):
    # y_pred: (B, H, n_levels)
    λ0, K, H, α, logK = self.λ0, self.K, self.H, self.α, self.logK
    Y = self.Y[ids]
    B = y_pred.shape[0]
    λ = self.λ[ids]
    
    joint, mixture, reassignment, conf_gating = self.joint, self.mixture, self.reassignment, self.conf_gating
    
    if joint:
      KpowH, HlogK = self.KpowH, self.HlogK

    if self.softplus:
      y_pred = F.softplus(y_pred)
      
    y_pred = (F.normalize(y_pred, dim=(1,2), p=1)*KpowH) if joint else (F.normalize(y_pred, dim=2, p=1)*K)
    

    with torch.no_grad():
      head_scores = y_pred / (λ[..., None, None] if joint else λ[..., None]) 
      log_p_h = head_scores.log_softmax(dim=2)
      p_h = log_p_h.exp() 

      if mixture:
        log_p_h_max, _mode = log_p_h.max(dim=2)
        mean, _var, _skew, _kurt = self._cent_moments(p_h) 
        ρ = self._estimate_ρ(_mode, _kurt / _var.square().clamp_min(self._eps), p_h)
      
      if conf_gating:
        # ρ gated update 
        level_counts_B = self._level_counts(Y, H, K)  # (H,K), local batch counts
        cumul_ρ_B = torch.zeros_like(self.cumul_ρ, dtype=self.cumul_ρ.dtype, device=self.cumul_ρ.device)
        cumul_ρ_B.scatter_add_(1, Y.T.to(self.cumul_ρ.device), self.ρ[ids].T.to(self.cumul_ρ.device, dtype=self.cumul_ρ.dtype))
        cumul_ρ = self.cumul_ρ - cumul_ρ_B
        mean_ρ_without_B_full = cumul_ρ / (self.level_counts - level_counts_B + 1)  # (H,K)
        mean_ρ_without_B = mean_ρ_without_B_full.gather(1, Y.T).T  # (B,H)
        assert (mean_ρ_without_B <= 1e0).all(), f"Mean ρ without B has values > 1: {mean_ρ_without_B}, cumul_ρ={cumul_ρ}, level_counts={self.level_counts}, level_counts_B={level_counts_B}"
        t = (global_step // self.steps_per_epoch) if (global_step is not None) else 0
        s_t = (global_step % self.steps_per_epoch) if (global_step is not None) else 0
        τ = s_t * self.def_batch_size / self.N + t
        γ = (τ + 1)**(-τ)
        ρ = γ * mean_ρ_without_B + (1 - γ) * ρ
        assert (ρ <= 1e0).all(), f"Estimated ρ has values > 1: {ρ}, mean_ρ_without_B={mean_ρ_without_B}"
      
      # λ update
      if mixture:
        Kπ_1_pred_max = self.Kπ_1[_mode]
        log_S_h = (ρ.unsqueeze(-1)*Kπ_1_pred_max).log1p() # -self.logK is added as per need
        min_idx = torch.where(
          _mode < K-_mode,
          torch.full_like(_mode, K-1),
          torch.zeros_like(_mode)
        ).long()
        log_S_min = log_S_h.take_along_dim(min_idx.unsqueeze(-1), 2).squeeze(-1)
        kl_div = (p_h * (log_p_h - log_S_h + logK)).sum(dim=2)
        if joint:
          kl_div = kl_div.sum(dim=1)
          log_S_min = log_S_min.sum(dim=1)
          u_bound = -(
            self._ce(p_h, log_p_h).sum(dim=1) +
            log_S_min - HlogK +
            log_p_h_max.sum(dim=1).exp() *
            (log_S_h.take_along_dim(_mode.unsqueeze(-1), 2).sum(dim=1).squeeze(-1) - log_S_min)
          )
        else:
          u_bound = -(
            self._ce(p_h, log_p_h) +
            log_S_min - logK +
            p_h.take_along_dim(_mode.unsqueeze(-1), 2).squeeze(-1) *
            (log_S_h.take_along_dim(_mode.unsqueeze(-1), 2).squeeze(-1) - log_S_min)
          )
      else: 
        kl_div = (p_h * (log_p_h + logK)).sum(dim=2)
        if joint:
          u_bound = HlogK
          kl_div = kl_div.sum(dim=1)
        else:
          u_bound = logK

      λt = λ0 * (1 - kl_div / (α * u_bound))
      λ_reg_loss = -.5*α*u_bound * (λt - λ0).square() / λ0

      
      # competitor assignment
      if mixture:
        Kπ_1_label = self.Kπ_1[Y]
        log_υ_h = (ρ.unsqueeze(-1) * (Kπ_1_label + ρ.unsqueeze(-1) * (Kπ_1_pred_max - Kπ_1_label))).log1p() if reassignment else log_S_h
        if joint:
          joint_log_υ = self._outer_sum(log_υ_h, flat=False)
          joint_log_υ = joint_log_υ.view(B, -1)
      
      
      thresholds = self.thresholds[ids]
      if joint:
        _1hot_label = torch.zeros_like(thresholds, dtype=thresholds.dtype)
        idx_label = (torch.arange(B, device=y_pred.device), *Y.unbind(1))
        _1hot_label[idx_label] = 1.
        if reassignment:
          _1hot_pred_max = torch.zeros_like(thresholds, dtype=thresholds.dtype)
          idx_pred_max = (torch.arange(B, device=y_pred.device), *_mode.unbind(1))
          _1hot_pred_max[idx_pred_max] = 1.
          _1hot_pred_max = _1hot_pred_max.view(B, -1)
        thresholds = thresholds.view(B, -1)
        _1hot_label = _1hot_label.view(B, -1)
        if reassignment:
          c = (
            thresholds - (thresholds + λt.unsqueeze(-1)*joint_log_υ).mul(_1hot_label) -
            (ρ.square().mean(dim=1).unsqueeze(-1) * thresholds + 
            λt.unsqueeze(-1)*self._outer_sum(ρ.square().unsqueeze(-1)*log_υ_h, flat=False).view(B, -1)).mul(_1hot_pred_max - _1hot_label)
          )
        elif mixture:
          c = thresholds - (thresholds + λt.unsqueeze(-1)*joint_log_υ).mul(_1hot_label)
        else:
          c = thresholds - thresholds.mul(_1hot_label)
      else:
        _1hot_label = F.one_hot(Y, K)
        if reassignment:
          _1hot_pred_max = F.one_hot(_mode, K)
          c = (
            thresholds - (thresholds + λt.unsqueeze(-1)*log_υ_h).mul(_1hot_label) -
            (ρ.square().unsqueeze(-1) * thresholds + 
            λt.unsqueeze(-1)*(ρ.square().unsqueeze(-1)*log_υ_h)).mul(_1hot_pred_max - _1hot_label)
          )
        elif mixture:
          c = thresholds - (thresholds + λt.unsqueeze(-1)*log_υ_h).mul(_1hot_label)
        else:
          c = thresholds - thresholds.mul(_1hot_label)
      
    
    y_label = y_pred.gather(2, Y.unsqueeze(-1))  
    y_pred = y_pred - y_label 
    if reassignment:
      y_pred = y_pred + ρ.square().unsqueeze(-1) * (y_pred.gather(2, _mode.unsqueeze(-1)) + y_label)
    if joint: 
      y_pred = self._outer_sum(y_pred, flat=False).view(B, -1) 
    diff_logits = y_pred + c 
    diff_logits_lam_fix = diff_logits /(λt.unsqueeze(1) if joint else λt.unsqueeze(-1))
    if mixture:
      diff_logits_lam_fix = diff_logits_lam_fix + (joint_log_υ if joint else log_υ_h)
    logsumexp = diff_logits_lam_fix.logsumexp(-1)
    loss_lam_fix = λt * logsumexp + λ_reg_loss
    
    if update_state:
      with torch.no_grad():
        if conf_gating:
          cumul_ρ_B_old = torch.zeros_like(self.cumul_ρ, dtype=self.cumul_ρ.dtype, device=self.cumul_ρ.device)
          cumul_ρ_B_old.scatter_add_(1, Y.T.to(self.cumul_ρ.device), self.ρ[ids].T.to(self.cumul_ρ.device, dtype=self.cumul_ρ.dtype))

          cumul_ρ_B_new = torch.zeros_like(self.cumul_ρ, dtype=self.cumul_ρ.dtype, device=self.cumul_ρ.device)
          cumul_ρ_B_new.scatter_add_(1, Y.T.to(self.cumul_ρ.device), ρ.T.to(self.cumul_ρ.device, dtype=self.cumul_ρ.dtype))

          delta = cumul_ρ_B_new - cumul_ρ_B_old

          if dist.is_available() and dist.is_initialized():
              dist.all_reduce(delta, op=dist.ReduceOp.SUM)

          self.cumul_ρ = self.cumul_ρ + delta
        
        if mixture:
          self.ρ[ids] = ρ
          if dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, (ids.detach().cpu(), ρ.detach().cpu()))
            for ids_g, ρ_g in gathered:
              if ids_g is None or len(ids_g) == 0:
                continue
              self.ρ[ids_g.to(self.ρ.device)] = ρ_g.to(self.ρ.device, dtype=self.ρ.dtype)
        
        self.λ[ids] = λt.clamp_min(0.0)

      
    if self.log_to_wandb and self._wandb_run_active:
      wandb.log({
          "jager/lambda_min": float(self.λ.min().item()),
          "jager/lambda_max": float(self.λ.max().item()),
          **({"loss/ρ": float(ρ.detach().mean().item())} if mixture else {})
      }, commit=False)
    
    return loss_lam_fix.mean()



class MultiHeadCELoss(nn.Module):

  def __init__(self, Y: torch.Tensor, K: int, level_offset: int = 1, label_smoothing: float = 0.0):
    super().__init__()
    self.register_buffer("Y", Y.to(torch.long))
    self.K = K
    self.level_offset = level_offset
    self.H = Y.shape[1]
    self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="mean")

  def forward(self, y_pred: torch.Tensor, ids: torch.Tensor, update_state: bool = False) -> torch.Tensor:
    B, H, K = y_pred.shape
    assert H == self.H and K == self.K, "shape mismatch for CE loss"
    Y = self.Y[ids]
    idx = Y - self.level_offset
    if (idx < 0).any() or (idx >= self.K).any():
        raise ValueError(
            f"Labels out of range for MultiHeadCELoss given level_offset={self.level_offset}, K={self.K}."
        )
    losses = []
    for h in range(H):
        losses.append(self.ce(y_pred[:, h, :], idx[:, h]))
    return torch.stack(losses, dim=0).mean()