import torch
import torch.nn as nn
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
    def_batch_size: int, # effective global micro-batch size seen per synchronized forward
    steps_per_epoch: int | None = None,  # synchronized micro-batch steps per epoch; if None, computed from stats rows / def_batch_size
    stats_ids: torch.Tensor | None = None, # row ids whose labels define train-only class stats; defaults to all rows
    joint: bool = True,
    mixture: bool = True,
    conf_gating: bool = True,
    reassignment: bool = True,
    level_offset: int = 0, 
    λ0: float = 1, # initial value for λ
    λmin: float | None = None, # overrides α
    α: float = 2,
    C: float = 1,
    debug: bool = True,
    log_to_wandb: bool = False,
    ):
      device = Y.device

      def _trunc_disc_norm_grid(K):
        φ_sq_inv_halved: tuple[float, ...] = (0.25541281188299525, 0.4479398673070137, 0.4887300185352654, 0.4978993024239318, 0.4996875664596105, 0.49996397161691486) # precomputed .5/φ(K)^2 for K=4..10; for K>10 approximate with .5
        k = torch.arange(K, device=device, dtype=torch.int16)
        return F.softmax(-(φ_sq_inv_halved[K-4] if K<=10 else .5) * (k[:,None] - k).square(), dim=1)


        
      super().__init__()
      
      # logging toggle 
      self.log_to_wandb = bool(log_to_wandb)
      self._wandb_run_active = WANDB_AVAILABLE and getattr(wandb, "run", None) is not None

      import os
      self._debug = debug or os.environ.get("JAGeR_DEBUG", "0") == "1"
      self.ε = 1e-8

      if α <= 1:
        raise ValueError("α must be greater than 1.")
      if λ0 <= 0:
        raise ValueError("λ0 must be positive.")
      if not (λmin is None or 0 < λmin < λ0):
        raise ValueError("λmin must be positive and less than λ0")
      if K < 4:
        raise ValueError("K must be at least 4.")
      if (reassignment or conf_gating) and not mixture:
        raise ValueError("Competitor reassignment and confidence gating require 'mixture' to be True.")
      
      self.joint = joint 
      self.mixture = mixture
      self.reassignment = reassignment
      self.conf_gating = conf_gating
      
      self.N, self.H = self.Y.shape
      if stats_ids is None:
        stats_Y = self.Y
      else:
        stats_ids = stats_ids.to(device=device, dtype=torch.long).reshape(-1)
        if stats_ids.numel() == 0:
          raise ValueError("stats_ids must be non-empty.")
        if int(stats_ids.min().item()) < 0 or int(stats_ids.max().item()) >= self.N:
          raise ValueError("stats_ids contains indices outside the range of Y.")
        stats_Y = self.Y[stats_ids]
      self.N_stats = int(stats_Y.shape[0])
      
      self.K = K
      self.logK = log(K)
      
      self.Y = (Y - level_offset).long().to(device)
      self.λ0 = λ0
      self.λ: torch.Tensor
      self.register_buffer(
        'λ',
        torch.full((self.N,) if joint else (self.N, self.H), λ0, dtype=torch.float64, device=device)
      )
      self.α = α if λmin is None else λ0 / (λ0 - λmin)
      
      
      if joint:
        KpowH = self.KpowH = K**self.H
        self.HlogK = self.H * self.logK
        
      if mixture:
        self.ρ: torch.Tensor
        self.register_buffer(
          'ρ',
          torch.zeros((self.N) if joint else (self.N, self.H), dtype=torch.float64, device=device)
        )
        
        if joint:
          φ_sq_inv_halved: torch.Tensor = torch.tensor([0.25541281188299525, 0.4479398673070137, 0.4887300185352654, 0.4978993024239318, 0.4996875664596105, 0.49996397161691486], dtype=torch.float64, device=device) # precomputed .5/φ(K)^2 for K=4..10; for K>10 approximate with .5
          k = torch.arange(K, device=device, dtype=torch.int16)
          self.D = D = (k[:,None] - k).square_()
          self.R_max = R_max = (K-1)**2
          z = (-D * (φ_sq_inv_halved[K-4] if K<=10 else .5)).exp_().sum(1)
          Z = z.clone()
          for _ in range(1, self.H):
            Z = Z.unsqueeze(-1) * z
          ker_shell = (-(φ_sq_inv_halved[K-4] if K<=10 else .5) * torch.arange(self.H * R_max + 1, device=device, dtype=torch.float64)).exp_()
          self.Kπminus1 = KpowH*ker_shell.view(*[1]*self.H, self.H*R_max+1)/Z.unsqueeze(-1) - 1
          try:
            self.Kπminus1_rec = self.Kπminus1.reciprocal()
            assert not (self.Kπminus1_rec.isnan().any() or self.Kπminus1_rec.isinf().any()), "Kπminus1 contains zero values, cannot compute reciprocal."
          except Exception as e:
            print("Error computing reciprocal of Kπminus1:", e)
            print("Kπminus1:", self.Kπminus1)
            raise
          self.factors = K ** torch.arange(self.H, device=device, dtype=torch.long)
        else:
          π = _trunc_disc_norm_grid(K)
          self.Kπminus1 = K*π - 1
          

          #--- Setup for marginal ρ estimation via kurtosis matching ---#

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
          ], dtype=torch.float64, device=device)

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
        self.steps_per_epoch = int(steps_per_epoch) if steps_per_epoch is not None else ceil(self.N_stats / self.def_batch_size)
        if joint:
          self.level_comb_counts = self._level_comb_counts(stats_Y, K).view(-1)
        else:
          self.level_counts = self._level_counts(stats_Y, K).to(torch.long)
        self.cumul_ρ: torch.Tensor
        self.register_buffer(
          'cumul_ρ',
          torch.zeros_like(self.level_comb_counts if joint else self.level_counts, dtype=torch.float64, device=device)
        )
      
      
      # LDAM Margins according to Cao et al. "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss", 2019
      _base = (
        (self._level_comb_counts(stats_Y, K).add(1) if joint else (self.level_counts if conf_gating else self._level_counts(stats_Y, K)))
          .to(torch.float64)
          .rsqrt().sqrt()
          .mul(C)
      )
      self._base_thresholds: torch.Tensor
      self.register_buffer('_base_thresholds', _base.to(device))


  def _level_comb_counts(self, Y, K):
    H = Y.shape[1]
    base = torch.full((H,), K, dtype=torch.long, device=Y.device)
    exp = torch.arange(H, dtype=torch.long, device=Y.device)
    factors = torch.pow(base, exp)
    flat_idx = (Y * factors).sum(dim=1)
    counts_flat = torch.bincount(flat_idx, minlength=K**H)
    return counts_flat.view([K]*H)


  def _level_counts(self, Y, K):
    H = Y.shape[1]
    counts = torch.zeros((H, K), dtype=torch.long, device=Y.device)
    counts.scatter_add_(1, Y.T, torch.ones_like(Y.T))
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
    kk = k.pow(torch.arange(5, device=k.device, dtype=k.dtype).unsqueeze(1))
    M_raw = torch.tensordot(probs, kk, dims=([-1], [1]))  # type: ignore[arg-type]
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

  def _all_gather_equal(self, x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
      return x
    world_size = dist.get_world_size()
    if world_size == 1:
      return x
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=0)

  def _unique_step_ids(self, ids: torch.Tensor) -> torch.Tensor:
    ids_all = self._all_gather_equal(ids.detach().to(self.λ.device, dtype=torch.long))
    return torch.unique(ids_all, sorted=True)

  def _aggregate_state_updates(self, ids: torch.Tensor, λt: torch.Tensor, ρ: torch.Tensor | None = None):
    ids_all = self._all_gather_equal(ids.detach().to(self.λ.device, dtype=torch.long))
    λ_all = self._all_gather_equal(λt.detach().to(self.λ.device, dtype=self.λ.dtype))
    ρ_all = self._all_gather_equal(ρ.detach().to(self.ρ.device, dtype=self.ρ.dtype)) if ρ is not None else None

    unique_ids, inverse = torch.unique(ids_all, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse, minlength=unique_ids.numel()).to(dtype=self.λ.dtype, device=self.λ.device)
    count_shape = (counts.shape[0],) + (1,) * max(0, λ_all.dim() - 1)

    λ_sum = torch.zeros((unique_ids.numel(),) + λ_all.shape[1:], dtype=λ_all.dtype, device=λ_all.device)
    λ_sum.index_add_(0, inverse, λ_all)
    λ_mean = λ_sum / counts.view(count_shape)

    ρ_mean = None
    if ρ_all is not None:
      counts_ρ = counts.to(dtype=self.ρ.dtype, device=self.ρ.device)
      count_shape_ρ = (counts_ρ.shape[0],) + (1,) * max(0, ρ_all.dim() - 1)
      ρ_sum = torch.zeros((unique_ids.numel(),) + ρ_all.shape[1:], dtype=ρ_all.dtype, device=ρ_all.device)
      ρ_sum.index_add_(0, inverse, ρ_all)
      ρ_mean = ρ_sum / counts_ρ.view(count_shape_ρ)

    return unique_ids, λ_mean, ρ_mean

  def _estimate_ρ_mvar(self, ν, probs):
    
    pass

  def _estimate_ρ_uvar(self, ν, β2, probs):

    ν = ν.long()
    β2 = β2.clamp_min(self.ε)

    # Gather per-ν constants (broadcast to (B, H))
    msk = self.eq_m1[ν]
    δv = self.δv[ν]  # (B, H)
    Ku = self.Ku[ν]
    δK = self.δK[ν]
    ρ0 = self.ρ0[ν]
    r0 = self.r0[ν]
    Kπminus1 = self.Kπminus1[ν]
    logK = self.logK

    ρ = torch.empty_like(ν, dtype=torch.float64)  # Initialize ρ tensor


    # Case 1: equal mean (quadratic)
    b = δK[msk] / (2 * δv[msk].square() * β2[msk])
    D = (b.square() + (Ku[msk] - δK[msk]*ρ0[msk]) / (δv[msk].square() * β2[msk])).clamp_min(0).sqrt()
    kurt_quad_roots = (b - ρ0[msk]).unsqueeze(-1) + torch.stack((-D, D), dim=1)  # (M,2)
    kurt_quad_roots = kurt_quad_roots.clamp(0, 1)
    
    log_υ_roots = (kurt_quad_roots.unsqueeze(-1) * Kπminus1[msk].unsqueeze(1)).log1p() - logK  # (M,2,K)
    
    ce = self._ce(probs[msk].unsqueeze(1), log_υ_roots)  # (M,2)
    idx = ce.argmin(dim=1)  # (M,)
    
    ρ[msk] = kurt_quad_roots.gather(1, idx.unsqueeze(-1)).squeeze(-1)


    # Case 2: general (quartic) — only compute for ~msk elements
    nmsk = ~msk
    if nmsk.any():
      coef_K0 = self.coef_K[0][ν][nmsk]
      coef_K1 = self.coef_K[1][ν][nmsk]
      coef_K2 = self.coef_K[2][ν][nmsk]
      β2_n = β2[nmsk]
      r0_n = r0[nmsk]
      ρ0_n = ρ0[nmsk]
      Kπminus1_n = Kπminus1[nmsk]
      probs_n = probs[nmsk]
      _13rd = self._13rd
      _cbrt = self._cbrt
      
      den = 1 + β2_n * _13rd
      p = -(coef_K2 - 2 * β2_n * r0_n * _13rd) * .5 / den
      q = (coef_K1 * .5) / den
      r = -(coef_K0 + β2_n * r0_n.square() * _13rd) / den

      p13 = p * _13rd
      p2 = p13 * p13
      f = r * _13rd - p2
      g = p13 * (2 * p2 - r) + p * r - .5 * q.square()
      h = .25 * g.square() + f * f * f

      mask_h = h <= 0
      # h <= 0 branch
      l = f.abs().sqrt()
      acos_arg = torch.clamp(.5 * g / (f * l), -1.0, 1.0)
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
      ], dim=-1) + ρ0_n.unsqueeze(-1) 
      kurt_quart_roots = kurt_quart_roots.clamp(0, 1)

      log_υ_roots = (kurt_quart_roots.unsqueeze(-1) * Kπminus1_n.unsqueeze(-2)).log1p() - logK  
      ce = self._ce(probs_n.unsqueeze(-2), log_υ_roots)
      idx = ce.argmin(dim=-1)  
      ρ[nmsk] = kurt_quart_roots.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

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
      if mixture:
        factors = self.factors
        R_max = self.R_max
      
    y_pred = F.softplus(y_pred)
      
    y_pred = F.normalize(y_pred, dim=(1,2), p=1)*KpowH if joint else F.normalize(y_pred, dim=2, p=1)*K  # type: ignore[arg-type]
    

    with torch.no_grad():
      log_p_h = (y_pred / (λ[..., None, None] if joint else λ[..., None])).log_softmax(dim=2)
      p_h = log_p_h.exp() 

      if mixture:
        log_p_h_max, _mode = log_p_h.max(dim=2)
        _mode_unsq = _mode.unsqueeze(-1) # (B, H, 1)
        if joint:
          p_shell_h = torch.zeros((B, H, R_max+1), dtype=torch.float64, device=y_pred.device)
          p_shell_h.scatter_add_(2, self.D[_mode].long(), p_h)
          p_shell = p_shell_h[:, 0, :]
          for h in range(1, H):
            p_shell_old = p_shell
            p_shell_old = F.pad(p_shell_old, (R_max, R_max))
            p_shell_old = p_shell_old.unfold(-1, R_max+1, 1)
            p_shell = (p_shell_old*p_shell_h[:, h, :].flip(-1).unsqueeze(1)).sum(-1) # convolution via flipped kernel
          Kπminus1_rec_pred_mode = self.Kπminus1_rec[*_mode.T]
          P_max = log_p_h_max.sum(-1).exp_()
          ρ = ((KpowH * P_max - 1)*Kπminus1_rec_pred_mode[:,0]).clamp_(0,1)
          for _ in range(10):
            den = (ρ.unsqueeze(-1) + Kπminus1_rec_pred_mode).unsqueeze_(0).pow(torch.arange(1, 4, device=p_h.device, dtype=p_h.dtype)[..., None, None])
            der = (p_shell.unsqueeze(0)/den).sum(-1)
            if der[0].abs().le(self.ε).all(): break
            den = der[1].square() - der[0]*der[2]
            sgn = den.sign()
            sgn = sgn.where(sgn.ne(0), 1)
            den.abs_().clamp_min_(self.ε).mul_(sgn)
            Δρ = der[0]*der[1]/den # Halley's step
            if Δρ.abs().le(self.ε).all(): break
            ρ = (ρ + Δρ).clamp_(0, 1)
          
        else:
          mean, _var, _skew, _kurt = self._cent_moments(p_h) 
          ρ = self._estimate_ρ_uvar(_mode, _kurt / _var.square().clamp_min(self.ε), p_h)
      
        if conf_gating:
          # ρ gated update 
          unique_ids_gate = self._unique_step_ids(ids) if update_state else ids.detach().to(self.λ.device, dtype=torch.long)
          Y_gate = self.Y[unique_ids_gate]
          old_ρ_gate = self.ρ[unique_ids_gate]
          level_counts_B_gate = self._level_comb_counts(Y_gate, K).view(-1) if joint else self._level_counts(Y_gate, K)
          cumul_ρ_B_gate = torch.zeros_like(self.cumul_ρ, dtype=self.cumul_ρ.dtype)
          if joint:
            Y_gate = (Y_gate * factors).sum(dim=1).T
          cumul_ρ_B_gate.scatter_add_(-1, Y_gate.T, old_ρ_gate.to(dtype=self.cumul_ρ.dtype))
          cumul_ρ = self.cumul_ρ - cumul_ρ_B_gate
          mean_ρ_without_B_full = cumul_ρ / ((self.level_comb_counts if joint else self.level_counts) - level_counts_B_gate + 1)
          mean_ρ_without_B = mean_ρ_without_B_full.gather(0, (Y*factors).sum(1)) if joint else mean_ρ_without_B_full.gather(1, Y.T).T
          t = (global_step // self.steps_per_epoch) if (global_step is not None) else 0
          s_t = (global_step % self.steps_per_epoch) if (global_step is not None) else 0
          τ = s_t * self.def_batch_size / max(self.N_stats, 1) + t
          γ = (τ + 1)**(-τ)
          ρ = γ * mean_ρ_without_B + (1 - γ) * ρ
      
      # λ update
      Ent_p_h = self._ce(p_h, log_p_h)
      if mixture:
        Kπminus1_pred_mode: torch.Tensor = self.Kπminus1[*_mode.T if joint else _mode]
        log_S = (ρ.unsqueeze(-1)*Kπminus1_pred_mode).log1p() # logK is subtracted as per need
        _cond: torch.Tensor = _mode < K-_mode  # type: ignore[assignment]
        min_idx = torch.where(_cond, K-1, 0)
        if joint:
          log_S_min = log_S.gather(-1, (min_idx-_mode).square().sum(-1, keepdim=True)).squeeze_(-1)
          log_S_at_mode = log_S[:,0]
          kl_div = -Ent_p_h.sum(-1) - (p_shell*log_S).sum(-1) + HlogK
          u_bound = -(
            Ent_p_h.sum(-1) +
            log_S_min - HlogK +
            P_max *
            (log_S_at_mode - log_S_min)
          )
        else:
          log_S_min = log_S.take_along_dim(min_idx.unsqueeze_(-1), 2).squeeze_(-1)
          log_S_at_mode = log_S.take_along_dim(_mode_unsq, 2).squeeze_(-1)
          kl_div = (p_h * (log_p_h - log_S)).sum(-1) + logK
          u_bound = -(
            Ent_p_h +
            log_S_min - logK +
            p_h.take_along_dim(_mode_unsq, 2).squeeze_(-1) *
            (log_S_at_mode - log_S_min)
          )
          if self._debug and (u_bound.isnan().any() or u_bound.less(0).any() or kl_div.isnan().any() or kl_div.less(0).any()):
            torch.set_printoptions(precision=15, sci_mode=False)
            print("Debug Info:")
            print(f"log_p_h: {log_p_h}")
            print(f"log_S_h: {log_S}")
            print(f"log_S_min: {log_S_min}")
            print(f"p_h: {p_h}")
            print(f"kl_div: {kl_div}")
            print(f"u_bound: {u_bound}")
            raise ValueError("Negative or NaN kl_div or u_bound encountered. See debug info for details.")
          if self._debug and kl_div.gt(u_bound+self.ε).any():
            torch.set_printoptions(precision=15, sci_mode=False)
            print("Debug Info:")
            print(f"log_p_h: {log_p_h}")
            print(f"log_S_h: {log_S}")
            print(f"log_S_min: {log_S_min}")
            print(f"p_h: {p_h}")
            print(f"kl_div: {kl_div}")
            print(f"u_bound: {u_bound}")
            raise ValueError("kl_div exceeds u_bound. See debug info for details.")
      else: 
        kl_div = -Ent_p_h + logK
        if joint:
          u_bound = HlogK
          kl_div = kl_div.sum(-1)
        else:
          u_bound = logK


      λt = λ0 * (1 - kl_div / (α * u_bound))
      if self._debug and ((λt + self.ε < λ0*(1 - 1/α)).any() or λt.isnan().any()):
        torch.set_printoptions(precision=15, sci_mode=False)
        print("Debug Info:")
        print(f"kl_div: {kl_div}")
        print(f"u_bound: {u_bound}")
        print(f"λt: {λt + self.ε}")
        print(f"Expected λt range: [{λ0*(1 - 1/α)}, {λ0}]")
        raise ValueError("λt out of expected range or NaN. See debug info for details.")
      λ_reg_loss = -.5*α*u_bound * (λt - λ0).square() / λ0
      λt_unsq = λt.unsqueeze(-1)

      if reassignment:
        ρ_sq = ρ.square()

      # competitor assignment
      if mixture:
        if reassignment:
          Kπminus1_label: torch.Tensor = self.Kπminus1[*Y.T if joint else Y]
        if joint:
          r_mode = self._outer_sum(self.D[_mode].to(torch.long))
          if reassignment:
            r_label = self._outer_sum(self.D[Y].to(torch.long))
            Kπminus1_label: torch.Tensor = Kπminus1_label.gather(1, r_label)
            Kπminus1_pred_mode: torch.Tensor = Kπminus1_pred_mode.gather(1, r_mode)
        if reassignment:
          log_υ = (ρ.unsqueeze(-1) * (Kπminus1_label + ρ.unsqueeze(-1) * (Kπminus1_pred_mode - Kπminus1_label))).log1p() # type: ignore[reportOperatorIssue]
        else:
          if joint:
            log_S = log_S.gather(1, r_mode)
          log_υ = log_S
      
      thresholds = self._base_thresholds.unsqueeze(0).expand(B, *self._base_thresholds.shape)
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
          c = thresholds - (thresholds + λt_unsq*log_υ).mul(
            _1hot_label + ρ_sq.unsqueeze(-1).mul(_1hot_pred_max - _1hot_label))
        elif mixture:
          c = thresholds - (thresholds + λt_unsq*log_υ).mul(_1hot_label)
        else:
          c = thresholds - thresholds.mul(_1hot_label)
      else:
        _1hot_label = F.one_hot(Y, K)
        if reassignment:
          _1hot_pred_max = F.one_hot(_mode, K)
          c = (
            thresholds - (thresholds + λt_unsq*log_υ).mul(_1hot_label) -
            (ρ_sq.unsqueeze(-1) * thresholds + 
            λt_unsq*(ρ_sq.unsqueeze(-1)*log_υ)).mul(_1hot_pred_max - _1hot_label)
          )
        elif mixture:
          c = thresholds - (thresholds + λt_unsq*log_υ).mul(_1hot_label)
        else:
          c = thresholds - thresholds.mul(_1hot_label)
      
    
    y_label = y_pred.gather(2, Y.unsqueeze(-1))  
    y_pred = y_pred - y_label 
    if reassignment:
      y_pred = y_pred - (ρ_sq[..., None, None] if joint else ρ_sq[..., None]) * y_pred.gather(2, _mode_unsq)
    if joint: 
      y_pred = self._outer_sum(y_pred, flat=False).view(B, -1) 
    diff_logits = y_pred + c 
    diff_logits_lam_fix = diff_logits / λt_unsq
    if mixture:
      diff_logits_lam_fix = diff_logits_lam_fix + log_υ
    logsumexp = diff_logits_lam_fix.logsumexp(-1)
    loss_lam_fix = λt * logsumexp
    
    if update_state:
      with torch.no_grad():
        unique_ids_update, λt_update, ρ_update = self._aggregate_state_updates(ids, λt, ρ if mixture else None)

        if conf_gating:
          Y_update = self.Y[unique_ids_update]
          old_ρ_update = self.ρ[unique_ids_update]
          assert ρ_update is not None
          delta = torch.zeros_like(self.cumul_ρ, dtype=self.cumul_ρ.dtype)
          delta.scatter_add_(-1, (Y_update*factors).sum(1) if joint else Y_update.T, (ρ_update - old_ρ_update).to(dtype=self.cumul_ρ.dtype))
          self.cumul_ρ = self.cumul_ρ + delta

        self.λ[unique_ids_update] = λt_update
        if mixture and ρ_update is not None:
          self.ρ[unique_ids_update] = ρ_update
      
    self._last_lambda_reg = float(λ_reg_loss.mean().item())

    if self.log_to_wandb and self._wandb_run_active:
      wandb.log({  # type: ignore[union-attr]
          "jager/lambda_min": float(self.λ.min().item()),
          "jager/lambda_max": float(self.λ.max().item()),
          "jager/lambda_reg": self._last_lambda_reg,
          **({"loss/ρ": float(ρ.detach().mean().item())} if mixture else {})
      }, commit=False)
    
    return loss_lam_fix.mean()



class MultiHeadCELoss(nn.Module):

  def __init__(self, Y: torch.Tensor, level_offset: int = 1, label_smoothing: float = 0.0):
    super().__init__()
    self.Y = (Y - level_offset).long()
    self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="mean")

  def forward(self, y_pred: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    Y = self.Y[ids]
    return self.ce(y_pred.mT, Y)
