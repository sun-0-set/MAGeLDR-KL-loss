import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from math import log, sqrt, ceil


def _cache(module: nn.Module, name: str, value: torch.Tensor) -> torch.Tensor:
  module.register_buffer(name, value, persistent=False)
  return getattr(module, name)


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
    debug: bool = False,
    log_to_wandb: bool = False,
    ):
      device = Y.device

      def _trunc_disc_norm_grid(K):
        φ_sq_inv_halved: tuple[float, ...] = (0.25541281188299525, 0.4479398673070137, 0.4887300185352654, 0.4978993024239318, 0.4996875664596105, 0.49996397161691486) # precomputed .5/φ(K)^2 for K=4..10; for K>10 approximate with .5
        k = torch.arange(K, device=device, dtype=torch.int16)
        return F.softmax(-(φ_sq_inv_halved[K-4] if K<=10 else .5) * (k[:,None] - k).square(), dim=1, dtype=torch.float64)


        
      super().__init__()
      
      # logging toggle 
      self.log_to_wandb = bool(log_to_wandb)

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
      # if reassignment and not conf_gating:
      #   raise ValueError("Competitor reassignment requires 'conf_gating' to be True.")
      if (reassignment or conf_gating) and not mixture:
        raise ValueError("Competitor reassignment and confidence gating require 'mixture' to be True.")
      
      self.joint = joint 
      self.mixture = mixture
      self.reassignment = reassignment
      self.conf_gating = conf_gating
      
      self.Y = _cache(self, 'Y', (Y - level_offset).long().to(device))
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
        self.flat_factors = _cache(
          self,
          'flat_factors',
          K ** torch.arange(self.H - 1, -1, -1, device=device, dtype=torch.long)
        )
        
      if mixture:
        self.ρ: torch.Tensor
        self.register_buffer(
          'ρ',
          torch.zeros((self.N) if joint else (self.N, self.H), dtype=torch.float64, device=device)
        )
        
        if joint:
          φ_sq_inv_halved: torch.Tensor = torch.tensor([0.25541281188299525, 0.4479398673070137, 0.4887300185352654, 0.4978993024239318, 0.4996875664596105, 0.49996397161691486], dtype=torch.float64, device=device) # precomputed .5/φ(K)^2 for K=4..10; for K>10 approximate with .5
          k = torch.arange(K, device=device, dtype=torch.long)
          self.D = D = _cache(self, 'D', (k[:,None] - k).square_())
          self.R_max = R_max = (K-1)**2
          z = (-D * (φ_sq_inv_halved[K-4] if K<=10 else .5)).exp_().sum(1)
          Z = z.clone()
          for _ in range(1, self.H):
            Z = Z.unsqueeze(-1) * z
          ker_shell = (-(φ_sq_inv_halved[K-4] if K<=10 else .5) * torch.arange(self.H * R_max + 1, device=device, dtype=torch.float64)).exp_()
          self.Kπ = _cache(self,'Kπ',KpowH * ker_shell.view(*[1] * self.H, self.H * R_max + 1) / Z.unsqueeze(-1),)
        else:
          π = _trunc_disc_norm_grid(K)
          self.Kπminus1 = _cache(self, 'Kπminus1', K*π - 1)
          

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
          δm = self.δm = _cache(self, 'δm', Mn - Mu_raw[1])
          δm_sq = self.δm_sq = _cache(self, 'δm_sq', self.δm.square())
          δm_cu2 = self.δm_cu2 = _cache(self, 'δm_cu2', δm_sq*self.δm*2)
          δm_qu = self.δm_qu = _cache(self, 'δm_qu', δm_sq.square())

          eq_m1 = self.eq_m1 = _cache(self, 'eq_m1', torch.isclose(δm, torch.zeros_like(δm)))

          ### Variance ###
          Vu = self.Vu = _cache(self, 'Vu', Mu_raw[2] - Mu_raw[1]**2)
          δv = self.δv = _cache(self, 'δv', Vn - Vu)
          # Depressing shift
          self.ρ0 = _cache(self, 'ρ0', (δv / δm_sq + 1)*.5)
          self.ρ0[eq_m1] = Vu/δv[eq_m1]
          ρ0 = self.ρ0
          r0 = self.r0 = _cache(self, 'r0', Vu/δm_sq + ρ0.square())

          ### Skewness ###
          Sn /= δm_cu2

          ### Kurtosis ###
          self.Ku = _cache(self, 'Ku', (Mu_raw[4] - 4*Mu_raw[3]*Mu_raw[1] + 6*Mu_raw[2]*Mu_raw[1].square() - 3*Mu_raw[1].square().square()).tile(K))
          self.δK = _cache(self, 'δK', Kn - self.Ku)
          self.Ku[~eq_m1] /= 3*δm_qu[~eq_m1]
          self.δK[~eq_m1] /= 3*δm_qu[~eq_m1]
          Ku = self.Ku
          δK = self.δK
          # Numerator coefficients (coef3 = 0, coef4 = 1)
          S = 8*_13rd*Sn + 2*r0
          self.coef_K = _cache(self, 'coef_K', torch.stack((
            -Ku - ρ0*(δK - (ρ0-1)*S + 5*((ρ0-2*_13rd)**3 - 1/27)),
            2*(ρ0-.5)*(S - 6*(ρ0-.5).square() - .5*_13rd) - δK,
            S - 8*(ρ0-.5).square() - 2*_13rd
          )))
        
        
      if conf_gating:
        # Mean ρ setup
        self.def_batch_size = def_batch_size
        self.steps_per_epoch = int(steps_per_epoch) if steps_per_epoch is not None else ceil(self.N_stats / self.def_batch_size)
        if joint:
          self.level_comb_counts = _cache(self, 'level_comb_counts', self._level_comb_counts(stats_Y, K).view(-1))
        else:
          self.level_counts = _cache(self, 'level_counts', self._level_counts(stats_Y, K).to(torch.long))
        self.cumul_ρ: torch.Tensor
        self.register_buffer(
          'cumul_ρ',
          torch.zeros_like(self.level_comb_counts if joint else self.level_counts, dtype=torch.float64, device=device)
        )
      
      
      # LDAM Margins according to Cao et al. "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss", 2019
      _base = (
        (self._level_comb_counts(stats_Y, K).add(1) if joint else (self.level_counts if conf_gating else self._level_counts(stats_Y, K)))
          .to(torch.float64)
          .clamp_min(1)
          .rsqrt().sqrt()
          .mul(C)
      )
      self._base_thresholds: torch.Tensor
      self.register_buffer('_base_thresholds', _base.to(device))


  def _level_comb_counts(self, Y, K):
    H = Y.shape[1]
    # Keep the flattened class order consistent with `flat_factors` and
    # the row-major layout produced by `.view(-1)` on the joint tensor.
    factors = K ** torch.arange(H - 1, -1, -1, dtype=torch.long, device=Y.device)
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


  def _joint_shell_mass_fft(self, p_shell_h: torch.Tensor) -> torch.Tensor:
    # Full H-fold convolution of the per-head shell masses along the squared-distance axis.
    out_len = p_shell_h.shape[1] * (p_shell_h.shape[-1] - 1) + 1
    spec = torch.fft.rfft(p_shell_h, n=out_len, dim=-1)
    return torch.fft.irfft(spec.prod(dim=1), n=out_len, dim=-1)


  def _joint_ρ_derivatives(self, p_shell: torch.Tensor, Kπ_pred_mode: torch.Tensor, ρ: torch.Tensor | None = None) -> torch.Tensor:
    if ρ is None:
      κ_mom1 = (p_shell * Kπ_pred_mode).sum(-1)
      κ_mom2 = (p_shell * Kπ_pred_mode.square()).sum(-1)
      κ_mom3 = (p_shell * Kπ_pred_mode.pow(3)).sum(-1)
      return torch.stack((
        κ_mom1 - 1,
        κ_mom2 - 2 * κ_mom1 + 1,
        κ_mom3 - 3 * κ_mom2 + 3 * κ_mom1 - 1,
      ))

    der = torch.empty((3, p_shell.shape[0]), dtype=p_shell.dtype, device=p_shell.device)
    ρ_is_zero = ρ.eq(0)
    if ρ_is_zero.any():
      der[:, ρ_is_zero] = self._joint_ρ_derivatives(p_shell[ρ_is_zero], Kπ_pred_mode[ρ_is_zero], None)
    if (~ρ_is_zero).any():
      log_S = torch.lerp(
        torch.ones_like(Kπ_pred_mode[~ρ_is_zero]),
        Kπ_pred_mode[~ρ_is_zero],
        ρ[~ρ_is_zero].unsqueeze(-1)
      ).log()
      ratio = -torch.expm1(-log_S) / ρ[~ρ_is_zero].unsqueeze(-1)
      orders = torch.arange(1, 4, device=p_shell.device, dtype=p_shell.dtype)[..., None, None]
      der[:, ~ρ_is_zero] = (p_shell[~ρ_is_zero].unsqueeze(0) * ratio.unsqueeze(0).pow(orders)).sum(-1)
    return der


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
    return -(p * log_q.masked_fill(p.eq(0), 0)).sum(dim=-1)

  def _all_gather_equal(self, x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
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
    λ0, K, H, α, logK, ε = self.λ0, self.K, self.H, self.α, self.logK, self.ε
    Y = self.Y[ids]
    B = y_pred.shape[0]
    λ = self.λ[ids]
    
    joint, mixture, reassignment, conf_gating = self.joint, self.mixture, self.reassignment, self.conf_gating 
    
    if joint:
      KpowH, HlogK = self.KpowH, self.HlogK
      if mixture:
        factors = self.flat_factors
        R_max = self.R_max
      
    y_pred = F.softplus(y_pred)
    if joint:
      y_pred = y_pred / y_pred.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12) * KpowH
    else:
      y_pred = y_pred / y_pred.sum(dim=2, keepdim=True).clamp_min(1e-12) * K
    

    with torch.no_grad():
      log_p_h = (y_pred / (λ[..., None, None] if joint else λ[..., None])).log_softmax(dim=2)
      p_h = log_p_h.exp() 

      if mixture:
        log_p_h_max, _mode = log_p_h.max(dim=2)
        _mode_unsq = _mode.unsqueeze(-1) # (B, H, 1)
        if joint:
          p_shell_h = torch.zeros((B, H, R_max+1), dtype=torch.float64, device=y_pred.device)
          p_shell_h.scatter_add_(2, self.D[_mode], p_h)
          out_len = p_shell_h.shape[1] * (p_shell_h.shape[-1] - 1) + 1
          spec = torch.fft.rfft(p_shell_h, n=out_len, dim=-1)
          p_shell = torch.fft.irfft(spec.prod(dim=1), n=out_len, dim=-1)
          Kπ_pred_mode = self.Kπ[*_mode.T]
          P_max = log_p_h_max.sum(-1).exp_()
          der_lo = self._joint_ρ_derivatives(p_shell, Kπ_pred_mode)
          ρ_hi_eval = torch.nextafter(torch.ones_like(P_max), torch.zeros_like(P_max))
          der_hi0 = self._joint_ρ_derivatives(p_shell, Kπ_pred_mode, ρ_hi_eval)[0]
          active = der_lo[0].gt(0) & der_hi0.lt(0)
          ρ = torch.where(
            der_lo[0].le(0),
            torch.zeros_like(P_max),
            torch.ones_like(P_max)
          )
          ρ_init = ((KpowH * P_max - 1) / Kπ_pred_mode[:, 0].sub(1)).clamp_(0,1)
          ρ = torch.where(active, torch.minimum(ρ_init, ρ_hi_eval), ρ)
          for _ in range(10):
            if not active.any(): break
            der = self._joint_ρ_derivatives(p_shell[active], Kπ_pred_mode[active], ρ[active])
            if der[0].abs().le(ε).all(): break
            den = der[1].square() - der[0]*der[2]
            sgn = den.sign()
            sgn = sgn.where(sgn.ne(0), 1)
            den.abs_().clamp_min_(ε).mul_(sgn)
            Δρ = der[0]*der[1]/den # Halley's step
            if Δρ.abs().le(ε).all(): break
            ρ_next = (ρ[active] + Δρ).clamp_min_(0)
            ρ[active] = torch.minimum(ρ_next, ρ_hi_eval[active])
        else:
          mean, _var, _skew, _kurt = self._cent_moments(p_h) 
          ρ = self._estimate_ρ_uvar(_mode, _kurt / _var.square().clamp_min(ε), p_h)
      
        if conf_gating:
          # ρ gated update 
          unique_ids_gate = self._unique_step_ids(ids) if update_state else ids.detach().to(self.λ.device, dtype=torch.long)
          Y_gate = self.Y[unique_ids_gate]
          old_ρ_gate = self.ρ[unique_ids_gate]
          level_counts_B_gate = self._level_comb_counts(Y_gate, K).view(-1) if joint else self._level_counts(Y_gate, K)
          cumul_ρ_B_gate = torch.zeros_like(self.cumul_ρ, dtype=self.cumul_ρ.dtype)
          if joint:
            Y_gate = (Y_gate * factors).sum(dim=1)
          cumul_ρ_B_gate.scatter_add_(-1, Y_gate.movedim(0, -1), old_ρ_gate.to(dtype=self.cumul_ρ.dtype).movedim(0, -1))
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
        if joint:
          log_S = Kπ_pred_mode.clone().sub_(1).mul_(ρ.unsqueeze(-1)).log1p_()
        else:
          Kπminus1_pred_mode: torch.Tensor = self.Kπminus1[_mode]
          log_S = (ρ.unsqueeze(-1)*Kπminus1_pred_mode).log1p() # logK is subtracted as per need
        _cond: torch.Tensor = _mode < K-_mode  # type: ignore[assignment]
        min_idx = torch.where(_cond, K-1, 0)
        if joint:
          log_S_min = log_S.gather(-1, (min_idx-_mode).square().sum(-1, keepdim=True)).squeeze_(-1)
          log_S_at_mode = log_S[:,0]
          kl_div = -Ent_p_h.sum(-1) + self._ce(p_shell, log_S) + HlogK
          u_bound = -(
            Ent_p_h.sum(-1) +
            log_S_min - HlogK +
            P_max *
            (log_S_at_mode - log_S_min)
          )
        else:
          log_S_min = log_S.take_along_dim(min_idx.unsqueeze_(-1), 2).squeeze_(-1)
          log_S_at_mode = log_S.take_along_dim(_mode_unsq, 2).squeeze_(-1)
          kl_div = -self._ce(p_h, log_p_h - log_S) + logK
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
          if self._debug and kl_div.gt(u_bound+ε).any():
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
      if self._debug and ((λt + ε < λ0*(1 - 1/α)).any() or λt.isnan().any()):
        torch.set_printoptions(precision=15, sci_mode=False)
        print("Debug Info:")
        print(f"kl_div: {kl_div}")
        print(f"u_bound: {u_bound}")
        print(f"λt: {λt + ε}")
        print(f"Expected λt range: [{λ0*(1 - 1/α)}, {λ0}]")
        raise ValueError("λt out of expected range or NaN. See debug info for details.")
      λ_reg_loss = -.5*α*u_bound * (λt - λ0).square() / λ0
      λt_unsq = λt.unsqueeze(-1)

      if reassignment:
        ρ_sq = ρ.square()

      # competitor assignment
      if mixture:
        if reassignment:
          if joint:
            Kπ_label: torch.Tensor = self.Kπ[*Y.T]
          else:
            Kπminus1_label: torch.Tensor = self.Kπminus1[Y]
        if joint:
          r_mode = self._outer_sum(self.D[_mode])
          if reassignment:
            r_label = self._outer_sum(self.D[Y])
            Kπ_label = Kπ_label.gather(1, r_label)
            Kπ_pred_mode = Kπ_pred_mode.gather(1, r_mode)
        if reassignment:
          if joint:
            ρ_unsq = ρ.unsqueeze(-1)
            log_υ = Kπ_pred_mode.sub(Kπ_label)
            log_υ.mul_(ρ_unsq).add_(Kπ_label).sub_(1).mul_(ρ_unsq).log1p_()
          else:
            log_υ = (ρ.unsqueeze(-1) * (Kπminus1_label + ρ.unsqueeze(-1) * (Kπminus1_pred_mode - Kπminus1_label))).log1p() # type: ignore[reportOperatorIssue]
        else:
          if joint:
            log_S = log_S.gather(1, r_mode)
          log_υ = log_S
      
      if joint:
        thresholds = self._base_thresholds.reshape(1, -1).expand(B, -1)
        flat_idx_label = (Y * self.flat_factors).sum(1, keepdim=True)
        if reassignment:
          flat_idx_mode = (_mode * self.flat_factors).sum(1, keepdim=True)
      else:
        thresholds = self._base_thresholds.unsqueeze(0).expand(B, *self._base_thresholds.shape)
        _1hot_label = F.one_hot(Y, K).to(thresholds.dtype)
        if reassignment:
          _1hot_pred_max = F.one_hot(_mode, K).to(thresholds.dtype)
      
      if not joint:
        payload = thresholds + λt_unsq * log_υ if mixture else thresholds
        mass = torch.lerp(_1hot_label, _1hot_pred_max, ρ_sq[..., None]) if reassignment else _1hot_label
        c = thresholds - payload * mass
    
    if joint:
      y_pred = self._outer_sum(y_pred, flat=True)
      y_label = y_pred.gather(1, flat_idx_label)
      if reassignment:
        ρ_sq_unsq: torch.Tensor = ρ_sq.unsqueeze(-1)
        y_pred_max = y_pred.gather(1, flat_idx_mode)
        ref = torch.lerp(y_label, y_pred_max, ρ_sq_unsq)
      else:
        ref = y_label
      diff_logits = y_pred - ref
      diff_logits.add_(thresholds)
      payload_label = thresholds.gather(1, flat_idx_label)
      if mixture:
        payload_label = payload_label + λt_unsq * log_υ.gather(1, flat_idx_label)
      if reassignment:
        payload_mode = thresholds.gather(1, flat_idx_mode)
        if mixture:
          payload_mode = payload_mode + λt_unsq * log_υ.gather(1, flat_idx_mode)
        diff_logits.scatter_add_(1, flat_idx_label, -(1 - ρ_sq_unsq) * payload_label)
        diff_logits.scatter_add_(1, flat_idx_mode, -ρ_sq_unsq * payload_mode) # type: ignore[arg-type]
      else:
        diff_logits.scatter_add_(1, flat_idx_label, -payload_label)
    else:
      y_label = y_pred.gather(2, Y.unsqueeze(-1))
      if reassignment:
        y_pred_max = y_pred.gather(2, _mode_unsq)
      ref = torch.lerp(y_label, y_pred_max, ρ_sq[..., None]) if reassignment else y_label
      diff_logits = y_pred - ref
      diff_logits.add_(c) 
    diff_logits_lam_fix = diff_logits.div_(λt_unsq)
    if mixture:
      diff_logits_lam_fix.add_(log_υ)
    logsumexp = diff_logits_lam_fix.logsumexp(-1)
    loss_lam_fix = λt * logsumexp
    
    if update_state:
      with torch.no_grad():
        unique_ids_update, λt_update, ρ_update = self._aggregate_state_updates(ids, λt, ρ if mixture else None)

        if conf_gating:
          Y_update = self.Y[unique_ids_update]
          old_ρ_update = self.ρ[unique_ids_update]
          assert ρ_update is not None, "ρ_update should not be None when conf_gating is enabled."
          delta = torch.zeros_like(self.cumul_ρ, dtype=self.cumul_ρ.dtype)
          delta.scatter_add_(-1, (Y_update*factors).sum(1) if joint else Y_update.T, (ρ_update - old_ρ_update).to(dtype=self.cumul_ρ.dtype).movedim(0, -1))
          self.cumul_ρ = self.cumul_ρ + delta

        self.λ[unique_ids_update] = λt_update
        if mixture and ρ_update is not None:
          self.ρ[unique_ids_update] = ρ_update
      
    self._last_lambda_reg = λ_reg_loss.detach().mean()

    return loss_lam_fix.mean()



class MultiHeadCELoss(nn.Module):

  def __init__(self, Y: torch.Tensor, level_offset: int = 1, label_smoothing: float = 0.0):
    super().__init__()
    self.Y = _cache(self, 'Y', (Y - level_offset).long())
    self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="mean")

  def forward(self, y_pred: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    Y = self.Y[ids]
    return self.ce(y_pred.mT, Y)
