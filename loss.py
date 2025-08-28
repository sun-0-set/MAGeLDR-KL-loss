import torch
import torch.nn as nn
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False
    
from math import log, pi, sqrt

class MAGe_LDRLoss(nn.Module):


  def __init__ (
    self, 
    Y: torch.Tensor, # true label column
    K: int, # number of classes
    distribution: str = 'mixture',
    level_offset: int = 0, 
    λ0: float = 1, # initial value for λ
    α: float = 2, # initial value for α
    C: float = 1e-1, # initial value for C
    softplus: bool = False, # whether to use softplus activation
    debug: bool = False,
    log_to_wandb: bool = False
    ):
      
      
      
      def _trunc_disc_norm_grid(K):
        _kk = torch.arange(K, device=dev).tile((K,1))
        return (
          (_kk - _kk.T).tile(K,1,1).permute(2,1,0) *\
          (
            (_kk + _kk.T).tile(K,1,1).permute(2,1,0)*.5 -\
            _kk.tile(K,1,1)
          )
        ).exp_().sum(1).reciprocal_().T
          
          
      def _level_comb_counts(Y, κ, K):
        base = torch.full((κ,), K, dtype=torch.long, device=dev)
        exp = torch.arange(κ, dtype=torch.long, device=dev)
        factors = torch.pow(base, exp)                  # [κ] integer powers
        flat_idx = (Y * factors).sum(dim=1)
        counts_flat = torch.bincount(flat_idx, minlength=K**κ)
        return counts_flat.view([K]*κ)

        
      super().__init__()
      
      # logging toggle (safe default = off)
      self.log_to_wandb = bool(log_to_wandb)

      import os
      self._debug = debug or os.environ.get("MAGE_DEBUG", "0") == "1"
      self._eps = 1e-12
      
      
      if distribution not in ('mixture', 'uniform'):
        raise ValueError(f"Invalid distribution type: {distribution}. Choose 'mixture' or 'uniform'.")
      self.distribution = distribution
      self.Y = (Y - level_offset).long().to(dev)
      N, self.κ = self.Y.shape
      self.K = K
      self.logK = log(K)
      self.κlogK = self.κ * self.logK
      self.φ = _trunc_disc_norm_grid(K)
      # self.Kφ_1 = self.K * φ - 1
      self.register_buffer('ρ', torch.zeros((N, self.κ), dtype=torch.float64, device=dev))
      # per-head normalized log priors ψ_h ≡ uniform at start
      self.register_buffer('log_ψ_h',
                           torch.full((N, self.κ, self.K), fill_value=-self.logK,
                                      dtype=torch.float64, device=dev))
      self.λ0 = λ0
      self.register_buffer('λ', torch.full((N,), λ0, dtype=torch.float64, device=dev))
      self.α = α
      self.softplus = softplus
      
      # Margins according to Cao et al. "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss", 2019
      _base = (
        _level_comb_counts(self.Y, self.κ, K)
          .add(1)
          .to(torch.float64)
          .pow(-.25)
          .mul(C)
      )
      self.thresholds = (
        _base
          .unsqueeze(0)
          .expand(N, *_base.shape)
          .contiguous()
          .to(dev)
      )
      self.thresholds[torch.arange(N, device=dev), *self.Y.unbind(1)] = 0  # Set thresholds for true labels to zero

      #--- Setup for ρ estimation during loss instantiation

      # Numerical constants
      self._sqrt3 = sqrt(3)
      self._13rd = 1/3

      _Mn, _Vn, _Sn, _Kn = self._cent_moments(φ)
      
      _Mu_raw = torch.tensor([
        1,
        (K-1)/2,
        (K-1)*(2*K-1)/6,
        (K-1)**2*(K)/4,
        (K-1)*(2*K-1)*(3*(K-1)**2+3*K-4)/30
      ], dtype=torch.float64, device=dev)

      ### Mean ###
      self._Mu1 = _Mu_raw[1]
      self._δm = _Mn - _Mu_raw[1]
      self._δm_sq = self._δm.square()
      self._δm_cu2 = self._δm_sq*self._δm*2
      self._δm_qu = self._δm_sq.square()

      self._eq_m1 = torch.isclose(self._δm, torch.zeros_like(self._δm))

      ### Variance ###
      self._Vu = _Mu_raw[2] - _Mu_raw[1]**2
      self._δv = _Vn - self._Vu
      # Depressing shift
      self._ρ0 = (self._δv / self._δm_sq + 1)*.5
      self._ρ0[self._eq_m1] = self._Vu/self._δv[self._eq_m1]
      self._r0 = self._Vu/self._δm_sq + self._ρ0**2

      ### Skewness ###
      _Sn /= self._δm_cu2
      # Su == 0 for all ν
      # Numerator coefficients (coef2 = 0, coef3 = 1
      self._coef_s0 = self._ρ0*(_Sn - (self._ρ0-1)*(2*self._ρ0-1))
      _coef_s1 = _Sn - 3*(self._ρ0-.5)**2 - .25
      self._coef_s1_sqrt13 = (-_coef_s1/3).sqrt()
      # Extrema
      a = 3*self._r0+2*_coef_s1
      self._kurt_ext_sgn = a.sign()
      b = 3*self._coef_s0*.5
      self._extrema_γ1 = (-b+(b**2 - _coef_s1*self._r0*a).sqrt())/a

      ### Kurtosis ###
      self._Ku = (_Mu_raw[4] - 4*_Mu_raw[3]*_Mu_raw[1] + 6*_Mu_raw[2]*_Mu_raw[1]**2 - 3*_Mu_raw[1]**4).tile(K)
      self._δK = _Kn - self._Ku
      self._Ku[~self._eq_m1] /= 3*self._δm_qu[~self._eq_m1]
      self._δK[~self._eq_m1] /= 3*self._δm_qu[~self._eq_m1]
      # Numerator coefficients (coef3 = 0, coef4 = 1)
      S = 8*self._13rd*_Sn + 2*self._r0
      self._coef_K = torch.stack((
        -self._Ku - self._ρ0*(self._δK - (self._ρ0-1)*S + 5*((self._ρ0-2*self._13rd)**3 - 1/27)),
        2*(self._ρ0-.5)*(S - 6*(self._ρ0-.5)**2 - .5*self._13rd) - self._δK,
        S - 8*(self._ρ0-.5)**2 - 2*self._13rd
      ))
      # Extrema
      E0, E1, E2 = torch.stack((
        self._coef_K[1]*self._r0,
        2*(2*self._coef_K[0] + self._r0*self._coef_K[2]),
        self._coef_K[1]
      )) / (2*(self._coef_K[2] + 2*self._r0))
      p = E1 - E2*E2 * 3
      q = 2 * E2*E2*E2 - E2 * E1 + E0
      R = (-p*self._13rd).sqrt()
      θ = (-q*.5 / (R*R*R)).acos() + ~a.signbit()*2*pi
      self._extrema_γ2 = (
        (
          (
            torch.tensor(([2], [-2]), dtype=torch.float64, device=dev)*pi + θ)*self._13rd
          ).cos()*2*R - E2
        ).T
      self._extrema_γ2[self._eq_m1] = torch.stack((
        torch.full_like(self._Ku[self._eq_m1], -torch.inf),
        -2 * self._Ku[self._eq_m1] / self._δK[self._eq_m1]
      ), dim=1)


  def _outer_sum_heads(self, x: torch.Tensor, flat: bool = True) -> torch.Tensor:
    """
    x: (B, κ, K)  →  broadcast-sum across heads.
    - If x are per-head *log-probs* (log ψ_h or log q_h), result is log ψ_joint or log q_joint.
    - If x are per-head *scores* (e.g., per-head logits already scaled/normalized), result is the joint score grid.
    Returns (B, K^κ) when flat=True, else (B, K, .., K).
    """
    B, κ, K = x.shape
    joint = x[:, 0, :]                               # (B, K)
    for h in range(1, κ):
      joint = (joint.unsqueeze(-1) + x[:, h, :].unsqueeze(1)).reshape(B, -1)
    if flat:
      return joint
    # reshape back to (B, K, .., K)
    return joint.view(B, *([K] * κ))


  def _cent_moments(self, probs):
    # Compute raw moments
    k = torch.arange(self.K, dtype=probs.dtype, device=dev)
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
    κ = pred_log_prob.dim() - 1  # number of heads
    marginals_log = [
      pred_log_prob.logsumexp(dim=tuple(d for d in range(1, κ+1) if d != h+1))  # (B, K)
      for h in range(κ)
    ]
    return torch.stack(marginals_log, dim=1).exp()  # (B, κ, K)


  def _estimate_ρ(self, ν, μ, σ2, γ1, γ2):

    ν = ν.long()

    # Gather per-ν constants (broadcasted to (B, κ))
    msk  = self._eq_m1[ν]
    δm       = self._δm[ν]
    δm_sq    = self._δm_sq[ν]
    δm_cu2   = self._δm_cu2[ν]
    δv = self._δv[ν]  # (B, κ)
    Ku = self._Ku[ν]
    δK = self._δK[ν]
    ρ0 = self._ρ0[ν]
    r0 = self._r0[ν]
    coef_s0 = self._coef_s0[ν]
    coef_s1_sqrt13 = self._coef_s1_sqrt13[ν]
    ext_γ1 = self._extrema_γ1[ν]
    ext_γ2 = self._extrema_γ2[ν]  # (B, κ, 2) TODO

    ρ = torch.empty_like(ν, dtype=torch.float64)  # Initialize ρ tensor

    # Case 1: equal-mean branch (Variance -> Kurtosis)
    b = δK[msk] / (2 * δv[msk].square() * γ2[msk].clamp_min(1e-12))
    D = (b.square() + (Ku[msk] - δK[msk]*ρ0[msk]) / (δv[msk].square() * γ2[msk].clamp_min(1e-12))).clamp_min(0).sqrt()
    ρ[msk] = (b + torch.where(b < (σ2[msk]/δv[msk]), D, -D) - ρ0[msk]).clamp(0, 1)

    # Case 2: general branch (Mean -> Variance -> Skewness -> Kurtosis)
    # Skewness-based ρ_γ1
    k13 = ((γ1 * σ2.pow(1.5) / δm_cu2 - coef_s0) / (coef_s1_sqrt13**3) * 0.5).acos() * self._13rd
    m = k13.cos()
    left_term = (ρ0*ρ0 - (σ2 - self._Vu) / δm_sq).clamp_min(0).sqrt()
    sign = torch.where(((μ - self._Mu1) / δm) < ρ0, -torch.ones_like(ρ0), torch.ones_like(ρ0))
    ρ_γ1 = torch.where(
      left_term * sign < ext_γ1,
      -coef_s1_sqrt13 * (m - self._sqrt3 * k13.sin()),
      2 * coef_s1_sqrt13 * m
    )

    # Kurtosis quartic solve
    coef_K0 = self._coef_K[0][ν]
    coef_K1 = self._coef_K[1][ν]
    coef_K2 = self._coef_K[2][ν]
    den = 1 + γ2.clamp_min(1e-12) * self._13rd
    p = -(coef_K2 - 2 * γ2.clamp_min(1e-12) * r0 * self._13rd) * 0.5 / den
    q = (coef_K1 * 0.5) / den
    r = -(coef_K0 + γ2.clamp_min(1e-12) * r0.square() * self._13rd) / den

    p13 = p * self._13rd
    p2 = p13 * p13
    f = r * self._13rd - p2
    g = p13 * (2 * p2 - r) + p * r - 0.5 * q.square()
    h = 0.25 * g.square() + f * f * f

    mask_h = h <= 0
    # h <= 0 branch
    l = f.abs().sqrt()
    acos_arg = torch.clamp(0.5 * g / (f * l), -1.0, 1.0)
    m_res = (acos_arg.acos() * self._13rd).cos()
    cr1 = 2 * l * m_res - p13
    # h > 0 branch
    sqrt_h = h.clamp_min(0).sqrt()
    cr2 = self._cbrt(-0.5 * g + sqrt_h) + self._cbrt(-0.5 * g - sqrt_h) - p13
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
    ])

    # multi-head: shape (4, B, κ)
    diff = torch.abs(kurt_quart_roots - ρ_γ1.unsqueeze(0))
    idx = diff.argmin(dim=0)
    closest_roots = torch.gather(kurt_quart_roots, 0, idx.unsqueeze(0)).squeeze(0)


    # assign final ρ for kurtosis branch
    ρ[~msk] = (closest_roots[~msk] + ρ0[~msk]).clamp(0, 1)

    return ρ
        
        
  def forward_mixture(self, y_pred, ids, update_state: bool = True):
    # y_pred: (B, κ==n_heads, n_levels)

    λ0, K, α = self.λ0, self.K, self.α
    Y = self.Y[ids]  # (B, κ)
    B = y_pred.shape[0]
    λ = self.λ[ids] # (B)
    

    # logits tensor normalised per head
    if self.softplus:
      y_pred = F.softplus(y_pred)
    y_pred = F.normalize(y_pred, dim=2, p=1)*K

    with torch.no_grad():
      head_scores = y_pred.detach() / λ.view(-1,1,1).clamp_min(1e-12)
      log_ψ_h = self.log_ψ_h[ids] 
      log_q_h = (head_scores + log_ψ_h).log_softmax(dim=2)
      q_h = log_q_h.exp() 

      _mode = q_h.argmax(dim=2) 
      _mean, _var, _skew, _kurt = self._cent_moments(q_h) 
      ρ = self._estimate_ρ(_mode, _mean, _var, _skew / _var.pow(1.5).clamp_min(self._eps), _kurt / _var.square().clamp_min(self._eps)) 
      φ = self.φ[_mode]
      log_ψ_h_new = (ρ.unsqueeze(-1) * φ + (1 - ρ.unsqueeze(-1))/K).log()

      kl_reg = (q_h * (log_q_h - log_ψ_h_new)).sum(dim=(1,2)) 
      min_idx = torch.where(
        _mode < K-_mode,
        torch.full_like(_mode, K-1),
        torch.zeros_like(_mode)
      ).long()  
      min_per_head = log_ψ_h_new.gather(2, min_idx.unsqueeze(-1)).squeeze(-1) 
      mixture_min  = min_per_head.sum(dim=1)
      l_bound = -mixture_min  
      λt = λ0 * (1 - kl_reg / (α * l_bound))
      λ_reg_loss = -.5*α*l_bound / λ0 * (self.λ[ids] - λ0).square()
      joint_log_ψ = self._outer_sum_heads(log_ψ_h_new, flat=True) 
      

    # weights update
    y_pred = self._outer_sum_heads(y_pred, flat=False)
    y_true = y_pred[torch.arange(B, device=y_pred.device), *Y.unbind(1)].unsqueeze(1)
    diff_logits = y_pred.view(B, -1) - y_true + self.thresholds[ids].view(B, -1)
    diff_logits_lam_fix = diff_logits / λt.unsqueeze(1).clamp_min(1e-12) + joint_log_ψ
    logsumexp_weighted = torch.logsumexp(diff_logits_lam_fix, dim=1)
    loss_lam_fix = λt * logsumexp_weighted + λ_reg_loss
    
    if update_state:
      self.ρ[ids] = ρ
      self.log_ψ_h[ids] = log_ψ_h_new
      self.λ[ids] = λt.clamp_min(0.0)
      
    if self.log_to_wandb and wandb.run is not None:
      wandb.log({
          "loss/ρ": float(ρ.detach().mean()),
          "loss/kl_reg": float(kl_reg.detach().mean()),
          "mage/lambda_min": float(self.λ.min().detach().cpu()),
          "mage/lambda_max": float(self.λ.max().detach().cpu())
      }, commit=False)
    
    return loss_lam_fix.mean()
  
  
  # @torch.no_grad() TODO
  # def prior_for_inference(self, y_pred: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
  #   """
  #   Compute per-head log prior log(ψ̂_h) for MAP decoding, without mutating buffers.
  #   Returns shape: (B, κ, K). For 'uniform' it is just log(1/K).
  #   """
  #   B, κ, K = y_pred.shape
  #   dev = y_pred.device
  #   dtype = y_pred.dtype

  #   if self.distribution == 'uniform':
  #     # Adding a constant per class won't change argmax; still return it for completeness.
  #     return torch.full((B, κ, K), -self.logK, device=dev, dtype=dtype)

  #   # Use current λ buffer (val/test usually λ0). Read-only.
  #   lam = self.λ[ids].view(-1, 1, 1).clamp_min(self._eps).to(device=dev, dtype=dtype)  # (B,1,1)
  #   head_scores = y_pred.detach() / lam                                               # (B,κ,K)
  #   log_ψ_h_old = self.log_ψ_h[ids].to(dev)                                           # (B,κ,K)

  #   # Posterior per head under current prior (same as forward, read-only)
  #   log_q_h = (head_scores + log_ψ_h_old).log_softmax(dim=2)                          # (B,κ,K)
  #   q_h = log_q_h.exp()

  #   # Moments for ρ estimation
  #   # Build 0..K-1 on the fly to avoid relying on a class buffer.
  #   idx = torch.arange(K, device=dev, dtype=y_pred.dtype)                                                             # (K,)
  #   _mode = q_h.argmax(dim=2)                                                         # (B,κ)
  #   _mean = (q_h * idx).sum(dim=2)                                                    # (B,κ)
  #   _var  = (q_h * (idx - _mean.unsqueeze(-1)).square()).sum(dim=2).clamp_min(self._eps)
  #   _skew = (q_h * (idx - _mean.unsqueeze(-1)).pow(3)).sum(dim=2)
  #   _kurt = (q_h * (idx - _mean.unsqueeze(-1)).pow(4)).sum(dim=2)
  #   s_skew = _skew / _var.pow(1.5)
  #   s_kurt = _kurt / _var.square()

  #   ρ = self._estimate_ρ(_mode, _mean, _var, s_skew, s_kurt)                          # (B,κ)
  #   φ_rows = φ[_mode]                                                            # (B,κ,K)
  #   ψ_h = ρ.unsqueeze(-1) * φ_rows + (1. - ρ).unsqueeze(-1) / K                       # (B,κ,K)
  #   log_ψ_h_new = ψ_h.clamp_min(self._eps).log().to(dtype)
  #   return log_ψ_h_new


  def forward_uniform(self, y_pred, ids, update_state: bool = True):
    # y_pred: (B, κ==n_heads, n_levels)
    λ0, K, logK, κlogK, α = self.λ0, self.K, self.logK, self.κlogK, self.α
    Y = self.Y[ids] 
    B = y_pred.shape[0]
    λ = self.λ[ids] 

    # logits tensor normalised per head
    if self.softplus:
      y_pred = F.softplus(y_pred)
    y_pred = F.normalize(y_pred, dim=2, p=1)*K

    with torch.no_grad():
      head_scores = y_pred.detach() / λ.view(-1,1,1).clamp_min(1e-12) 
      log_q_h = head_scores.log_softmax(dim=2)
      q_h = log_q_h.exp()
      kl_reg = (q_h * (log_q_h + logK)).sum(dim=(1,2))
      λt = λ0*(1 - kl_reg / (α*κlogK))

      λ_reg_loss = -.5*α*κlogK / λ0 * (λt - λ0).square()

    # weights update
    y_pred = self._outer_sum_heads(y_pred, flat=False)
    y_true = y_pred[torch.arange(B, device=y_pred.device), *Y.unbind(1)].unsqueeze(1)
    diff_logits = y_pred.view(B, -1) - y_true + self.thresholds[ids].view(B, -1)
    diff_logits_lam_fix = diff_logits / λt.unsqueeze(1).clamp_min(1e-12)
    logsumexp_weighted = torch.logsumexp(diff_logits_lam_fix, dim=1)
    loss_lam_fix = λt * logsumexp_weighted + λ_reg_loss
    
    if update_state:
      self.λ[ids] = λt.clamp_min(.0)
    return loss_lam_fix.mean()


  def forward(self, y_pred, ids, update_state: bool = True):
    return self.forward_mixture(y_pred, ids, update_state) if self.distribution == 'mixture' else self.forward_uniform(y_pred, ids, update_state)


class MultiHeadUnivariateALDR_KL(nn.Module):

  def __init__ (
    self, 
    Y: torch.Tensor, # true label column
    K: int, # number of classes
    level_offset: int = 0, 
    λ0: float = 1, # initial value for λ
    α: float = 2, # initial value for α
    C: float = 1e-1, # initial value for C
    softplus: bool = False, # whether to use softplus activation
    debug: bool = False
    ):

      def _level_counts(Y, κ, K):
        counts = torch.zeros((κ, K), dtype=torch.long, device=Y.device)
        for h in range(κ):
          counts[h] = torch.bincount(Y[:, h], minlength=K)
        return counts  
      
      super().__init__()
      self.Y = (Y - level_offset).long().to(dev)
      N, self.κ = self.Y.shape
      self.K = K
      self.logK = log(K)
      self.κlogK = self.κ * self.logK
      
      import os
      self._debug = debug or os.environ.get("MAGE_DEBUG", "0") == "1"
      
      self.λ0 = λ0
      self.register_buffer('λ', torch.full((N,self.κ), λ0, dtype=torch.float64, device=dev))
      self.α = α
      self.softplus = softplus
      
      # Margins according to Cao et al. "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss", 2019
      _base = (
        _level_counts(self.Y, self.κ, K)
          .to(torch.float64)
          .pow(-.25)
          .mul(C)
      )
      thresholds = (
        _base
          .unsqueeze(0)
          .expand(N, *_base.shape)
          .contiguous()
          .to(dev)
      )
      # Set thresholds for true labels to zero
      mask = F.one_hot(self.Y, K).bool()  # (N, κ, K)
      thresholds[mask] = 0

      self.register_buffer('thresholds', thresholds)

  def forward(self, y_pred, ids, update_state: bool = True):
    # y_pred: (B, κ==n_heads, n_levels)
    λ0, logK, α, K = self.λ0, self.logK, self.α, self.K
    Y = self.Y[ids]  # (B, κ)
    B = y_pred.shape[0]
    λ = self.λ[ids] # (B, κ)

    # logits tensor normalised per head
    if self.softplus:
      y_pred = F.softplus(y_pred)
    y_pred = F.normalize(y_pred, dim=2, p=1)*K

    with torch.no_grad():
      head_scores = y_pred.detach() / λ.unsqueeze(-1).clamp_min(1e-12)        # (B, κ, K)
      log_q_h = head_scores.log_softmax(dim=2)                # (B, κ, K)
      q_h = log_q_h.exp()                                             # (B, κ, K)
      kl_reg = (q_h * (log_q_h + logK)).sum(dim=(2))            # (B,κ)
      λt = λ0*(1 - kl_reg / (α*logK))  # (B,κ)

      λ_reg_loss = -(.5*α*logK / λ0 * (λt - λ0).square())

    # weights update
    y_true = y_pred.gather(2, Y.unsqueeze(-1))  # (B, κ, 1)
    diff_logits = y_pred - y_true + self.thresholds[ids]
    diff_logits_lam_fix = diff_logits / λt.unsqueeze(-1).clamp_min(1e-12)
    logsumexp_weighted = diff_logits_lam_fix.logsumexp(dim=-1)
    loss_lam_fix = λt * logsumexp_weighted + λ_reg_loss
    
    if update_state:
      self.λ[ids] = λt.clamp_min(.0)
    return loss_lam_fix.mean()


class MultiHeadCELoss(nn.Module):
    """
    Average CE across κ heads.
    - y_pred: (B, κ, K) logits
    - Y: global targets tensor (N, κ) of int labels
    """
    def __init__(self, Y: torch.Tensor, K: int, label_smoothing: float = 0.0):
      super().__init__()
      self.register_buffer("Y", Y.to(torch.long))
      self.K = K
      self.κ = Y.shape[1]
      self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="mean")

    def forward(self, y_pred: torch.Tensor, ids: torch.Tensor, update_state: bool = False) -> torch.Tensor:
      # y_pred: (B, κ, K)
      B, κ, K = y_pred.shape
      assert κ == self.κ and K == self.K, "shape mismatch for CE loss"
      y = self.Y[ids]  # (B, κ)
      # compute per-head CE and average
      losses = []
      for h in range(κ):
          losses.append(self.ce(y_pred[:, h, :], y[:, h] - 1))
      return torch.stack(losses, dim=0).mean()