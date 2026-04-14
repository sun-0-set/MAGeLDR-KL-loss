import math
import unittest

import torch
import torch.nn.functional as F

from loss import JAGeRLoss


class JAGeRLossRegressionTests(unittest.TestCase):

  def test_joint_level_comb_counts_follow_flat_joint_order(self):
    labels = torch.tensor([
      [1, 5, 2],
      [1, 5, 2],
      [5, 1, 2],
    ], dtype=torch.long)
    loss_fn = JAGeRLoss(
      Y=labels,
      K=5,
      def_batch_size=1,
      joint=True,
      mixture=False,
      conf_gating=False,
      reassignment=False,
      level_offset=1,
      λ0=1.0,
      λmin=0.5,
      C=1e-1,
    )

    counts = loss_fn._level_comb_counts(loss_fn.Y, loss_fn.K).view(-1)
    expected = torch.zeros(loss_fn.KpowH, dtype=torch.long)

    repeated = torch.tensor([0, 4, 1], dtype=torch.long)
    single = torch.tensor([4, 0, 1], dtype=torch.long)
    repeated_idx = int((repeated * loss_fn.flat_factors).sum().item())
    single_idx = int((single * loss_fn.flat_factors).sum().item())
    expected[repeated_idx] = 2
    expected[single_idx] = 1

    torch.testing.assert_close(counts, expected)

  def _joint_reassignment_loss(self, labels: list[int]) -> float:
    loss_fn = JAGeRLoss(
      Y=torch.tensor([labels]),
      K=5,
      def_batch_size=1,
      joint=True,
      mixture=True,
      conf_gating=True,
      reassignment=True,
      level_offset=1,
      λ0=1.0,
      λmin=0.5,
      C=1e-1,
    )
    logits = torch.tensor([[
      [0.1, 0.1, 0.1, 0.1, 50.0],
      [0.1, 0.1, 0.1, 0.1, 50.0],
      [0.1, 0.1, 0.1, 0.1, 50.0],
    ]], dtype=torch.float64)

    with torch.no_grad():
      return float(loss_fn(logits, torch.tensor([0]), update_state=False).item())

  def test_joint_reassignment_keeps_wrong_confident_loss_positive(self):
    loss = self._joint_reassignment_loss([1, 1, 1])

    self.assertTrue(math.isfinite(loss))
    self.assertGreater(loss, 1e-6)

  def test_joint_reassignment_penalizes_wrong_mode_more_than_correct_mode(self):
    wrong_loss = self._joint_reassignment_loss([1, 1, 1])
    correct_loss = self._joint_reassignment_loss([5, 5, 5])

    self.assertTrue(math.isfinite(wrong_loss))
    self.assertTrue(math.isfinite(correct_loss))
    self.assertGreater(wrong_loss, correct_loss + 1.0)

  def test_reassignment_requires_conf_gating(self):
    with self.assertRaisesRegex(ValueError, "conf_gating"):
      JAGeRLoss(
        Y=torch.tensor([[1, 1, 1]]),
        K=5,
        def_batch_size=1,
        joint=True,
        mixture=True,
        conf_gating=False,
        reassignment=True,
        level_offset=1,
        λ0=1.0,
        λmin=0.5,
        C=1e-1,
      )

  def test_joint_shell_fft_matches_iterated_full_convolution(self):
    loss_fn = JAGeRLoss(
      Y=torch.tensor([[1, 1, 1, 1]]),
      K=6,
      def_batch_size=1,
      joint=True,
      mixture=True,
      conf_gating=True,
      reassignment=True,
      level_offset=1,
      λ0=1.0,
      λmin=0.5,
      C=1e-1,
    )
    torch.manual_seed(0)
    p_shell_h = torch.rand((3, 4, loss_fn.R_max + 1), dtype=torch.float64)
    p_shell_h /= p_shell_h.sum(dim=-1, keepdim=True)

    expected = p_shell_h[:, 0, :]
    for h in range(1, p_shell_h.shape[1]):
      expected_old = F.pad(expected, (loss_fn.R_max, loss_fn.R_max))
      expected_old = expected_old.unfold(-1, loss_fn.R_max + 1, 1)
      expected = (expected_old * p_shell_h[:, h, :].flip(-1).unsqueeze(1)).sum(-1)

    actual = loss_fn._joint_shell_mass_fft(p_shell_h)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
  unittest.main()
