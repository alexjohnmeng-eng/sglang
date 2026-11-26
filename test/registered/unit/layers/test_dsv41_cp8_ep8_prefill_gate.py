"""CP8/EP8 opt-in never changes attention-DP or EP1 execution."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_v4_cp_prefill import cp8_ep8_prefill_enabled


class TestCp8Ep8PrefillGate(unittest.TestCase):
    def test_disabled_before_runtime_initialization(self):
        with envs.SGLANG_OPT_DSV41_CP8_EP8_PREFILL.override(False):
            with patch(
                "sglang.srt.models.deepseek_v4_cp_prefill.get_parallel",
                side_effect=AssertionError("runtime must not be read"),
            ):
                self.assertFalse(cp8_ep8_prefill_enabled())

    def test_decode_and_ep1_are_excluded_even_with_flag(self):
        with envs.SGLANG_OPT_DSV41_CP8_EP8_PREFILL.override(True):
            for cp, ep, dp, tp, expected in [
                (8, 8, 1, 1, True),
                (1, 8, 8, 1, False),  # DP8/EP8 decode
                (1, 4, 8, 1, False),  # DP8/EP4 decode
                (8, 1, 1, 1, False),  # paired W2
                (4, 8, 1, 2, False),
                (8, 8, 8, 1, False),
            ]:
                with self.subTest(cp=cp, ep=ep, dp=dp, tp=tp):
                    parallel = SimpleNamespace(
                        attn_cp_size=cp,
                        moe_ep_size=ep,
                        attn_dp_size=dp,
                        attn_tp_size=tp,
                    )
                    with patch(
                        "sglang.srt.models.deepseek_v4_cp_prefill.get_parallel",
                        return_value=parallel,
                    ):
                        self.assertEqual(cp8_ep8_prefill_enabled(), expected)
