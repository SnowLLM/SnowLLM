# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import unittest

import torch

from snowllm.engine import _check_dflash_target, draft_block
from snowllm.errors import SnowLLMError
from snowllm.models.geometry import DFlashGeometry
from snowllm.models.qwen3_5.dflash_draft import NeoXSource


class LfmDraftTests(unittest.TestCase):
    def geometry(self):
        return DFlashGeometry.from_config(dict(
            architectures=['Lfm2DSparkDraftModel'], hidden_size=2048,
            num_hidden_layers=5, num_attention_heads=32, num_key_value_heads=8,
            head_dim=64, intermediate_size=6144, vocab_size=128000,
            rms_norm_eps=1e-5, rope_theta=5000000, rope_is_neox_style=False,
            markov_rank=256, markov_head_type='vanilla', block_size=9,
            dflash_config=dict(mask_token_id=125017, target_layer_ids=[2,6,10,14,18],
                               num_target_layers=24)))

    def test_official_metadata_and_block(self):
        g=self.geometry()
        self.assertTrue(g.dspark)
        self.assertEqual((g.qkv_proj_n,g.ctx_kv_proj_n),(3072,5120))
        self.assertEqual(g.tap_layers,(2,6,10,14,18))
        self.assertEqual((g.num_target_layers,g.rope_theta),(24,5000000))
        self.assertFalse(g.rope_is_neox_style)
        self.assertEqual(draft_block(g,0),9)
        self.assertEqual(draft_block(g,9),9)
        with self.assertRaises(SnowLLMError):
            draft_block(g,15)

    def test_other_target_families_are_rejected(self):
        with self.assertRaisesRegex(SnowLLMError,'matching LFM2-MoE'):
            _check_dflash_target(SimpleNamespace(geo=None,config={'model_type':'qwen3_5_moe'}),
                                 self.geometry())

    def test_safetensors_permutation_preserves_rotary_attention(self):
        gen=torch.Generator(device='cpu').manual_seed(181)
        qweight=torch.randn(128,12,generator=gen)
        gamma=torch.randn(64,generator=gen)
        weights={'layers.0.self_attn.q_proj.weight':qweight,
                 'layers.0.self_attn.q_norm.weight':gamma}
        src=NeoXSource(SimpleNamespace(dense=weights.__getitem__),64)
        x=torch.randn(3,12,generator=gen)
        q=(x@qweight.T).view(3,2,64)
        q=q*torch.rsqrt(q.square().mean(-1,keepdim=True)+1e-5)*gamma
        angle=torch.randn(3,1,32,generator=gen)
        even,odd=q[...,::2],q[...,1::2]
        ref=torch.stack((even*angle.cos()-odd*angle.sin(),
                         odd*angle.cos()+even*angle.sin()),-1).flatten(-2)
        z=(x@src.dense('layers.0.self_attn.q_proj.weight').T).view(3,2,64)
        z=z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-5)*src.dense('layers.0.self_attn.q_norm.weight')
        a,b=z.chunk(2,-1)
        got=torch.stack((a*angle.cos()-b*angle.sin(),
                         b*angle.cos()+a*angle.sin()),-1).flatten(-2)
        torch.testing.assert_close(got,ref)


if __name__=='__main__':
    unittest.main()
