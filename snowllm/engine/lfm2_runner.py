# SPDX-License-Identifier: Apache-2.0
from .qwen_runner import QwenRunner
from ..errors import SnowLLMError


class Lfm2Runner(QwenRunner):
    tunable_rope = False

    def _configure(self):
        if self.num_spec and not (self.draft_geo is not None and self.draft_geo.dspark):
            raise SnowLLMError('LFM speculative decoding requires a matching DSpark draft')
        super()._configure()

    def prefix_residue(self):
        from .prefix_cache import RecurrentResidue
        return RecurrentResidue([(m.state,m.ckpt) for _,m in self.conv_mods])

    def linear_advance(self, state_indices, num_accepted, B, T, cu_seqlens=None):
        from ..ops.lfm2 import conv_advance
        for _,attn in self.conv_mods:
            conv_advance(attn.retain,attn.state[0],state_indices[:B],num_accepted[:B])

    def _ctx(self, batch, plan_key=("main",)):
        ctx=super()._ctx(batch,plan_key)
        if batch.varlen_decode:
            ctx.plan_key=(*ctx.plan_key,"varlen")
        return ctx

    def _activation_cases(self, chunk, decode):
        yield from super()._activation_cases(chunk,decode)
        if self.draft_geo is not None:
            from .resources import ActivationProbe
            batch=self._probe_batch(decode,False)
            batch.varlen_decode=True
            batch.roll_forward=True
            yield ActivationProbe(self._ctx(batch).plan_key,decode,
                lambda:self._forward_eager(batch),
                lifetimes='uses' if self.plan_activations else 'references')
