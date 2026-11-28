# SPDX-License-Identifier: Apache-2.0
from torch import nn

from ..ops.inference import inference_moe, inference_moe_norm, inference_moe_workspace_bytes


class ResidentMoE:
    def __init__(self, gate, down, gate_scale=None, down_scale=None):
        self.gate, self.down = gate, down
        self.gate_scale, self.down_scale = gate_scale, down_scale

    workspace_bytes = staticmethod(inference_moe_workspace_bytes)

    def forward(self, x, router, shared_gate, shared_down, out, workspace, token_ids=None, tid2eid=None):
        inference_moe(x, router, self.gate, self.down, shared_gate, shared_down, out, workspace,
                      self.gate_scale, self.down_scale, token_ids=token_ids, tid2eid=tid2eid)

    def forward_norm(self, x, residual, gamma, hidden_row, eps, router, shared_gate, shared_down,
                     out, workspace):
        inference_moe_norm(x, residual, gamma, eps, hidden_row, router, self.gate, self.down,
                           shared_gate, shared_down, out, workspace, self.gate_scale, self.down_scale)


class StandardMoE(nn.Module):
    def __init__(self, router, gate, down, shared_gate, shared_down, gate_scale=None, down_scale=None):
        super().__init__()
        self.router_w = router
        self.shared_gate_up, self.shared_down = shared_gate, shared_down
        self.backend = ResidentMoE(gate, down, gate_scale, down_scale)

    @property
    def cache(self):
        return None if isinstance(self.backend, ResidentMoE) else self.backend

    @property
    def gate_up(self):
        return self.backend.gate

    @property
    def down(self):
        return self.backend.down

    def bind_disk(self, layer):
        from ..ops.disk_moe import bind_standard
        return bind_standard(self, layer)
