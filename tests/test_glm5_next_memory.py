# SPDX-License-Identifier: Apache-2.0
"""GLM runtime budgets use the common dry-load and placement pipeline."""
from types import SimpleNamespace
from unittest.mock import patch
import os
from pathlib import Path
import sys

import _harness
import torch
from snowllm import ops
from snowllm._capi import GEO_GLM53_FLASH, SnowLLMError
from snowllm.checkpoint import loader
from snowllm.engine import glm5_next_runner as runners
from snowllm.engine import runner as common
from snowllm.models import placement
from snowllm.models.glm5_next.layers import Glm5NextGeometry, LinearAttention


def main():
    ck = _harness.Checks()
    ops.select_geometry(GEO_GLM53_FLASH)
    geo = SimpleNamespace(layer_types=('linear_attention',)*34+('deepseek_sparse_attention',)*11,
                          index_ratio=4,index_topk=2048,hidden=4096,
                          state_bytes=lambda n: 0, block_bytes=lambda n: 0)
    candidate_runner=object.__new__(runners.Glm5NextRunner)
    candidate_runner._want_chunk='auto';candidate_runner.ctx_cap=4096
    ck('GLM inherits common chunk selection',
       runners.Glm5NextRunner.candidates is common.BaseRunner.candidates)
    ck('auto respects context cap and removes duplicates',
       candidate_runner.candidates()==[4096,2048,1024,512,256,128])
    candidate_runner._want_chunk=2048;candidate_runner.ctx_cap=8192
    ck('explicit chunk is respected',candidate_runner.candidates()==[2048])
    candidate_runner._want_chunk=0
    try:
        candidate_runner.candidates()
    except SnowLLMError:
        ck('nonpositive chunk rejected',True)
    else:
        ck('nonpositive chunk rejected',False)
    gib=1<<30
    sizes={'experts':24*gib,'dense':72*gib}
    by={'experts':{i:2*gib for i in range(12)}}
    real_geo=Glm5NextGeometry(4096,154880,45,geo.layer_types,64,64,128,512,128,32,4,2048,4,1e-6,4)
    class BudgetModel:
        def __init__(self, workspace):
            self.geo, self.layers, self.mtp = real_geo, [], None
            self.workspace = workspace
            self.width = 4096
            self.lm_head = SimpleNamespace(scratch=None)
            self.embed = torch.empty(1,4096,dtype=torch.bfloat16,device="meta")
            self.weights = ops.empty_bytes(96*gib)

        def modules(self):
            return iter(())

        def __call__(self, ctx):
            rows = ctx.batch.input_ids.numel()
            ctx.workspace(self.workspace(rows))
            return ctx.arena.new(rows,self.width)

    def build():
        return BudgetModel(lambda rows: (11 if rows==8192 else 5)*gib)
    kv=dict(ctx=8192,slots=1,util=.95,chunk='auto')
    with patch('snowllm.engine.gpu_mem_info',return_value=(96*gib,96*gib)), \
         patch.object(placement,'auto_budget',return_value=15*gib):
        need=loader.dry_need(build,{},kv,15*gib)
        ck('common dry-load tries narrower chunk to fit host budget',9*gib<need<11*gib)
        dmap=loader._device_map('auto',{},kv,build,sizes,by)
        ck('common planner offloads only fitting expert slabs',
           dmap.host==frozenset({'experts'}) and dmap.host_bytes==10*gib and not dmap.short)
        wide_need=loader.dry_need(build,{},dict(kv,chunk=8192),15*gib)
        ck('explicit wide chunk is not silently reduced',wide_need>15*gib)
        with patch.object(placement,'auto_budget',return_value=23*gib):
            reserved=loader._device_map('auto',{},dict(kv,host_taken=8*gib),build,sizes,by)
            ck('shared placement subtracts drafter host memory',reserved.host_bytes==10*gib)
        mixed={'experts':{0:7*gib,1:7*gib,2:4*gib,3:4*gib,4:4*gib}}
        combination=placement.parse('auto',sizes,12*gib,12*gib,mixed)
        ck('common planner fits mixed slabs when largest-first cannot',
           combination.host_bytes==12*gib and len(combination.layers['experts'])==3 and not combination.short)
        ck('explicit device-only placement does not become auto',
           not loader._device_map('off',{},kv,build,sizes,by).host)
        ck('empty placement follows the common no-offload default',
           loader._device_map('',{},kv,build,sizes,by) is None)
        def transient_build():
            temporary = ops.empty_bytes(12*gib)
            model = BudgetModel(lambda rows: 0)
            del temporary
            return model
        peak_need = loader.dry_need(transient_build, {}, dict(kv, chunk=128), 16*gib)
        ck('shared planner prices loading peak against physical free memory',
           12*gib <= peak_need < 12*gib+(1<<20), f'{peak_need/gib:.3f} GiB')
    with patch('snowllm.engine.gpu_mem_info',return_value=(96*gib,96*gib)), \
         patch.object(common,'gpu_mem_info',return_value=(96*gib,96*gib)):
        config=dict(ctx=131072,util=.95,chunk=128)
        def small_build():
            return BudgetModel(lambda rows: 0)
        one=loader.dry_need(small_build,{},dict(config,slots=1),32*gib)
        many=loader.dry_need(small_build,{},dict(config,slots=16),32*gib)
        ck('long-context offload prices extra states and outputs, not 15 full KV contexts',
           0<=many-one<(64<<20))
    from snowllm.ops._common import dry_alloc
    with ops.dry_load():
        model=BudgetModel(lambda rows: 0)
    with dry_alloc(), patch.object(ops,'drying',return_value=None), \
         patch.object(common,'gpu_mem_info',return_value=(88*gib,96*gib)):
        live=runners.Glm5NextRunner(model,None,512,1,plan_activations=False)
        ck('live runner needs no placement metadata or private overhead estimate',
           live.fixed_bytes(128)==0)
    with ops.dry_load():
        model=BudgetModel(lambda rows: rows*1024)
        dry=runners.Glm5NextRunner(model,None,64,2,max_prefill_tokens=128,
                                  plan_activations=False)
        budget=dry.act_bytes_for(128)
        arena,workspace,static=dry._activation_plan(128)
        ck('dry budget includes packed arena, workspace and static outputs',
           budget==arena.planner.high+workspace+static and workspace==128*1024)
        ck('decode can be wider than the prefill chunk',
           dry._activation_plan(1)[1]==2*1024
           and dry.act_bytes_for(1)>=dry._activation_plan(1)[1]+2*4096*2)
        with patch.object(dry,'_activation_plan',side_effect=AssertionError('replanned')):
            ck('budget is cached per chunk',dry.act_bytes_for(128)==budget)
        dry.plan_activations=True
        dry._reserve_activations()
        ck('startup freezes exactly the budgeted plan',
           dry.act_bytes==budget and dry.arena is arena and dry.ws.nbytes==workspace)
    # A changed forward allocation must change the dry budget, without tuning constants.
    with ops.dry_load():
        bigger=BudgetModel(lambda rows: rows*2048)
        larger=runners.Glm5NextRunner(bigger,None,64,2,max_prefill_tokens=128,
                                     plan_activations=False)
        ck('budget follows forward workspace queries',larger.act_bytes_for(128)==budget+128*1024)
        bigger=BudgetModel(lambda rows: rows*1024)
        bigger.width=8192
        larger=runners.Glm5NextRunner(bigger,None,64,2,max_prefill_tokens=128,
                                     plan_activations=False)
        ck('budget follows forward activation shapes',larger.act_bytes_for(128)==budget+128*4096*2)
    root=Path(os.environ.get('SNOWLLM_GLM_DIR',Path.home()/'models/UD-IQ2_XXS'))
    if root.is_dir():
        for mtp in (False,True):
            with ops.dry_load() as dry:
                model=loader.load(root,mtp=mtp,vision=False,device_map='off')
                counted=dry.device
                ck(f'MTP {mtp}: shared measurement equals final dry-loaded storage',
                   model.device_map.total_bytes == model.device_map.device_bytes == counted
                   and set(model.device_map.sizes) == set(placement.GROUPS))
                ck(f'MTP {mtp}: shared loader preserves optional NextN and metadata',
                   (model.mtp is not None)==mtp and model.visual is None and
                   model.vision_config is None and model.image_token_id is None)
                runner=runners.Glm5NextRunner(model,None,512,2,max_prefill_tokens=128,
                                              num_spec=2 if mtp else 0,plan_activations=False)
                expected_states = sum(a.history.nbytes+a.state.nbytes
                    for a in model.modules() if isinstance(a, LinearAttention))
                ck(f'MTP {mtp}: fixed state budget matches every live recurrent tensor',
                   runner.resources.states.bytes_for()==expected_states)
                ck(f'MTP {mtp}: placement floor and runtime pool use identical allocation',
                   runner.fixed_bytes(128)-expected_states==runner._pool_plan(0).bytes_for(512))
                saved=[]
                for layer in model.layers:
                    attn=layer.attention
                    names=(('history','state') if isinstance(attn,LinearAttention) else
                           ('bs','kv','index','carry_k','carry_g','sinks'))
                    saved += [(attn,name,getattr(attn,name,None)) for name in names]
                if mtp:
                    attn=model.mtp.attention
                    saved += [(attn,name,getattr(attn,name,None)) for name in
                              ('bs','kv','index','carry_k','carry_g','sinks')]
                for rows in (3,128,8192):
                    budget=runner.act_bytes_for(rows)
                    arena,workspace,static=runner._activation_plan(rows)
                    ck(f'MTP {mtp} M={rows}: actual forward paths determine budget',
                       budget==arena.planner.high+workspace+static and workspace>0
                       and len(arena.planner.plans)==(8 if mtp else 4))
                    ck(f'MTP {mtp} M={rows}: planning restores all attention state',
                       all(getattr(attn,name,None) is value for attn,name,value in saved))
                with patch.object(model,'forward',side_effect=RuntimeError('probe failed')):
                    try:
                        runner.act_bytes_for(7)
                    except RuntimeError as exc:
                        ck(f'MTP {mtp}: failed planning propagates the error',str(exc)=='probe failed')
                    else:
                        ck(f'MTP {mtp}: failed planning propagates the error',False)
                ck(f'MTP {mtp}: failed planning restores state and is not cached',
                   all(getattr(attn,name,None) is value for attn,name,value in saved)
                   and 7 not in runner._activation_plans and 7 not in runner._act)
                runner.plan_activations=True
                runner._reserve_activations()
                ck(f'MTP {mtp}: runtime allocation matches dry budget',
                   runner.act_bytes==runner.act_bytes_for(128))
    return ck.done()


if __name__=='__main__':
    sys.exit(main())
