# SPDX-License-Identifier: Apache-2.0
import asyncio


class BenchmarkLease:
    def __init__(self, registry):
        self.registry = registry
        self.owner = None
        self.model_id = None
        self.started = False

    def reserve(self, owner, model_id, instance=None):
        if self.owner or self.registry.operation or self.registry.closing:
            raise ValueError('wait for the current benchmark or model operation to finish')
        record = self.registry.acquire(model_id)
        record.active -= 1
        row = record.snapshot()
        if instance is not None and row.get('instance') != instance:
            raise ValueError('the queued benchmark refers to a model instance that is no longer loaded')
        self.owner, self.model_id, self.started = owner, record.id, False
        return row

    def idle(self):
        return all(not record.active and (not record.state or not record.state.engine.busy)
                   for record in self.registry.records.values())

    def begin(self, owner):
        self.check(owner)
        if not self.idle():
            raise ValueError('waiting for active requests to finish')
        self.started = True
        return self.registry.records[self.model_id].snapshot()

    def check(self, owner):
        if not owner or self.owner != owner:
            raise ValueError('benchmark reservation does not match')

    async def release(self, owner):
        self.check(owner)
        if self.started:
            engine = self.registry.records[self.model_id].state.engine
            engine.abort_requests()
            while not self.idle():
                await asyncio.sleep(.1)
        self.owner = self.model_id = None
        self.started = False
