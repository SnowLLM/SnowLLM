# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections import OrderedDict
from datetime import datetime, timezone
import uuid


TERMINAL = {'completed', 'failed', 'cancelled', 'interrupted'}


def now():
    return datetime.now(timezone.utc).isoformat()


class TaskQueue:
    def __init__(self, *, history_limit=100, pending_limit=64, wait_timeout=3600):
        self.records = OrderedDict()
        self.work = {}
        self.runner = None
        self.execution = None
        self.active = None
        self.closed = False
        self.history_limit = history_limit
        self.pending_limit = pending_limit
        self.wait_timeout = wait_timeout

    @property
    def busy(self):
        return any(row['kind'] != 'inference' and row['state'] not in TERMINAL
                   for row in self.records.values())

    def _trim(self):
        finished = sorted((key for key, row in self.records.items() if row['state'] in TERMINAL),
                          key=lambda key: self.records[key]['finished_at'])
        for key in finished[:-self.history_limit]:
            self.records.pop(key)

    def create(self, kind, model=None, target=None, *, task_id=None):
        if self.closed:
            raise ValueError('service is shutting down')
        task_id = task_id or uuid.uuid4().hex
        if task_id in self.records:
            raise ValueError('task ID already exists')
        row = dict(id=task_id, kind=kind, model=model, target=target, state='queued',
                   created_at=now(), started_at=None, finished_at=None, error=None,
                   wait_reason=None, waiting_for=None, cancel_requested=False, cancellable=False)
        self.records[task_id] = row
        self._trim()
        return row

    def submit(self, kind, model, target, run, *, cancel=None, task_id=None, on_finish=None):
        if sum(row['state'] == 'queued' and row['kind'] != 'inference'
               for row in self.records.values()) >= self.pending_limit:
            raise ValueError('task queue is full')
        row = self.create(kind, model, target, task_id=task_id)
        row['cancellable'] = True
        future = asyncio.get_running_loop().create_future()
        timer = asyncio.get_running_loop().call_later(self.wait_timeout, self._expire, row['id'])
        self.work[row['id']] = dict(run=run, cancel=cancel, future=future, timer=timer, on_finish=on_finish, started=False)
        if self.runner is None or self.runner.done():
            self.runner = asyncio.create_task(self._run())
        return row, future

    def update(self, task_id, **values):
        row = self.records[task_id]
        if row['state'] not in TERMINAL:
            row.update(values)

    def finish(self, task_id, state, error=None):
        row = self.records[task_id]
        if row['state'] in TERMINAL:
            return
        row.update(state=state, error=error, finished_at=now(), wait_reason=None,
                   waiting_for=None, cancellable=False)
        work = self.work.pop(task_id, None)
        if work:
            if work['timer']:
                work['timer'].cancel()
            try:
                if work['on_finish']:
                    work['on_finish'](dict(row))
            except Exception as exc:
                row.update(state='failed', error=f'could not save task result: {exc}')
            finally:
                if not work['future'].done():
                    work['future'].set_result(dict(row))
        self._trim()

    def _expire(self, task_id):
        row = self.records.get(task_id)
        if row and row['state'] == 'queued':
            self.finish(task_id, 'failed', 'task queue wait timed out')

    async def _run(self):
        while not self.closed:
            row = next((row for row in self.records.values()
                        if row['kind'] != 'inference' and row['state'] == 'queued'), None)
            if row is None:
                break
            work = self.work[row['id']]
            work['timer'].cancel()
            self.active = row
            row.update(state='running', started_at=now(), wait_reason=None,
                       cancellable=work['cancel'] is not None)
            async def execute():
                work['started'] = True
                await work['run'](row)

            self.execution = asyncio.create_task(execute())
            state, error = 'completed', None
            try:
                await self.execution
            except asyncio.CancelledError:
                state = 'cancelled'
            except Exception as exc:
                state, error = 'failed', str(exc)
            finally:
                if event := work.get('cancel_done'):
                    await event.wait()
                self.finish(row['id'], 'cancelled' if row['cancel_requested'] else state, error)
                self.active = self.execution = None

    def snapshot(self):
        rows = []
        previous = self.active['id'] if self.active else None
        position = 0
        for row in self.records.values():
            item = dict(row)
            if item['kind'] != 'inference' and item['state'] == 'queued':
                position += 1
                item.update(position=position, wait_reason='previous_task', waiting_for=previous)
                previous = item['id']
            rows.append(item)
        return rows

    async def cancel(self, task_id):
        row = self.records.get(task_id)
        if row is None:
            raise LookupError('unknown task')
        if row['state'] in TERMINAL or row['cancel_requested']:
            return dict(row)
        if not row['cancellable']:
            raise ValueError('this task cannot be cancelled after it starts')
        if row['state'] == 'queued':
            self.finish(task_id, 'cancelled')
        else:
            work = self.work[task_id]
            row.update(cancel_requested=True, state='cancelling', cancellable=False)
            work['cancel_done'] = asyncio.Event()
            try:
                if row['kind'] != 'inference' and not work['started']:
                    self.execution.cancel()
                else:
                    await work['cancel']()
            except BaseException:
                if row['state'] not in TERMINAL:
                    row.update(cancel_requested=False, state='running', cancellable=True)
                raise
            finally:
                work['cancel_done'].set()
        return dict(row)

    def begin_request(self, model, target, cancel):
        row = self.create('inference', model, target)
        row.update(state='running', started_at=now(), cancellable=True)
        self.work[row['id']] = dict(cancel=cancel, timer=None,
                                   future=asyncio.get_running_loop().create_future(), on_finish=None)
        return row

    def finish_request(self, task_id, state, error=None):
        self.finish(task_id, state, error)

    async def close(self):
        self.closed = True
        requests = []
        for row in list(self.records.values()):
            if row['state'] == 'queued':
                self.finish(row['id'], 'cancelled')
            elif row['kind'] == 'inference' and row['state'] not in TERMINAL:
                requests.append(self.work[row['id']]['future'])
                await self.cancel(row['id'])
        if self.execution:
            self.execution.cancel()
        if self.runner:
            await self.runner
        if requests:
            await asyncio.wait(requests, timeout=5)
