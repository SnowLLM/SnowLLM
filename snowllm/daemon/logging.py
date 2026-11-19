# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
import logging
import re

from uvicorn.config import LOGGING_CONFIG


class PollingAccessFilter(logging.Filter):
    def filter(self, record):
        if record.levelno != logging.INFO or not isinstance(record.args, tuple) or len(record.args) != 5:
            return True
        _, method, path, _, status = record.args
        if not isinstance(path, str) or not isinstance(status, int) or method != 'GET' or not 200 <= status < 300:
            return True
        path = path.partition('?')[0]
        return (path not in {'/daemon/status', '/daemon/metrics', '/daemon/logs', '/worker/status', '/health', '/metrics',
                             '/daemon/benchmarks'}
                and re.fullmatch(r'/daemon/benchmarks/[a-f0-9]{32}', path) is None)


def access_log_config():
    config = deepcopy(LOGGING_CONFIG)
    config['filters'] = {'polling': {'()': PollingAccessFilter}}
    config['loggers']['uvicorn.access']['filters'] = ['polling']
    return config
