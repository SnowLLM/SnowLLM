# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
import logging

from uvicorn.config import LOGGING_CONFIG


class PollingAccessFilter(logging.Filter):
    def filter(self, record):
        if record.levelno != logging.INFO or not isinstance(record.args, tuple) or len(record.args) != 5:
            return True
        _, method, path, _, status = record.args
        if not isinstance(path, str) or not isinstance(status, int):
            return True
        return not (method == 'GET' and 200 <= status < 300 and path.partition('?')[0] in
                    {'/daemon/status', '/daemon/metrics', '/daemon/logs', '/worker/status', '/health', '/metrics'})


def access_log_config():
    config = deepcopy(LOGGING_CONFIG)
    config['filters'] = {'polling': {'()': PollingAccessFilter}}
    config['loggers']['uvicorn.access']['filters'] = ['polling']
    return config
