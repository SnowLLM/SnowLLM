#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import ipaddress
import os
import random
from datetime import datetime, timedelta

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "access.log")
SEED = 20261002
ANOMALY_IP = "203.0.113.66"

NORMAL_PATHS = [
    ("GET", "/"), ("GET", "/index.html"), ("GET", "/about"), ("GET", "/pricing"),
    ("GET", "/api/v1/items"), ("GET", "/api/v1/users"), ("GET", "/api/v1/health"),
    ("POST", "/api/v1/orders"), ("GET", "/static/app.js"), ("GET", "/static/style.css"),
    ("GET", "/static/logo.svg"), ("GET", "/favicon.ico"), ("GET", "/docs/getting-started"),
    ("GET", "/docs/api"), ("GET", "/blog/2026/09/release"), ("GET", "/search?q=pricing"),
]
NORMAL_UA = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/127.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gecko/20100101 Firefox/128.0",
    "curl/8.5.0",
]
ANOMALY_PATHS = [
    ("POST", "/admin/login"), ("POST", "/admin/login"), ("POST", "/admin/login"),
    ("POST", "/wp-login.php"), ("GET", "/phpmyadmin/index.php"), ("GET", "/admin"),
    ("POST", "/api/v1/auth/token"), ("GET", "/../../etc/passwd"), ("GET", "/.env"),
    ("GET", "/config.json"),
]


def main():
    rng = random.Random(SEED)
    normal_ips = [str(ipaddress.ip_address("192.0.2.0") + i) for i in range(1, 21)]
    normal_ips += [str(ipaddress.ip_address("198.51.100.0") + i) for i in range(1, 11)]
    normal_ips = normal_ips[:24]

    t = datetime(2026, 10, 2, 0, 0, 0)
    lines = []
    for ip in normal_ips:
        for _ in range(rng.randint(2, 5)):
            t += timedelta(seconds=rng.randint(0, 6))
            method, path = rng.choice(NORMAL_PATHS)
            status = rng.choices([200, 304, 404, 500], weights=[80, 12, 7, 1])[0]
            lines.append((t, f'{ip} - - [{t:%d/%b/%Y:%H:%M:%S} +0000] "{method} {path} HTTP/1.1" '
                             f'{status} {rng.randint(150, 60000)} "-" "{rng.choice(NORMAL_UA)}"'))

    burst = datetime(2026, 10, 2, 3, 14, 0)
    for i in range(38):
        ts = burst + timedelta(seconds=i * 3 + rng.randint(0, 2))
        method, path = rng.choice(ANOMALY_PATHS)
        status = 401 if path in ("/admin/login", "/api/v1/auth/token") else rng.choice([403, 404])
        lines.append((ts, f'{ANOMALY_IP} - - [{ts:%d/%b/%Y:%H:%M:%S} +0000] "{method} {path} HTTP/1.1" '
                          f'{status} {rng.randint(120, 900)} "-" "python-requests/2.31.0"'))

    lines.sort(key=lambda x: x[0])
    with open(OUT, "w") as f:
        for _, line in lines:
            f.write(line + "\n")
    print(f"{OUT}: {len(lines)} lines")


if __name__ == "__main__":
    main()
