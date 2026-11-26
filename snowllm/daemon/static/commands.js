// SPDX-License-Identifier: Apache-2.0
function managementArgs(managementUrl) {
  const address = new URL(managementUrl);
  if (address.protocol === 'https:') return ['--admin-url', address.origin];
  return ['--admin-host', address.hostname.replace(/^\[|\]$/g, ''), '--admin-port', address.port || '80'];
}

export function modelLoadCommand(request, managementUrl) {
  const args = ['snowllm', 'load', ...managementArgs(managementUrl)];
  if (request.id) args.push(`--id=${request.id}`);
  args.push(`${request.target}:${request.model}`);
  if (request.options.length) args.push('--', ...request.options);
  return args.map(value => /^[A-Za-z0-9_@%+=:,./-]+$/.test(value) ? value : `'${value.replace(/'/g, `'"'"'`)}'`).join(' ');
}

export function benchmarkCommand(request, managementUrl) {
  const args = ['snowllm', 'bench', 'run', ...managementArgs(managementUrl), `--model-id=${request.model_id}`];
  if (request.workload) args.push('--workload', request.workload);
  for (const key of ['isl', 'osl', 'conc', 'reps', 'warm', 'thinking']) {
    args.push(`--${key}`, Array.isArray(request[key]) ? request[key].join(',') : String(request[key]));
  }
  return args.map(value => /^[A-Za-z0-9_@%+=:,./-]+$/.test(value) ? value : `'${value.replace(/'/g, `'"'"'`)}'`).join(' ');
}
