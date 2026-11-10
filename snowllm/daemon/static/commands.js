// SPDX-License-Identifier: Apache-2.0
export function modelLoadCommand(request, managementUrl) {
  const address = new URL(managementUrl);
  const host = address.hostname.replace(/^\[|\]$/g, '');
  const port = address.port || (address.protocol === 'https:' ? '443' : '80');
  const args = ['snowllm', 'load', '--admin-host', host, '--admin-port', port];
  if (request.id) args.push(`--id=${request.id}`);
  args.push(`${request.target}:${request.model}`);
  if (request.options.length) args.push('--', ...request.options);
  return args.map(value => /^[A-Za-z0-9_@%+=:,./-]+$/.test(value) ? value : `'${value.replace(/'/g, `'"'"'`)}'`).join(' ');
}
