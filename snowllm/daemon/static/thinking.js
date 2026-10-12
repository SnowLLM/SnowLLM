// SPDX-License-Identifier: Apache-2.0
export function thinkingProfile(capabilities) {
  const options = [['', 'Model default']];
  if (capabilities?.switch) options.push(['off', 'Off'], ['on', 'On']);
  for (const level of capabilities?.levels || []) {
    if (typeof level === 'string') options.push([`effort:${level}`, level]);
  }
  return { options };
}

export function thinkingParameters(level, capabilities) {
  if (!thinkingProfile(capabilities).options.some(([value]) => value === level)) {
    throw new Error('This thinking level is not supported by the selected model.');
  }
  if (level === '') return {};
  if (level === 'off') return { enable_thinking: false };
  if (level === 'on') return { enable_thinking: true };
  return { reasoning_effort: level.slice('effort:'.length) };
}
