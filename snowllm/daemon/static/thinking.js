// SPDX-License-Identifier: Apache-2.0
export function thinkingProfile(capabilities) {
  const levels = (capabilities?.levels || []).filter(level => typeof level === 'string' && level.trim());
  const options = [['', capabilities?.required ? 'Always on' : 'Default']];
  if (capabilities?.switch && !capabilities?.required) {
    options.push(['off', 'Off']);
    if (!levels.some(level => level.trim().toLowerCase() !== 'none')) options.push(['on', 'On']);
  }
  for (const level of levels) {
    if (!(capabilities?.required && level.trim().toLowerCase() === 'none')) {
      options.push([`effort:${level}`, level]);
    }
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
