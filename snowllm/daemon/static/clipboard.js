// SPDX-License-Identifier: Apache-2.0
export async function writeClipboard(text) {
  if (navigator.clipboard) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const focus = document.activeElement;
  const field = document.createElement('textarea');
  field.value = text;
  field.readOnly = true;
  field.className = 'sr-only';
  document.body.append(field);
  try {
    field.select();
    if (!document.execCommand('copy')) throw new Error('Copy failed');
  } finally {
    field.remove();
    focus?.focus();
  }
}
