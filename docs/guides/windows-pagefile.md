# Windows page file

If SnowLLM fails to allocate memory on native Windows, first **ensure the page file is at least
as large as the GPU's total memory (VRAM)**, then restart Windows and retry. This is SnowLLM's
troubleshooting baseline; some workloads and other running applications may need more.

This applies to allocation failures such as `malloc failed`, `hipMalloc` / `HIP out of memory`,
`Windows commit preflight failed`, or a message that the paging file is too small.

## Why allocations can fail

Windows limits system committed memory using RAM and page files. Reaching that limit can cause
allocations to fail even when GPU memory is still available. See Microsoft's
[page file introduction](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/introduction-to-the-page-file).

An automatically managed page file can also grow too slowly for a large allocation. Microsoft
documents this in [Memory allocation errors can be caused by slow page file growth](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/slow-page-file-growth-memory-allocation-errors).

## Choose the size

Use the GPU memory capacity reported by the driver to SnowLLM. On a Ryzen AI Max APU, use the
configured GPU memory allocation. Set the page file's **Initial size** to at least that capacity,
and its **Maximum size** no smaller than the initial size. Setting both to the same value
provides the full capacity before model loading.

| GPU memory | Minimum initial size (MB) | Minimum maximum size (MB) |
| --- | ---: | ---: |
| 32 GiB | 32768 | 32768 |
| 64 GiB | 65536 | 65536 |
| 96 GiB | 98304 | 98304 |
| 128 GiB | 131072 | 131072 |

The values above use 1024 MB per GiB. If using multiple page files, their combined allocated
capacity must meet the requirement. Leave enough free disk space for the configured page files
and normal system operation.

## Configure Windows 11

1. Press **Win + R**, enter `sysdm.cpl`, and open **Advanced**.
2. Under **Performance**, select **Settings**, then **Advanced**.
3. Under **Virtual memory**, select **Change**.
4. Clear **Automatically manage paging file size for all drives**.
5. Select a drive with enough free space, choose **Custom size**, and enter the initial and
   maximum sizes from the table or larger values.
6. Select **Set**, confirm the dialogs, and **restart Windows**.

## Verify after restarting

Run this read-only command in PowerShell:

```powershell
Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage
```

`AllocatedBaseSize` is the allocated page file capacity in MB; add the values if multiple files
are listed. `CurrentUsage` is the amount currently used, not the capacity to check. Microsoft
documents these fields in [Win32_PageFileUsage](https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-pagefileusage).

For a 96 GiB GPU, the combined `AllocatedBaseSize` must be **at least 98304**. A large maximum
setting alone does not establish that this capacity is already allocated.

Retry model loading while watching **Task Manager → Performance → Memory → Committed**.
If committed memory approaches its limit, increase the page file further or close other
memory-intensive applications. If GPU memory itself is exhausted, reduce the context length,
concurrent request count, or GPU memory budget; see [Performance tuning](./performance).
