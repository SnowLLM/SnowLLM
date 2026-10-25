# Windows page file {#windows-page-file}

If loading a model on Windows fails with a paging-file-too-small or memory allocation error, check the page file. Insufficient capacity can cause allocation failures even when GPU memory is still available.

## Choose the size

We recommend setting **virtual memory (the page file) to the same size as physical RAM**, with identical initial and maximum sizes. For example, with 128 GiB of physical RAM, enter `131072` MB for both.

Choose a drive with enough free space for the file.

## Configure Windows 11

1. Press **Win + R**, enter `sysdm.cpl`, and open **Advanced**.
2. Open **Performance → Settings → Advanced → Virtual memory → Change**.
3. Clear **Automatically manage paging file size for all drives**, select a drive with enough space, and choose **Custom size**.
4. Set both initial and maximum sizes to the physical RAM capacity and select **Set**.
5. Confirm the dialogs, **restart Windows**, then load the model again.

## Verify after restarting

While loading the model, open **Task Manager → Performance → Memory → Committed**. The left value is current usage; the right is the limit. If usage approaches the limit, increase the page file or close other applications. If commit capacity is still sufficient, check the [memory budget](./performance#memory-budget).

<details>
<summary>Optional: check the allocated page file capacity with PowerShell</summary>

```powershell
Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize
```

`AllocatedBaseSize` is the allocated capacity in MB; add the values if multiple files are listed.

Page files extend system commit capacity, and automatic growth can be too slow for large allocations. See Microsoft's [page file introduction](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/introduction-to-the-page-file) and [allocation failure guidance](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/slow-page-file-growth-memory-allocation-errors).

</details>
