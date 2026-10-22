# Windows 分页文件（page file） {#windows-page-file}

在原生 Windows 上运行 SnowLLM，遇到内存分配失败时，首先**确保分页文件容量至少等于 GPU 总显存大小（VRAM）**，然后重启 Windows 再试。这是 SnowLLM 的排查基线；实际工作负载及其他应用可能需要更大的分页文件。

适用报错包括 `malloc failed`、`hipMalloc` / `HIP out of memory`、`Windows commit preflight failed`，以及“页面文件太小，无法完成操作”等分配失败提示。

## 为什么显存有余量也会失败？ {#why-allocations-can-fail}

Windows 的系统提交内存上限由 RAM 和分页文件共同决定。达到提交上限时，即使 GPU 显存还有余量，内存分配也可能失败。见微软的[分页文件介绍](https://learn.microsoft.com/zh-cn/troubleshoot/windows-client/performance/introduction-to-the-page-file)。

自动管理的分页文件还可能来不及为大块分配扩容。微软在[分页文件增长缓慢可能导致内存分配错误](https://learn.microsoft.com/zh-cn/troubleshoot/windows-client/performance/slow-page-file-growth-memory-allocation-errors)中说明了这一情况。

## 应该设置多大？ {#choose-the-size}

以驱动向 SnowLLM 报告的 GPU 显存容量为准。Ryzen AI Max APU 使用配置给 GPU 的显存大小。将分页文件的**初始大小**设为至少等于该容量，**最大值**不小于初始大小。两者设为相同值，可在加载模型前准备好全部容量。

| GPU 显存 | 初始大小下限（MB） | 最大值下限（MB） |
| --- | ---: | ---: |
| 32 GiB | 32768 | 32768 |
| 64 GiB | 65536 | 65536 |
| 96 GiB | 98304 | 98304 |
| 128 GiB | 131072 | 131072 |

上表按每 GiB 对应 1024 MB 换算。使用多个分页文件时，已分配容量之和需满足要求。所在磁盘应有足够空间容纳配置的分页文件，并留出正常使用所需的余量。

## Windows 11 设置步骤 {#configure-windows-11}

1. 按 **Win + R**，输入 `sysdm.cpl`，打开“**高级**”选项卡。
2. 在“**性能**”区域点击“**设置**”，再选择“**高级**”。
3. 在“**虚拟内存**”区域点击“**更改**”。
4. 取消勾选“**自动管理所有驱动器的分页文件大小**”。
5. 选择空间充足的磁盘，勾选“**自定义大小**”，按上表或更大容量填写初始大小和最大值。
6. 点击“**设置**”，确认各对话框，然后**重启 Windows**。

## 重启后如何确认？ {#verify-after-restarting}

在 PowerShell 中运行以下只读命令：

```powershell
Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage
```

`AllocatedBaseSize` 是已分配的分页文件容量，单位为 MB；有多个文件时将各行相加。`CurrentUsage` 是当前使用量，不是要核对的容量。字段含义见微软的 [Win32_PageFileUsage 文档](https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-pagefileusage)。

对于 96 GiB 显存，`AllocatedBaseSize` 的合计值应**至少为 98304**。仅将最大值设大，并不能说明这些空间已经分配。

重新加载模型，同时观察“**任务管理器 → 性能 → 内存 → 已提交**”。若已提交内存接近上限，继续增大分页文件或关闭其他占用内存较多的应用。若 GPU 显存本身不足，应缩短上下文、减少并发请求数或降低 GPU 内存预算，见[性能调优](./performance)。
