# Windows 分页文件 {#windows-page-file}

在 Windows 上加载模型时，如果出现“页面文件太小”或内存分配失败，可以先检查分页文件；容量不足时，即使显存仍有余量，也可能无法分配内存。

## 应该设置多大？ {#choose-the-size}

我们推荐将**虚拟内存（分页文件）的大小设置为与物理内存容量相同**，初始大小和最大值填写相同数值。例如，物理内存为 128 GiB 时，两项均填写 `131072` MB。

所选磁盘需要有足够空间容纳这份文件。

## Windows 11 设置步骤 {#configure-windows-11}

1. 按 **Win + R**，输入 `sysdm.cpl`，打开“**高级**”选项卡。
2. 依次打开“**性能 → 设置 → 高级 → 虚拟内存 → 更改**”。
3. 取消“**自动管理所有驱动器的分页文件大小**”，选择空间充足的磁盘，选中“**自定义大小**”。
4. 将初始大小和最大值都设为物理内存容量，点击“**设置**”。
5. 确认各对话框，**重启 Windows**，再重新加载模型。

## 重启后如何确认？ {#verify-after-restarting}

加载模型时，查看“**任务管理器 → 性能 → 内存 → 已提交**”。左侧是已用额度，右侧是上限；若左侧接近右侧，可继续增大分页文件或关闭其他程序。提交额度仍充足时，继续检查[内存预算](./memory-management)。

<details>
<summary>可选：用 PowerShell 查看实际分页文件容量</summary>

```powershell
Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize
```

`AllocatedBaseSize` 是实际已分配容量，单位为 MB；有多个文件时将各行相加。

分页文件用于扩大系统提交额度；自动扩容也可能来不及响应大块分配，参见微软的[分页文件说明](https://learn.microsoft.com/zh-cn/troubleshoot/windows-client/performance/introduction-to-the-page-file)和[自动扩容导致分配失败的说明](https://learn.microsoft.com/zh-cn/troubleshoot/windows-client/performance/slow-page-file-growth-memory-allocation-errors)。

</details>
