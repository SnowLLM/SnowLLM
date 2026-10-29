# Windows 额外设置 {#windows-page-file}

## 分页文件 {#page-file}

<span id="choose-the-size"></span>

建议将分页文件的初始大小和最大值都设为整机物理内存容量：32G 填 `32768` MB，128G 填 `131072` MB。

<span id="configure-windows-11"></span>

1. 按 **Win + R**，输入 `sysdm.cpl`。
2. 打开“**高级 → 性能设置 → 高级 → 虚拟内存 → 更改**”。
3. 取消自动管理，选择空间充足的磁盘，填写自定义大小，点击“**设置**”。
4. 确认并重启 Windows。

<span id="verify-after-restarting"></span>

重启后，在“**任务管理器 → 性能 → 内存 → 已提交**”检查额度。分页文件可缓解提交额度不足，但不会增加物理显存。参见[微软说明](https://learn.microsoft.com/zh-cn/troubleshoot/windows-client/performance/introduction-to-the-page-file)。

## BIOS UMA 设置 {#bios-uma}

::: warning 其他 APU 请特别注意 UMA 设置
**Ryzen AI Max+ 395 和 Ryzen AI Max+ PRO 495 一般无需特别调整 UMA。** 使用**其他 APU** 时，请重点检查 BIOS 中的 UMA 分配；默认分配过低可能限制推理可用的显存。
:::

显存不足时，将 UMA 调到厂商允许的最大值。常见 BIOS 路径为：

**Advanced → AMD CBS → NBIO Common Options → GFX Configuration → UMA Frame Buffer Size**

部分 BIOS 需先将 `UMA Mode` 或 `iGPU Configuration` 设为 `UMA_SPECIFIED` 才能调整。菜单因厂商而异，参见[厂商示例](https://www.asrock.com/support/faq.asp?id=464)。保存后重启。

任务管理器显示的**专用 GPU 内存**对应 UMA，**共享 GPU 内存**通常约为扣除 UMA 后剩余内存的一半，具体取决于厂商和驱动：

```text
推理可用显存上限 ≈ 专用内存（UMA）+ 共享内存
```

普通 APU 通常最多能把 **50%** 的内存分给 UMA，GPU 还能使用剩余内存的一半，合计约 **75%**。

Ryzen AI Max+ 395 和 Ryzen AI Max+ PRO 495 支持更高的 UMA 比例；如果出厂已设为 **75%**，一般不用调整。具体以厂商设置为准。

我们的 **7940HS / Radeon 780M、32G 内存**将 UMA 设为最大值 **16G**：剩余 16G 的一半约为 **8G 共享内存**，合计 **16 + 8 = 24G**。测试使用的 **MiniCPM-V 4.7 35B-A3B Q3（`mix_q3gud_im`）需要准备约 24G 推理显存预算**，具体占用随上下文等配置变化。实际可用量以任务管理器和引擎报告为准。
