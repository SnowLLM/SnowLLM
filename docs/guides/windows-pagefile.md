# Windows additional settings {#windows-page-file}

## Page file {#page-file}

<span id="choose-the-size"></span>

Set the page file's initial and maximum sizes to the installed physical RAM capacity: `32768` MB for 32 GiB, or `131072` MB for 128 GiB.

<span id="configure-windows-11"></span>

1. Press **Win + R** and enter `sysdm.cpl`.
2. Open **Advanced → Performance Settings → Advanced → Virtual memory → Change**.
3. Clear automatic management, select a drive with enough free space, enter the custom size and select **Set**.
4. Confirm and restart Windows.

<span id="verify-after-restarting"></span>

After restarting, check **Task Manager → Performance → Memory → Committed**. A page file can relieve commit pressure but does not increase physical GPU memory. See [Microsoft's explanation](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/introduction-to-the-page-file).

## BIOS UMA settings {#bios-uma}

::: warning Other APUs: check UMA settings
**Ryzen AI Max+ 395 and Ryzen AI Max+ PRO 495 generally need no special UMA adjustment.** On **other APUs**, pay particular attention to BIOS UMA allocation: a low default can limit the GPU memory available for inference.
:::

If GPU memory is insufficient, increase UMA to the manufacturer's supported maximum. A common BIOS path is:

**Advanced → AMD CBS → NBIO Common Options → GFX Configuration → UMA Frame Buffer Size**

Some BIOS versions require `UMA Mode` or `iGPU Configuration` to be set to `UMA_SPECIFIED` first. Menus vary by manufacturer; see this [manufacturer example](https://www.asrock.com/support/faq.asp?id=464). Save and restart.

Task Manager's **Dedicated GPU memory** corresponds to UMA. **Shared GPU memory** is commonly about half the RAM remaining after UMA reservation, depending on the manufacturer and driver:

```text
Inference memory ceiling ≈ dedicated memory (UMA) + shared memory
```

Ordinary APUs can commonly reserve up to **50%** of RAM for UMA. The GPU can also use half of the remaining RAM, bringing the combined ceiling to approximately **75%**.

Ryzen AI Max+ 395 and Ryzen AI Max+ PRO 495 support a higher UMA proportion. If the factory setting is already **75%**, adjustment is usually unnecessary. Check the manufacturer's settings for your machine.

Our **7940HS / Radeon 780M with 32 GiB RAM** used its maximum **16 GiB UMA** setting. Half the remaining 16 GiB gives about **8 GiB shared memory**, for **16 + 8 = 24 GiB** combined. Plan for approximately **24 GiB of inference memory for the tested MiniCPM-V 4.7 35B-A3B Q3 (`mix_q3gud_im`)**; actual usage varies with context and other settings. Check Task Manager and the engine's report for the available budget.
