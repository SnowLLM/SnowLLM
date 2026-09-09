# Never rebuild or edit the binding while tests are running

`scripts/run-tests.sh` runs each of the 73 tests in `tests/` as its own process, against
`SNOWLLM_LIB` (default: the installed `snowllm-kernels`). Two things silently invalidate a run in
progress, and neither announces itself:

- **Rebuilding `libsnowllm.so`.** The run splits across two builds — half the results describe code
  that no longer exists, and nothing in the output says so.
- **Editing `snowllm/_capi.py`.** It resolves every entry point at import and raises
  `"{lib} is missing N entry points this binding needs"` (`snowllm/_capi.py:416`) if one is absent.
  Declaring a symbol before building it turns every remaining test in the run into an import error —
  dozens of failures that are pure artifact. That is how 44 fake failures happened.

If you need to build, wait for the run or kill it first, and say the run was superseded.

# Blast radius

`tests/` splits cleanly into subsystems: DeepSeek-V4 (`test_deepseek4_*`, ~20 files), DFlash
(`test_dflash_*`, 7 files), and the rest. A DeepSeek-V4 op cannot break the DFlash draft path. Pick
by subsystem — `run-tests.sh` takes name substrings:

```sh
withgpu scripts/run-tests.sh deepseek4_attn      # just the ones the edit can reach
```

The DFlash e2e and server tests (`test_dflash_e2e.py`, `test_dflash_server.py`) are minutes each;
never pull them in as a general safety net. If a change touches shared code, name that code and pick
the cheapest test covering it.

# GPU

This is ROCm/HIP (`libamdhip64.so`). The card is shared — every test, benchmark, and `rocprof` run
goes through `withgpu`, as does anything else that initializes a HIP context.
