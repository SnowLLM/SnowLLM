# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path


def main():
    import torch
    from mineru.model.mfr.pp_formulanet.predict_formula import FormulaRecognizer
    from snowllm.checkpoint.loader import load_auxiliary
    from snowllm.models.mineru.pp_formulanet import FormulaRunner
    from snowllm.mineru.graphs import GraphCache

    root = Path(
        os.environ.get("SNOWLLM_MINERU_DIR", "~/models/MinerU-4.1")
    ).expanduser()
    root = root / "small-models/MFR/pp_formulanet_plus_m"
    native = load_auxiliary(root / "PP-FormulaNet_plus-M.pth", "PPFormulaNetModel")
    reference = FormulaRecognizer(str(root), device="cuda").net
    native._native_graph_cache = GraphCache(lambda: 12 << 30)
    native._memory_admit = native._native_graph_cache.admit
    torch.manual_seed(73)
    with torch.inference_mode():
        for batch in (1, 2, 1):
            encoded = (
                torch.randn(batch, 197, 2048, dtype=torch.float16, device="cuda") * 0.1
            )
            projected = reference.head.enc_to_dec_proj(encoded)
            runner = FormulaRunner(native, batch, 197)
            runner.prepare(encoded)
            runner.position = torch.zeros(1, dtype=torch.int64, device="cuda")
            empty = encoded.new_empty(batch, 16, 0, 32)
            cache = [(empty,) * 4 for _ in range(6)]
            maximum = 0.0
            for index in range(130):
                ids = torch.randint(4, 3000, (batch, 1), device="cuda")
                actual = runner.step(ids, index)
                expected = reference.head.decoder(
                    input_ids=ids,
                    encoder_hidden_states=projected,
                    past_key_values=cache,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                torch.testing.assert_close(
                    actual, expected.logits, atol=0.03, rtol=0.003
                )
                maximum = max(maximum, float((actual - expected.logits).abs().max()))
                cache = expected.past_key_values
                for i, states in enumerate(cache):
                    difference = (
                        runner.cache[i, 0, :, :, : index + 1].float()
                        - states[0].float()
                    )
                    assert float(difference.norm() / states[0].float().norm()) < 0.003
                if index in (0, 63, 64, 127, 128, 129):
                    print(
                        "CHECK",
                        batch,
                        index + 1,
                        "capacity",
                        runner.capacity,
                        "max_error",
                        maximum,
                        flush=True,
                    )
            runner.close()
            print(
                "PASS: own formula decoder, cache growth and refreshed graph inputs",
                batch,
                flush=True,
            )
        assert native._native_graph_cache.replays > 300, (
            native._native_graph_cache.status()
        )
        print("GRAPHS", native._native_graph_cache.status(), flush=True)
    native._native_graph_cache.clear()


if __name__ == "__main__":
    main()
