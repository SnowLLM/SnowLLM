# SPDX-License-Identifier: Apache-2.0
import gc
import os
from pathlib import Path
from unittest.mock import patch


def main():
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from transformers import AutoModelForImageTextToText, AutoProcessor

    root = Path(
        os.environ.get("SNOWLLM_MINERU_DIR", "~/models/MinerU-4.1")
    ).expanduser()
    image = Image.new("RGB", (576, 256), "white")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    drawing = ImageDraw.Draw(image)
    drawing.text((20, 40), "SnowLLM document parsing", font=font, fill="black")
    drawing.text((20, 100), "The answer is 42.", font=font, fill="black")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "\nText Recognition:"},
            ],
        }
    ]
    processor = AutoProcessor.from_pretrained(root)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    reference = AutoModelForImageTextToText.from_pretrained(
        root, dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa"
    ).eval()
    with torch.inference_mode():
        device_inputs = inputs.to("cuda", dtype=torch.bfloat16)
        output = reference.generate(**device_inputs, max_new_tokens=64, do_sample=False)
        expected = output[0, inputs["input_ids"].shape[1] :].tolist()
    print("REFERENCE", expected, processor.tokenizer.decode(expected), flush=True)
    del reference, device_inputs, output
    gc.collect()
    torch.cuda.empty_cache()
    from snowllm.checkpoint import loader
    from snowllm.engine import Engine, SamplingParams
    from snowllm.models.multimodal import prepare

    model = loader.load(
        root, mtp=False, kv=dict(ctx=2048, slots=1, util=0.9, chunk=128)
    )
    engine = Engine(
        model,
        max_model_len=2048,
        max_num_seqs=1,
        num_spec=0,
        stop_token_ids=(151643, 151645),
        prefill_chunk=128,
        enforce_eager=True,
    )
    from snowllm.ops.arena import dummy_run

    with patch("snowllm.ops.arena.dummy_run", wraps=dummy_run) as probe:
        multimodal = prepare(model, processor, messages, [image])
        repeated = prepare(model, processor, messages, [image])
        assert probe.call_count == 1, probe.call_count
        torch.testing.assert_close(multimodal["embeds"], repeated["embeds"])
    del repeated
    request = engine.add(
        multimodal.pop("prompt"),
        SamplingParams(temperature=0, max_new_tokens=64),
        **multimodal,
    )
    engine.run()
    print("NATIVE", request.out, processor.tokenizer.decode(request.out), flush=True)
    assert request.out == expected, (request.out, expected)
    print(
        "PASS: native MinerU image generation matches Transformers token for token",
        flush=True,
    )


if __name__ == "__main__":
    main()
