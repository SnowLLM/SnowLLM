# SPDX-License-Identifier: Apache-2.0
import sys

import _harness
from snowllm.checkpoint.gguf import GGUF
from snowllm.checkpoint.gguf.names import config
from snowllm.checkpoint.gguf.source import find_gguf
from snowllm.checkpoint.gguf.tokenizer import build, stop_token_ids
from snowllm.models.glm5_next.glm5_next_weights import validate_config
from snowllm.serve.thinking import thinking_capabilities
from snowllm.serve.tool_parser import StreamingToolParser, parse_tool_calls, tool_types


def main():
    root = _harness.checkpoint_path('UD-IQ2_XXS', env='SNOWLLM_GLM_DIR')
    if not root.is_dir():
        _harness.skip("no GLM-5.3-Flash checkpoint")
    g = GGUF(find_gguf(root))
    c = config(g)
    validate_config(c)
    ck = _harness.Checks()
    ck("four complete GGUF tensor tables",len(g.parts)==4 and len(g.tensors)==1412)
    ck("all tensor data within its part",all(g.file_offset(n)+t.nbytes<=g.file_of(n).stat().st_size
                                            for n,t in g.tensors.items()))
    ck("45 trunk layers, 34 KDA and 11 sparse",c["num_hidden_layers"]==45 and
       c["layer_types"].count("linear_attention")==34 and c["layer_types"].count("deepseek_sparse_attention")==11)
    ck("all three generation stops",stop_token_ids(g)==(154820,154827,154829))
    t = build(g)
    ck("Chinese tokenizer ids",t.encode("你好，世界！",add_special_tokens=False)==[109377,3837,99011,6313])
    ck("glm4 groups digits",t.encode("1234567890",add_special_tokens=False)==[108714,100461,21,100928,24,15])
    msgs=[{"role":"user","content":"hi"}]
    for effort in ("low","high","max"):
        prompt=t.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,reasoning_effort=effort)
        ck(f"thinking prompt {effort}",f"Reasoning Effort: {effort.capitalize()}" in prompt and prompt.endswith("<think>"))
    ck("thinking levels exposed",thinking_capabilities(t)["levels"]==["low","high","max"])
    schema=tool_types([{"name":"f","parameters":{"properties":{"s":{"type":"string"},"n":{"type":"integer"}}}}])
    args={"s":"007","n":3}
    txt="hello<tool_call>f<arg_key>s</arg_key><arg_value>007</arg_value><arg_key>n</arg_key><arg_value>3</arg_value></tool_call>"
    text,calls=parse_tool_calls(txt,schema)
    ck("GLM tool arguments typed",text=="hello" and calls[0].arguments==args)
    p=StreamingToolParser(schema);text="";calls=[]
    for char in txt:
        s,cs=p.push(char);text+=s;calls+=cs
    s,cs=p.finish();text+=s;calls+=cs
    ck("GLM tools stream across any split",text=="hello" and len(calls)==1 and calls[0].arguments==args)
    # Existing Qwen and DeepSeek syntaxes stay recognized.
    for txt in ('<tool_call><function=f><parameter=n>\n3\n</parameter></function></tool_call>',
                '<｜DSML｜tool_calls><｜DSML｜invoke name="f"><｜DSML｜parameter name="n" string="false">3</｜DSML｜parameter></｜DSML｜invoke>'):
        ck("existing tool format preserved",parse_tool_calls(txt,schema)[1][0].arguments=={"n":3})
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
