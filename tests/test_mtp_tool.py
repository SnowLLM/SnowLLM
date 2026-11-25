# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'snowllm/engine/mtp_tool.py'
spec = importlib.util.spec_from_file_location('mtp_tool', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MtpToolFormat = module.MtpToolFormat


def tool(name, properties=(), required=()):
    params = dict(type='object', properties={p: {'type': 'string'} for p in properties},
                  required=list(required))
    return {'type': 'function', 'function': {'name': name, 'parameters': params}}


class Tokenizer:
    def __init__(self):
        parts = ['<tool_call>', '</tool_call>', '<eos>', '=list', '=file', '_names',
                 'function', 'parameter', '</', '  \n', '\n\n', ' \n']
        self.pieces = sorted(parts, key=len, reverse=True)
        self.id = {s: 1000+i for i, s in enumerate(parts)}
        self.rev = {v: k for k, v in self.id.items()}

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            word = next((w for w in self.pieces if text.startswith(w)), text[0])
            result.append(self.id[word] if word in self.id else ord(word))
            text = text[len(word):]
        return result

    def decode(self, tokens):
        return ''.join(self.rev[t] if t in self.rev else chr(t) for t in tokens)


class ToolDraftTests(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()
        self.enc = self.tok.encode
        self.tools = [tool('list_names', ['file'], ['file'])]
        self.fmt = MtpToolFormat.from_tools(self.tok, self.tools, self.enc('<eos>'))

    def advanced(self, text, fmt=None):
        state = (fmt or self.fmt).new_state()
        state.advance(self.enc(text))
        return state

    def test_registered_header_formats_only_propose_after_committed_opening(self):
        for model, marker, header in (
                ('deepseek_v4', '<｜DSML｜ calls>\n', '<｜DSML｜ invoke name="list_names">\n'),
                ('deepseek_v4', '<｜DSML｜tool_calls>\n', '<｜DSML｜invoke name="list_names">\n'),
                ('lfm2_moe', '<|tool_call_start|>', '[list_names(')):
            fmt = module.TOOL_DRAFT_FORMATS[model](self.tok, self.tools, ())
            state = fmt.new_state()
            ids = self.enc(marker)
            state.advance(ids[:-1])
            self.assertEqual(state.propose(8), [])
            state.advance(ids[-1:])
            expected = self.enc(marker + header)[len(ids):]
            self.assertEqual(state.propose(8), expected[:8])
            state.advance(expected[:3])
            self.assertEqual(state.propose(8), expected[3:11])
            state.advance(self.enc('!'))
            self.assertEqual(state.propose(8), [])

    def test_open_required_and_first_native_boundary(self):
        for text in ('', 'example <tool', '<function=list_names>', 'just plain text'):
            self.assertEqual(self.advanced(text).propose(7), [])
        s = self.advanced('prose <tool_call>')
        self.assertEqual(s.propose(7), self.enc('\n<function=list_names>\n')[:7])
        self.assertEqual(s.propose(0), [])
        self.assertEqual(s.propose(3), s.propose(10)[:3])
        self.assertIn(self.tok.id['=list'], s.propose(20))
        self.assertIn(self.tok.id['=file'], s.propose(40))

    def test_multiple_functions_only_common_tokens(self):
        fmt = MtpToolFormat.from_tools(self.tok, [tool('list_names'), tool('fetch')])
        s = self.advanced('<tool_call>', fmt)
        self.assertEqual(s.propose(99), self.enc('\n<function'))
        s.advance(self.enc('\n<function'))
        self.assertEqual(s.propose(99), [])
        s.advance(self.enc('=list'))
        self.assertEqual(s.propose(99), self.enc('_names>\n</function>\n</tool_call>'))

    def test_multiple_fields_optional_close_and_actual_choice(self):
        fmt = MtpToolFormat.from_tools(self.tok, [tool('f', ['file','path'], ['file'])])
        s = self.advanced('<tool_call>\n<function=f>\n', fmt)
        self.assertEqual(s.propose(99), self.enc('<parameter'))
        s.advance(self.enc('<parameter=path>\nv\n</parameter'))
        self.assertEqual(s.propose(99), self.enc('>\n<parameter=file>'))
        s.advance(self.enc('>\n<parameter=file>\nx\n</parameter>\n'))
        self.assertEqual(s.propose(99), self.enc('</function>\n</tool_call>'))
        other = self.advanced('<tool_call>\n<function=f>\n<parameter=file>\n1\n</parameter>\n',fmt)
        self.assertEqual(other.propose(7), [])
        other.advance(self.enc('<parameter'))
        self.assertEqual(other.propose(99), self.enc('=path>'))

    def test_free_value_and_fused_newline_closing(self):
        start = '<tool_call>\n<function=list_names>\n<parameter=file>'
        for val in ('\n', '\n\n', '\nx\n', '\nx inline </parameter', '\nx\n</', '\n<tool_call>'):
            with self.subTest(val=val):
                self.assertEqual(self.advanced(start+val).propose(7), [])
        for val in ('\nv\n', '\nx \n', '\n\n', '\n  \n'):
            s = self.advanced(start+val+'</parameter')
            self.assertEqual(s.propose(99), self.enc('>\n</function>\n</tool_call>'))

    def test_rejection_replay_eos_and_multiple_tool_calls(self):
        fmt = MtpToolFormat.from_tools(self.tok, [tool('list_names'),tool('fetch')], self.enc('<eos>'))
        text = '<tool_call>\n<function'
        s = self.advanced(text,fmt)
        self.assertEqual(s.propose(7), [])
        s.advance(self.enc('=fetch'))
        self.assertEqual(s.propose(7), self.advanced(text+'=fetch',fmt).propose(7))
        s.advance(self.enc('>\n</function>\n</tool_call> aside'))
        self.assertEqual(s.propose(7), [])
        s.advance(self.enc('<tool_call>'))
        self.assertEqual(s.propose(7), self.enc('\n<function'))
        s.advance(self.enc('<eos><tool_call>'))
        self.assertEqual(s.propose(7), [])
        rebuilt = self.advanced(text+'=list',fmt)
        self.assertEqual(rebuilt.propose(2), self.enc('_names>'))

    def test_malformed_and_unknown_fields_abstain(self):
        for tail in ('\n<function=unknown>\n', '\n<function=list_names>\n<parameter=unknown>\n'):
            self.assertEqual(self.advanced('<tool_call>'+tail).propose(7),[])
        self.assertIsNone(MtpToolFormat.from_tools(self.tok, []))
        self.assertIsNone(MtpToolFormat.from_tools(self.tok, self.tools+self.tools))
        self.assertIsNone(MtpToolFormat.from_tools(self.tok,[tool('bad\nname')]))
        bad=tool('f')
        bad['function']['parameters']['additionalProperties']=True
        self.assertIsNone(MtpToolFormat.from_tools(self.tok,[bad]))


@unittest.skipUnless(os.environ.get('SNOWLLM_TOOL_TEST_MODEL'), 'set SNOWLLM_TOOL_TEST_MODEL to a local Qwen tokenizer')
class RealTokenizerTest(unittest.TestCase):
    def test_rendered_calls_and_merged_boundaries(self):
        from snowllm.checkpoint.tokenizer import load_tokenizer
        tok,eos=load_tokenizer(Path(os.environ['SNOWLLM_TOOL_TEST_MODEL']))
        tools=[tool('list_files', ['file_path','rev'], ['file_path','rev']), tool('list_tasks')]
        fmt=MtpToolFormat.from_tools(tok,tools,eos)
        self.assertIsNotNone(fmt)
        messages=[{'role':'user','content':'go'},
                  {'role':'assistant','content':'','tool_calls':[
                      {'type':'function','function':{'name':'list_files','arguments':{'rev':'  ','file_path':'foo\n'}}},
                      {'type':'function','function':{'name':'list_tasks','arguments':{}}}]}]
        rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        rendered=rendered[rendered.index('<tool_call>'):]
        tokens=tok.encode(rendered,add_special_tokens=False)
        self.assertGreater(tokens.count(fmt.opening),1)
        s=fmt.new_state()
        hits=0
        for i,token in enumerate(tokens):
            draft=s.propose(7)
            self.assertEqual(draft,tokens[i:i+len(draft)],tok.decode(tokens[:i]))
            hits+=bool(draft)
            s.advance([token])
        self.assertGreater(hits,20)
        self.assertEqual(s.propose(7),[])
        s=fmt.new_state()
        s.advance([fmt.opening])
        expected=tok.encode('\n<function=list_files>\n',add_special_tokens=False)
        other=tok.encode('\n<function=list_tasks>\n',add_special_tokens=False)
        common=next(i for i,(a,b) in enumerate(zip(expected,other)) if a!=b)
        self.assertEqual(s.propose(100),expected[:common])


if __name__ == '__main__':
    unittest.main()
