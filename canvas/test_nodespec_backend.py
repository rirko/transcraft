"""NodeSpec 后端隔离测试：schema 镜像验证、与 nodespec.js 的共享规则一致性、
自由目标 + 声明式 imports 的候选真实执行（math/builtins/无参数方法/非 Tensor 预览/
缺模块报错）、HTTP 端点与项目隔离（保存/恢复/trace 运行/版本冲突）。

运行：.venv/Scripts/python.exe canvas/test_nodespec_backend.py
（需要 node 在 PATH 中做 JS 侧 dump；全部使用临时数据目录，不触碰真实项目。）
"""
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import nodespecs as N  # noqa: E402
from server import Handler, validate_literal, run_candidate  # noqa: E402


def safe_literal(source):
    """服务器语义下的 literal 校验：失败返回 {'error': …} 而不是抛异常。"""
    try:
        return validate_literal(source)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        return {'error': str(exc)}


PY_EXE = BASE_DIR.parent / '.venv' / 'Scripts' / 'python.exe'
if not PY_EXE.exists():
    PY_EXE = Path(sys.executable)

TEMPLATES = {
    'sqrt': {
        'schema': 1, 'id': 'sqrt', 'version': 1, 'label': '平方根 sqrt', 'description': '逐元素开平方', 'category': '自定义',
        'output': {'name': 'out', 'type': 'Tensor'},
        'inputs': [{'name': 'x', 'type': 'Tensor', 'optional': False, 'fallbackParam': None, 'help': '非负张量'}],
        'params': [],
        'call': {'kind': 'function', 'target': 'torch.sqrt',
                 'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]},
    },
    'sin': {
        'schema': 1, 'id': 'sin', 'version': 1, 'label': '正弦 sin', 'description': '逐元素正弦', 'category': '自定义',
        'output': {'name': 'out', 'type': 'Tensor'},
        'inputs': [{'name': 'x', 'type': 'Tensor', 'optional': False, 'fallbackParam': None, 'help': ''}],
        'params': [],
        'call': {'kind': 'function', 'target': 'torch.sin',
                 'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]},
    },
    'triu': {
        'schema': 1, 'id': 'triu', 'version': 1, 'label': '上三角 triu', 'description': 'diagonal 连线优先', 'category': '自定义',
        'output': {'name': 'out', 'type': 'Tensor'},
        'inputs': [{'name': 'input', 'type': 'Tensor', 'optional': False, 'fallbackParam': None, 'help': ''},
                   {'name': 'diagonal', 'type': 'int', 'optional': True, 'fallbackParam': 'diagonal', 'help': ''}],
        'params': [{'key': 'diagonal', 'label': 'diagonal', 'type': 'int', 'def': '0', 'help': ''}],
        'call': {'kind': 'function', 'target': 'torch.triu',
                 'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None},
                          {'source': 'input', 'index': 2, 'keyword': 'diagonal', 'fallbackParam': 'diagonal'}]},
    },
    'masked_fill': {
        'schema': 1, 'id': 'masked_fill', 'version': 1, 'label': '掩码填充', 'description': 'Tensor 方法示例', 'category': '自定义',
        'output': {'name': 'out', 'type': 'Tensor'},
        'inputs': [{'name': 'input', 'type': 'Tensor', 'optional': False, 'fallbackParam': None, 'help': ''},
                   {'name': 'mask', 'type': 'BoolTensor', 'optional': False, 'fallbackParam': None, 'help': ''},
                   {'name': 'value', 'type': 'number', 'optional': True, 'fallbackParam': 'value', 'help': ''}],
        'params': [{'key': 'value', 'label': 'value', 'type': 'number', 'def': '-inf', 'help': ''}],
        'call': {'kind': 'method', 'target': 'masked_fill',
                 'args': [{'source': 'input', 'index': 2, 'keyword': None, 'fallbackParam': None},
                          {'source': 'input', 'index': 3, 'keyword': 'value', 'fallbackParam': 'value'}]},
    },
}


def js_literal_stub(source):
    """与 test_nodespec_rules.cjs 的 literalOk 完全同规则的测试替身（仅用于两侧比对）。"""
    text = source.replace("'", '"')
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b', 'null', text)
    try:
        json.loads(text)
        return {'expression': source, 'type': 'stub'}
    except ValueError:
        return {'error': 'invalid literal'}


class SchemaMirrorTest(unittest.TestCase):
    def assert_rejected(self, raw, literal=js_literal_stub):
        with self.assertRaises(N.SpecError):
            N.validate_spec(raw, literal)

    def test_templates_valid(self):
        for spec in TEMPLATES.values():
            canonical = N.validate_spec(spec, js_literal_stub)
            self.assertEqual(canonical['id'], spec['id'])

    def test_render_matches_js(self):
        spec = N.validate_spec(TEMPLATES['sqrt'], js_literal_stub)
        self.assertEqual(N.render_call(spec, ['x']), 'torch.sqrt(x)')
        triu = N.validate_spec(TEMPLATES['triu'], js_literal_stub)
        self.assertEqual(N.render_call(triu, ['x', '2']), 'torch.triu(x, diagonal=2)')
        self.assertEqual(N.render_call(triu, ['x', None], {'diagonal': '1'}), 'torch.triu(x, diagonal=1)')
        mf = N.validate_spec(TEMPLATES['masked_fill'], js_literal_stub)
        self.assertEqual(N.render_call(mf, ['x', 'm', '-2.5']), 'x.masked_fill(m, value=-2.5)')
        self.assertEqual(N.render_call(mf, ['x', 'm', None]), 'x.masked_fill(m, value=float("-inf"))')

    def test_target_syntax_rules(self):
        """自由目标只做语法校验：合法点分标识符一律接受（无隐藏白名单），
        非法语法（通配/空段/空格/分号/括号/保留字段）一律拒绝。"""
        fn = lambda target: {'kind': 'function', 'target': target,
                             'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]}
        for target in ['math.sqrt', 'builtins.len', 'torch.load', 'os.system', 'subprocess.run',
                       'torch._C._nn.gelu', 'torch.utils.data', 'torch.nn.functional.softmax',
                       'eval', 'exec', '__import__', 'open', '_private.name', 'A.Uppercase.seg']:
            canonical = N.validate_spec({**TEMPLATES['sqrt'], 'call': fn(target)}, js_literal_stub)
            self.assertEqual(canonical['call']['target'], target)
        for target in ['torch.*', 'sqrt.', '.sqrt', 'torch..sqrt', 'a b', 'a;b', 'a\nb', 'eval(x',
                       'x) + y', '2bad', 'a-b', 'a.class.b', 'a.import.b', 'out=x', 'f()', 'a.b[0]',
                       'import', 'None', 'True', 42, None, '']:
            self.assert_rejected({**TEMPLATES['sqrt'], 'call': fn(target)})
        # method：单个方法名；无参数 item/tolist 允许零绑定（只靠接收者）；函数 kind 不允许零绑定
        for target in ['item', 'tolist', 'masked_fill', 'relu_', 'fill_', '__setattr__', 'numpy', 'cpu']:
            method = {'kind': 'method', 'target': target,
                      'args': [] if target in ('item', 'tolist')
                      else [{'source': 'input', 'index': 2, 'keyword': None, 'fallbackParam': None}]}
            spec = {**TEMPLATES['sqrt'], 'call': method}
            if method['args']:
                spec = {**spec, 'inputs': TEMPLATES['sqrt']['inputs'] +
                        [{'name': 'm', 'type': 'BoolTensor', 'optional': False, 'fallbackParam': None, 'help': ''}]}
            self.assertEqual(N.validate_spec(spec, js_literal_stub)['call']['target'], target)
        self.assertEqual(N.render_call(N.validate_spec(
            {**TEMPLATES['sqrt'], 'call': {'kind': 'method', 'target': 'item', 'args': []}}, js_literal_stub), ['x']), 'x.item()')
        self.assertEqual(N.render_call(N.validate_spec(
            {**TEMPLATES['sqrt'], 'call': {'kind': 'method', 'target': 'tolist', 'args': []}}, js_literal_stub), ['x']), 'x.tolist()')
        for target in ['a.b', 'class', 'for', 'item(', 'x y', '']:
            self.assert_rejected({**TEMPLATES['sqrt'], 'call': {'kind': 'method', 'target': target,
                                                                'args': [{'source': 'input', 'index': 2, 'keyword': None, 'fallbackParam': None}]}})
        self.assert_rejected({**TEMPLATES['sqrt'], 'call': {'kind': 'function', 'target': 'math.sqrt', 'args': []}})

    def test_bad_bindings_rejected(self):
        triu = TEMPLATES['triu']
        self.assert_rejected({**triu, 'call': {'kind': 'function', 'target': 'torch.triu',
                                               'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None},
                                                        {'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]}})
        self.assert_rejected({**triu, 'call': {'kind': 'function', 'target': 'torch.triu',
                                               'args': [{'source': 'input', 'index': 5, 'keyword': None, 'fallbackParam': None}]}})
        self.assert_rejected({**triu, 'call': {'kind': 'function', 'target': 'torch.triu',
                                               'args': [{'source': 'param', 'key': 'ghost', 'keyword': None, 'fallbackParam': None}]}})
        self.assert_rejected({**triu, 'call': {'kind': 'function', 'target': 'torch.triu',
                                               'args': [{'source': 'input', 'index': 1, 'keyword': 'for', 'fallbackParam': None}]}})

    def test_shared_rules_match_js_dump(self):
        """共享 JS/Python 规则一致性：目标语法/imports 判定、接受/拒绝、渲染逐字符一致。"""
        dump = subprocess.run(['node', str(BASE_DIR / 'test_nodespec_rules.cjs'), 'dump'],
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(dump.returncode, 0, dump.stderr)
        rules = json.loads(dump.stdout[dump.stdout.index('{'):])
        self.assertEqual(sorted(rules['functionTargets']), sorted(N.FUNCTION_TARGETS))
        self.assertEqual(sorted(rules['methodTargets']), sorted(N.METHOD_TARGETS))
        self.assertEqual(rules['paramTypes'], list(N.PARAM_TYPES))
        self.assertEqual(rules['inputTypeHints'], list(N.INPUT_TYPE_HINTS))
        for name, expected in rules['decisions']:
            fn_arg = [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]
            drafts = {
                'triu-template': TEMPLATES['triu'],
                # 自由目标：合法语法一律接受（真开放，无隐藏白名单），非法语法一律拒绝
                'free-target-math': {**TEMPLATES['sqrt'], 'call': {'kind': 'function', 'target': 'math.sqrt', 'args': fn_arg}},
                'free-target-torch-load': {**TEMPLATES['sqrt'], 'call': {'kind': 'function', 'target': 'torch.load', 'args': fn_arg}},
                'free-target-subprocess': {**TEMPLATES['sqrt'], 'call': {'kind': 'function', 'target': 'subprocess.run', 'args': fn_arg}},
                'wildcard-target': {**TEMPLATES['sqrt'], 'call': {'kind': 'function', 'target': 'torch.*', 'args': fn_arg}},
                'target-trailing-dot': {**TEMPLATES['sqrt'], 'call': {'kind': 'function', 'target': 'sqrt.', 'args': fn_arg}},
                'target-paren-injection': {**TEMPLATES['sqrt'], 'call': {'kind': 'function', 'target': 'eval(x', 'args': fn_arg}},
                'target-semicolon-injection': {**TEMPLATES['sqrt'], 'call': {'kind': 'function', 'target': 'os.system; import os', 'args': fn_arg}},
                'target-keyword-segment': {**TEMPLATES['sqrt'], 'call': {'kind': 'function', 'target': 'a.class.b', 'args': fn_arg}},
                # 方法：语法判定 + 无参数 item
                'method-inplace-allowed': {**TEMPLATES['sqrt'], 'call': {'kind': 'method', 'target': 'relu_', 'args': []}},
                'method-item-noargs-ok': {**TEMPLATES['sqrt'], 'call': {'kind': 'method', 'target': 'item', 'args': []}},
                'method-dotted-rejected': {**TEMPLATES['sqrt'], 'call': {'kind': 'method', 'target': 'a.b', 'args': []}},
                'method-keyword-rejected': {**TEMPLATES['sqrt'], 'call': {'kind': 'method', 'target': 'class', 'args': []}},
                # imports：合法声明 / 默认去重 / 别名冲突 / 注入
                'imports-math-ok': {**TEMPLATES['sqrt'], 'imports': [{'module': 'math', 'alias': None}]},
                'imports-default-dedupe': {**TEMPLATES['sqrt'], 'imports': [
                    {'module': 'torch', 'alias': None}, {'module': 'torch.nn', 'alias': 'nn'}, {'module': 'numpy', 'alias': 'np'}]},
                'imports-alias-conflict': {**TEMPLATES['sqrt'], 'imports': [
                    {'module': 'json', 'alias': 'np'}, {'module': 'numpy', 'alias': 'np'}]},
                'imports-torch-alias-rejected': {**TEMPLATES['sqrt'], 'imports': [{'module': 'numpy', 'alias': 'torch'}]},
                'imports-injection-rejected': {**TEMPLATES['sqrt'], 'imports': [{'module': 'os; import os', 'alias': None}]},
                'dup-input': {**TEMPLATES['triu'], 'call': {'kind': 'function', 'target': 'torch.triu',
                                                            'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None},
                                                                     {'source': 'input', 'index': 1, 'keyword': 'diagonal', 'fallbackParam': None}]}},
                'keyword-reserved': {**TEMPLATES['triu'],
                                     'params': [{'key': 'for', 'label': 'f', 'type': 'int', 'def': '0', 'help': ''}],
                                     'inputs': [TEMPLATES['triu']['inputs'][0],
                                                {**TEMPLATES['triu']['inputs'][1], 'fallbackParam': 'for'}],
                                     'call': {'kind': 'function', 'target': 'torch.triu',
                                              'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None},
                                                       {'source': 'input', 'index': 2, 'keyword': 'for', 'fallbackParam': 'for'}]}},
                'missing-binding': {**TEMPLATES['triu'], 'call': {'kind': 'function', 'target': 'torch.triu',
                                                                  'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]}},
            }
            if expected == 'ok':
                N.validate_spec(drafts[name], js_literal_stub)
            else:
                self.assert_rejected(drafts[name])
        for name, expected in rules['renderings']:
            if expected.startswith('ERROR:'):
                with self.assertRaises(N.SpecError, msg=name):
                    self._render(name)
                continue
            self.assertEqual(self._render(name), expected, name)

    def _render(self, name):
        if name == 'sqrt':
            return N.render_call(N.validate_spec(TEMPLATES['sqrt'], js_literal_stub), ['x'], None, js_literal_stub)
        if name == 'free-math':
            spec = {**TEMPLATES['sqrt'], 'call': {'kind': 'function', 'target': 'math.sqrt',
                                                  'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]}}
            return N.render_call(N.validate_spec(spec, js_literal_stub), ['x'], None, js_literal_stub)
        if name in ('method-item', 'method-tolist'):
            spec = {**TEMPLATES['sqrt'], 'call': {'kind': 'method', 'target': 'item' if name == 'method-item' else 'tolist',
                                                  'args': []}}
            return N.render_call(N.validate_spec(spec, js_literal_stub), ['x'], None, js_literal_stub)
        if name == 'triu-connected':
            return N.render_call(N.validate_spec(TEMPLATES['triu'], js_literal_stub), ['x', '2'], None, js_literal_stub)
        if name == 'triu-fallback':
            return N.render_call(N.validate_spec(TEMPLATES['triu'], js_literal_stub), ['x', None], {'diagonal': '1'}, js_literal_stub)
        if name in ('masked-fill', 'masked-fill-inf', 'number-bad'):
            spec = {**TEMPLATES['masked_fill']}
            if name == 'number-bad':
                spec['params'] = [{'key': 'value', 'label': 'value', 'type': 'number', 'def': 'abc', 'help': ''}]
            canonical = N.validate_spec(spec, js_literal_stub)
            if name == 'masked-fill':
                return N.render_call(canonical, ['x', 'm', '0.5'], None, js_literal_stub)
            if name == 'masked-fill-inf':
                return N.render_call(canonical, ['x', 'm', None], None, js_literal_stub)
        if name == 'greet-params':
            greet = {**TEMPLATES['sqrt'], 'id': 'greet', 'label': 'greet',
                     'params': [
                         {'key': 'sep', 'label': 'sep', 'type': 'string', 'def': '他说"hi"', 'help': ''},
                         {'key': 'flag', 'label': 'flag', 'type': 'bool', 'def': True, 'help': ''},
                         {'key': 'mode', 'label': 'mode', 'type': 'select', 'def': 'fast', 'options': ['fast', 'slow'], 'help': ''},
                         {'key': 'bias', 'label': 'bias', 'type': 'literal', 'def': '[1, 2.5, "x"]', 'help': ''},
                     ],
                     'call': {'kind': 'function', 'target': 'torch.sqrt', 'args': [
                         {'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None},
                         {'source': 'param', 'key': 'sep', 'keyword': None, 'fallbackParam': None},
                         {'source': 'param', 'key': 'flag', 'keyword': 'flag', 'fallbackParam': None},
                         {'source': 'param', 'key': 'mode', 'keyword': 'mode', 'fallbackParam': None},
                         {'source': 'param', 'key': 'bias', 'keyword': 'bias', 'fallbackParam': None}]}}
            return N.render_call(N.validate_spec(greet, js_literal_stub), ['x'], None, js_literal_stub)
        raise AssertionError('unknown rendering case ' + name)

    def test_imports_validation_matrix(self):
        """声明式 imports：合法声明 / 去重 / 默认 torch+nn 去重 / 别名冲突 / 注入拒绝。"""
        def imports(imports):
            return N.validate_spec({**TEMPLATES['sqrt'], 'imports': imports}, js_literal_stub)['imports']
        self.assertEqual(imports([{'module': 'math'}, {'module': 'numpy', 'alias': 'np'}, {'module': 'collections.abc'}]),
                         [{'module': 'math', 'alias': None}, {'module': 'numpy', 'alias': 'np'}, {'module': 'collections.abc', 'alias': None}])
        self.assertEqual(imports([{'module': 'math'}, {'module': 'math'}]), [{'module': 'math', 'alias': None}])
        # 默认 torch / torch.nn as nn 已存在 → 丢弃；裸 torch.nn 绑定名也是 torch → 去重
        self.assertEqual(imports([{'module': 'torch'}, {'module': 'torch.nn', 'alias': 'nn'},
                                  {'module': 'torch.nn'}, {'module': 'math'}]),
                         [{'module': 'math', 'alias': None}])
        # 旧规范无 imports 字段 → 空数组（兼容）
        self.assertEqual(N.validate_spec(TEMPLATES['sqrt'], js_literal_stub)['imports'], [])
        for bad in ([{'module': 'numpy', 'alias': 'torch'}], [{'module': 'numpy', 'alias': 'nn'}],
                    [{'module': 'torch.nn', 'alias': 'torch'}], [{'module': 'math', 'alias': 'nn'}]):
            self.assert_rejected({**TEMPLATES['sqrt'], 'imports': bad})
        self.assert_rejected({**TEMPLATES['sqrt'], 'imports': [{'module': 'json', 'alias': 'np'}, {'module': 'numpy', 'alias': 'np'}]})
        self.assert_rejected({**TEMPLATES['sqrt'], 'imports': [{'module': 'math'}, {'module': 'math.abc'}]})
        for bad in ([{'module': 'os; import os'}], [{'module': 'x y'}], [{'module': 'a', 'alias': 'for'}],
                    [{'module': 'a', 'alias': '1x'}], [{'module': 'a', 'extra': 1}], 'nope'):
            self.assert_rejected({**TEMPLATES['sqrt'], 'imports': bad})
        self.assert_rejected({**TEMPLATES['sqrt'], 'imports': [{'module': str(i)} for i in range(N.MAX_IMPORTS + 1)]})

    def test_validate_graph_specs(self):
        good = {'version': 3, 'seed': 1, 'nodeSpecs': [TEMPLATES['triu']],
                'main': {'1': {'data': {'type': 'custom.triu@1', 'name': 'u', 'diagonal': '1'}}},
                'functions': []}
        specs = N.validate_graph_specs(good)
        self.assertEqual(len(specs), 1)
        missing = {**good, 'main': {'1': {'data': {'type': 'custom.ghost@1', 'name': 'g'}}}}
        with self.assertRaises(N.SpecError):
            N.validate_graph_specs(missing)
        conflict = {**good, 'nodeSpecs': [TEMPLATES['triu'], {**TEMPLATES['triu'], 'label': '不同内容'}]}
        with self.assertRaises(N.SpecError):
            N.validate_graph_specs(conflict)
        duplicate_same = {**good, 'nodeSpecs': [TEMPLATES['triu'], TEMPLATES['triu']]}
        self.assertEqual(len(N.validate_graph_specs(duplicate_same)), 1)
        legacy_v2 = {'version': 2, 'seed': 1, 'main': {}, 'functions': []}
        self.assertEqual(N.validate_graph_specs(legacy_v2), [])


class CandidateExecutionTest(unittest.TestCase):
    """四个模板的后端生成代码用真实 PyTorch 执行。"""

    def try_run(self, spec, samples):
        canonical = N.validate_spec(spec, validate_literal)
        checked = N.validate_samples(samples, canonical)
        code = N.build_try_code(canonical, checked, safe_literal)
        result = run_candidate(code)
        self.assertTrue(result['ok'], result)
        preview = json.loads(result['stdout'].strip().splitlines()[-1])
        return preview, code

    def test_sqrt_template(self):
        preview, code = self.try_run(TEMPLATES['sqrt'], {'1': {'kind': 'tensor', 'shape': [2, 2], 'dtype': 'float32', 'fill': 'constant', 'value': '4'}})
        self.assertEqual(preview['shape'], [2, 2])
        self.assertEqual(preview['dtype'], 'torch.float32')
        self.assertEqual(preview['values'], [2.0, 2.0, 2.0, 2.0])
        self.assertIn('out = torch.sqrt(in1)', code)

    def test_sin_template(self):
        preview, code = self.try_run(TEMPLATES['sin'], {'1': {'kind': 'tensor', 'shape': [3], 'dtype': 'float32', 'fill': 'constant', 'value': '0'}})
        self.assertEqual(preview['shape'], [3])
        self.assertEqual(preview['values'], [0.0, 0.0, 0.0])
        self.assertIn('in1 = torch.full((3,), 0, dtype=torch.float32)', code)
        self.assertIn('out = torch.sin(in1)', code)

    def test_triu_template_connected_and_fallback(self):
        samples = {'1': {'kind': 'tensor', 'shape': [3, 3], 'dtype': 'int64', 'fill': 'constant', 'value': '1'}}
        preview, code = self.try_run(TEMPLATES['triu'], samples)
        self.assertEqual(code.count('out = torch.triu(in1, diagonal=0)'), 1)
        self.assertEqual(preview['values'], [1, 1, 1, 0, 1, 1, 0, 0, 1])
        spec = {**TEMPLATES['triu'], 'params': [{**TEMPLATES['triu']['params'][0], 'def': '1'}]}
        preview, code = self.try_run(spec, samples)
        self.assertEqual(code.count('out = torch.triu(in1, diagonal=1)'), 1)
        self.assertEqual(preview['values'], [0, 1, 1, 0, 0, 1, 0, 0, 0])

    def test_masked_fill_template_with_literal_input(self):
        samples = {'1': {'kind': 'tensor', 'shape': [2, 2], 'dtype': 'float32', 'fill': 'constant', 'value': '5'},
                   '2': {'kind': 'tensor', 'shape': [2, 2], 'dtype': 'bool', 'fill': 'constant', 'value': 'True'},
                   '3': {'kind': 'literal', 'source': '-0.5'}}
        preview, code = self.try_run(TEMPLATES['masked_fill'], samples)
        self.assertIn('in3 = -0.5', code)
        self.assertIn('out = in1.masked_fill(in2, value=in3)', code)
        self.assertEqual(preview['values'], [-0.5, -0.5, -0.5, -0.5])
        preview, code = self.try_run(TEMPLATES['masked_fill'],
                                     {'1': samples['1'], '2': samples['2']})
        self.assertIn('value=float("-inf")', code)
        self.assertTrue(all(v == float('-inf') for v in preview['values']))

    def test_bad_literal_sample_rejected(self):
        canonical = N.validate_spec(TEMPLATES['sqrt'], validate_literal)
        with self.assertRaises(N.SpecError):
            N.build_try_code(canonical, N.validate_samples({'1': {'kind': 'literal', 'source': '1+1'}}, canonical), safe_literal)

    def test_optional_omitted_requires_optional_input(self):
        canonical = N.validate_spec(TEMPLATES['sqrt'], validate_literal)
        with self.assertRaises(N.SpecError):
            N.validate_samples({}, canonical)


class FreeTargetCandidateTest(unittest.TestCase):
    """自由目标 + imports 的候选试运行：真实执行与有界非 Tensor 预览。

    math.sqrt / builtins.len / 无参数 item、tolist / dict 返回的 mapping 预览 /
    tuple 保持单输出 / 缺依赖模块的明确报错（不自动安装）。"""

    def try_run(self, spec, samples):
        canonical = N.validate_spec(spec, validate_literal)
        checked = N.validate_samples(samples, canonical)
        code = N.build_try_code(canonical, checked, safe_literal)
        result = run_candidate(code)
        self.assertTrue(result['ok'], result)
        preview = json.loads(result['stdout'].strip().splitlines()[-1])
        return preview, code, result

    def test_math_sqrt_with_imports(self):
        spec = {**TEMPLATES['sqrt'], 'id': 'math_sqrt', 'call': {'kind': 'function', 'target': 'math.sqrt',
                                                                'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]},
                'imports': [{'module': 'math', 'alias': None}]}
        preview, code, _ = self.try_run(spec, {'1': {'kind': 'literal', 'source': '4'}})
        self.assertEqual(preview['kind'], 'value')
        self.assertEqual(preview['type'], 'float')
        self.assertEqual(preview['repr'], '2.0')
        self.assertEqual(code.count('import math'), 1, 'imports 只生成一次')
        self.assertIn('out = math.sqrt(in1)', code)

    def test_builtins_len_unicode_string(self):
        # builtins 也是普通模块：目标 builtins.len 必须显式声明 import builtins（与正式生成一致）。
        spec = {**TEMPLATES['sqrt'], 'id': 'str_len', 'imports': [{'module': 'builtins', 'alias': None}],
                'inputs': [{'name': 'text', 'type': 'string', 'optional': False, 'fallbackParam': None, 'help': ''}],
                'call': {'kind': 'function', 'target': 'builtins.len',
                         'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]}}
        preview, code, _ = self.try_run(spec, {'1': {'kind': 'literal', 'source': "'知之'"}})
        self.assertEqual(preview['kind'], 'value')
        self.assertEqual(preview['repr'], '2')
        self.assertIn("in1 = '知之'", code)
        self.assertEqual(code.count('import builtins'), 1)
        # 不声明 imports → 运行期 NameError，错误原样展示（不做隐藏魔法注入）
        bare = {**spec, 'imports': []}
        canonical = N.validate_spec(bare, validate_literal)
        checked = N.validate_samples({'1': {'kind': 'literal', 'source': "'知之'"}}, canonical)
        result = run_candidate(N.build_try_code(canonical, checked, safe_literal))
        self.assertFalse(result['ok'])
        self.assertIn("name 'builtins' is not defined", result['error'])

    def test_tensor_item_noargs_value_preview(self):
        spec = {**TEMPLATES['sqrt'], 'id': 'pick', 'call': {'kind': 'method', 'target': 'item', 'args': []}}
        preview, code, _ = self.try_run(spec, {'1': {'kind': 'tensor', 'shape': [1, 1], 'dtype': 'float32', 'fill': 'constant', 'value': '4'}})
        self.assertEqual(preview['kind'], 'value')
        self.assertEqual(preview['type'], 'float')
        self.assertEqual(preview['repr'], '4.0')
        self.assertIn('out = in1.item()', code)

    def test_tensor_tolist_sequence_preview(self):
        spec = {**TEMPLATES['sqrt'], 'id': 'rows', 'call': {'kind': 'method', 'target': 'tolist', 'args': []}}
        preview, code, _ = self.try_run(spec, {'1': {'kind': 'tensor', 'shape': [2], 'dtype': 'int64', 'fill': 'constant', 'value': '7'}})
        self.assertEqual(preview['kind'], 'sequence')
        self.assertEqual(preview['type'], 'list')
        self.assertEqual(preview['count'], 2)
        self.assertEqual([item['repr'] for item in preview['items']], ['7', '7'])

    def test_dict_method_mapping_preview(self):
        spec = {**TEMPLATES['sqrt'], 'id': 'copydict',
                'inputs': [{'name': 'src', 'type': 'dict', 'optional': False, 'fallbackParam': None, 'help': ''}],
                'call': {'kind': 'method', 'target': 'copy', 'args': []}}
        preview, code, _ = self.try_run(spec, {'1': {'kind': 'literal', 'source': "{'a': 1, 'b': '知'}"}})
        self.assertEqual(preview['kind'], 'mapping')
        self.assertEqual(preview['type'], 'dict')
        self.assertEqual(preview['count'], 2)
        self.assertEqual([key for key, _ in preview['entries']], ["'a'", "'b'"])
        self.assertEqual(preview['entries'][0][1]['repr'], '1')
        self.assertEqual(preview['entries'][1][1]['repr'], "'知'")

    def test_tuple_return_is_single_output(self):
        spec = {**TEMPLATES['sqrt'], 'id': 'frexp', 'imports': [{'module': 'math', 'alias': None}],
                'call': {'kind': 'function', 'target': 'math.frexp',
                         'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]}}
        preview, code, _ = self.try_run(spec, {'1': {'kind': 'literal', 'source': '4'}})
        self.assertEqual(preview['kind'], 'sequence')
        self.assertEqual(preview['type'], 'tuple')
        self.assertEqual(preview['count'], 2, 'tuple 作为一个对象返回，不拆成多输出')

    def test_missing_module_clear_error_no_autostall(self):
        spec = {**TEMPLATES['sqrt'], 'id': 'needs_missing', 'imports': [{'module': 'canvas_no_such_module_xyz'}],
                'call': {'kind': 'function', 'target': 'canvas_no_such_module_xyz.func',
                         'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]}}
        canonical = N.validate_spec(spec, validate_literal)
        checked = N.validate_samples({'1': {'kind': 'literal', 'source': '1'}}, canonical)
        code = N.build_try_code(canonical, checked, safe_literal)
        result = run_candidate(code)
        self.assertFalse(result['ok'])
        self.assertIn("No module named 'canvas_no_such_module_xyz'", result['error'], result['error'])
        self.assertIn('不会自动安装', result['error'])

    def test_injection_targets_rejected_before_execution(self):
        for target in ['os.system; import os', 'eval(input())', 'x\ny', 'torch.*']:
            with self.assertRaises(N.SpecError):
                N.validate_spec({**TEMPLATES['sqrt'], 'call': {'kind': 'function', 'target': target,
                                                               'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]}},
                                validate_literal)


class NodeSpecHttpTest(unittest.TestCase):
    """隔离 HTTP：候选试运行端点 + 带 nodeSpecs 的项目保存/运行 + 项目文件不被触碰。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        import server
        self.server_module = server
        self._original_data_dir = server.DATA_DIR
        server.DATA_DIR = Path(self.temp.name)
        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = 'http://127.0.0.1:' + str(self.httpd.server_port)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()
        self.server_module.DATA_DIR = self._original_data_dir
        self.temp.cleanup()

    def request(self, route, data=None):
        request = Request(self.base + route, data=json.dumps(data).encode() if data is not None else None,
                          headers={'Content-Type': 'application/json'})
        with urlopen(request) as response:
            return json.load(response)

    def test_try_endpoint_success_and_rejection(self):
        result = self.request('/api/nodespec/try', {
            'spec': TEMPLATES['sqrt'],
            'samples': {'1': {'kind': 'tensor', 'shape': [2, 2], 'dtype': 'float32', 'fill': 'constant', 'value': '9'}}})
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['preview']['values'], [3.0, 3.0, 3.0, 3.0])
        self.assertIn('out = torch.sqrt(in1)', result['code'])
        bad = {**TEMPLATES['sqrt'], 'call': {'kind': 'function', 'target': 'eval',
                                            'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]}}
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/nodespec/try', {'spec': bad, 'samples': {}})
        self.assertEqual(caught.exception.code, 422)
        dup = {**TEMPLATES['triu'], 'call': {'kind': 'function', 'target': 'torch.triu',
                                            'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None},
                                                     {'source': 'input', 'index': 1, 'keyword': 'diagonal', 'fallbackParam': None}]}}
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/nodespec/try', {'spec': dup, 'samples': {'1': {'kind': 'tensor', 'shape': [2, 2], 'dtype': 'float32', 'fill': 'constant', 'value': '1'}}})
        self.assertEqual(caught.exception.code, 422)
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/nodespec/try', {'spec': TEMPLATES['sqrt'], 'samples': {}})
        self.assertEqual(caught.exception.code, 422)

    def test_try_endpoint_free_target_and_missing_module(self):
        """自由目标端到端：math.sqrt 带 imports 成功；缺依赖模块返回明确错误而非 500。"""
        math_spec = {**TEMPLATES['sqrt'], 'id': 'math_sqrt', 'imports': [{'module': 'math', 'alias': None}],
                     'call': {'kind': 'function', 'target': 'math.sqrt',
                              'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]}}
        result = self.request('/api/nodespec/try', {
            'spec': math_spec, 'samples': {'1': {'kind': 'literal', 'source': '4'}}})
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['preview']['kind'], 'value')
        self.assertEqual(result['preview']['repr'], '2.0')
        self.assertEqual(result['code'].count('import math'), 1)
        missing = {**math_spec, 'id': 'needs_missing', 'imports': [{'module': 'canvas_no_such_module_xyz'}],
                   'call': {'kind': 'function', 'target': 'canvas_no_such_module_xyz.func',
                            'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]}}
        result = self.request('/api/nodespec/try', {
            'spec': missing, 'samples': {'1': {'kind': 'literal', 'source': '1'}}})
        self.assertFalse(result['ok'])
        self.assertIn("No module named 'canvas_no_such_module_xyz'", result['error'])
        # 注入式目标在规范校验就被拒绝（422），不会执行
        bad = {**TEMPLATES['sqrt'], 'call': {'kind': 'function', 'target': 'os.system; import os',
                                             'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]}}
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/nodespec/try', {'spec': bad, 'samples': {'1': {'kind': 'literal', 'source': '1'}}})
        self.assertEqual(caught.exception.code, 422)

    def test_project_save_restore_trace_with_imports_spec(self):
        """带 imports 的规范：保存 / 版本推进 / 恢复 / trace 运行全链路。"""
        math_spec = {**TEMPLATES['sqrt'], 'id': 'math_sqrt', 'imports': [{'module': 'math', 'alias': None}],
                     'call': {'kind': 'function', 'target': 'math.sqrt',
                              'args': [{'source': 'input', 'index': 1, 'keyword': None, 'fallbackParam': None}]}}
        project = self.request('/api/projects/create', {'name': 'ImportSpecProject'})['project']
        graph = {'version': 3, 'seed': 5, 'nodeSpecs': [math_spec],
                 'main': {'1': {'data': {'type': 'custom.math_sqrt@1', 'name': 'root', 'x': None}}},
                 'functions': []}
        code = ('import torch\nimport torch.nn as nn\nimport math\n\n'
                'if __name__ == "__main__":\n'
                '    base = math.sqrt(4)\n'
                '    print(base)\n')
        trace_plan = [{'scope': 'main', 'node': '1', 'lines': [6]}]
        saved = self.request('/api/save', {'project_id': project['id'], 'version': project['version'],
                                           'graph': graph, 'code': code, 'trace_plan': trace_plan})['project']
        self.assertEqual(saved['graph']['nodeSpecs'][0]['imports'], [{'module': 'math', 'alias': None}])
        run = self.request('/api/run', {'project_id': saved['id'], 'version': saved['version'],
                                        'revision': saved['revision']})
        self.assertIn('2.0', run['stdout'], run['stderr'])
        self.assertTrue(run['trace']['records'], run['trace'])
        self.assertEqual(run['trace']['records'][0]['node'], '1')
        self.assertEqual(run['trace']['records'][0]['outputs'][0]['repr'], '2.0')
        # 恢复（/api/project）必须原样带回 imports
        restored = self.request('/api/project?id=' + saved['id'])['project']
        self.assertEqual(restored['graph']['nodeSpecs'][0]['imports'], [{'module': 'math', 'alias': None}])
        # 修改同 id+version 的 imports 属于不同内容：项目内出现两个版本冲突定义 → 拒绝保存
        before = (Path(self.temp.name) / 'projects' / saved['id'] / 'project.json').read_bytes()
        conflict = {**graph, 'nodeSpecs': [math_spec, {**math_spec, 'imports': []}]}
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/save', {'project_id': saved['id'], 'version': saved['version'],
                                       'graph': conflict, 'code': code})
        self.assertEqual(caught.exception.code, 422)
        self.assertEqual((Path(self.temp.name) / 'projects' / saved['id'] / 'project.json').read_bytes(), before)

    def test_try_run_never_touches_project_files(self):
        project = self.request('/api/projects/create', {'name': 'NodeSpecHost'})['project']
        graph = {'version': 3, 'seed': 7, 'main': {}, 'functions': [], 'nodeSpecs': [TEMPLATES['triu']]}
        project = self.request('/api/save', {'project_id': project['id'], 'version': project['version'],
                                             'graph': graph, 'code': 'print("host")'})['project']
        folder = Path(self.temp.name) / 'projects' / project['id']

        def snapshot():
            return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(folder.rglob('*')) if p.is_file() and p.parent.name != 'runs'}
        before = snapshot()
        for spec, samples in [
            (TEMPLATES['sqrt'], {'1': {'kind': 'tensor', 'shape': [2], 'dtype': 'float32', 'fill': 'constant', 'value': '4'}}),
            (TEMPLATES['masked_fill'], {'1': {'kind': 'tensor', 'shape': [2], 'dtype': 'float32', 'fill': 'constant', 'value': '1'},
                                        '2': {'kind': 'tensor', 'shape': [2], 'dtype': 'bool', 'fill': 'constant', 'value': 'True'}}),
        ]:
            result = self.request('/api/nodespec/try', {'spec': spec, 'samples': samples})
            self.assertTrue(result['ok'], result)
        self.assertEqual(snapshot(), before, '候选试运行不得改动任何项目文件')

    def test_project_save_run_with_custom_node_and_reference_check(self):
        result = self.request('/api/projects/create', {'name': 'CustomNodeProject'})
        project = result['project']
        graph = {'version': 3, 'seed': 3,
                 'nodeSpecs': [TEMPLATES['triu'], TEMPLATES['sqrt']],
                 'main': {'1': {'data': {'type': 'custom.triu@1', 'name': 'upper', 'diagonal': '1'}},
                          '2': {'data': {'type': 'tensor', 'name': 'base', 'values': '[1, 2, 3, 4]'}}},
                 'functions': []}
        code = ('import torch\nimport torch.nn as nn\n\ntorch.manual_seed(3)\n\n'
                'if __name__ == "__main__":\n'
                '    base = torch.tensor([[1, 2], [3, 4]])\n'
                '    upper = torch.triu(base, diagonal=1)\n'
                '    print(upper.tolist())\n')
        saved = self.request('/api/save', {'project_id': project['id'], 'version': project['version'],
                                           'graph': graph, 'code': code})['project']
        run = self.request('/api/run', {'project_id': saved['id'], 'version': saved['version'],
                                        'revision': saved['revision']})
        self.assertIn('[[0, 2], [0, 0]]', run['stdout'], run['stderr'])
        # 引用了未发布的定义 → 拒绝保存，且原项目文件保持不变
        before = (Path(self.temp.name) / 'projects' / saved['id'] / 'project.json').read_bytes()
        broken = {**graph, 'main': {'1': {'data': {'type': 'custom.ghost@1', 'name': 'g'}}}}
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/save', {'project_id': saved['id'], 'version': saved['version'],
                                       'graph': broken, 'code': code})
        self.assertEqual(caught.exception.code, 422)
        self.assertEqual((Path(self.temp.name) / 'projects' / saved['id'] / 'project.json').read_bytes(), before)
        # save-as 携带 nodeSpecs 的完整快照
        copy = self.request('/api/projects/save-as', {'project_id': saved['id'], 'name': 'CustomCopy',
                                                      'graph': graph})['project']
        self.assertEqual(copy['graph']['nodeSpecs'][0]['id'], 'triu')
        # 旧版 v3 图（无 nodeSpecs 字段）依然有效
        legacy = self.request('/api/projects/create', {'name': 'LegacyShape'})['project']
        saved_legacy = self.request('/api/save', {'project_id': legacy['id'], 'version': legacy['version'],
                                                  'graph': {'version': 3, 'seed': 1, 'main': {}, 'functions': []},
                                                  'code': 'print("legacy")'})['project']
        self.assertTrue(saved_legacy['revision'])


if __name__ == '__main__':
    unittest.main(verbosity=1)
