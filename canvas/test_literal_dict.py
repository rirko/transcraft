"""Literal 字典支持隔离回归测试：整数键 / 中文字符串键 / 嵌套 dict / 恶意表达式拒绝 /
缺键 KeyError 不静默 / 存档保存运行 / 运行查看器 dict 有界预览 / NodeSpec literal 参数。

只使用临时数据目录隔离，绝不触碰真实 projects/graph/generated。
运行：.venv/Scripts/python.exe canvas/test_literal_dict.py
"""
import io
import json
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

import runtime_trace  # noqa: E402
import server  # noqa: E402
import nodespecs  # noqa: E402
from server import Handler, validate_literal  # noqa: E402

# 用户目标写法：id/token 双向映射
ID_TO_TOKEN = "{0: '知', 1: '之', 2: '为', 3: '，', 4: '不', 5: '是', 6: '也'}"
TOKEN_TO_ID = "{'知': 0, '之': 1}"


def safe_literal(source):
    """服务器 literal_checked 的同款契约：失败返回 {'error': …} 而不是抛异常。"""
    try:
        return validate_literal(source)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        return {'error': str(exc)}


class ValidateLiteralDictTest(unittest.TestCase):
    def test_integer_and_chinese_keys_keep_types(self):
        result = validate_literal(ID_TO_TOKEN)
        self.assertEqual(result['type'], 'dict')
        # 生成 repr 的整数键必须保持 0: 形式，不能被 JSON 字符串化成 "0":
        self.assertIn('0:', result['expression'])
        self.assertNotIn('"0"', result['expression'])
        self.assertNotIn("'0'", result['expression'])
        self.assertIn("'知'", result['expression'])
        self.assertEqual(eval(result['expression']),   # 表达式由后端 repr 生成
                         {0: '知', 1: '之', 2: '为', 3: '，', 4: '不', 5: '是', 6: '也'})
        token = validate_literal(TOKEN_TO_ID)
        self.assertEqual(token['type'], 'dict')
        self.assertEqual(eval(token['expression']), {'知': 0, '之': 1})

    def test_empty_dict_and_tuple_keys(self):
        self.assertEqual(validate_literal('{}'), {'expression': '{}', 'type': 'dict'})
        result = validate_literal('{(0, 1): "知", ((2,),): 3}')
        self.assertEqual(result['type'], 'dict')
        self.assertEqual(eval(result['expression']), {(0, 1): '知', ((2,),): 3})

    def test_nested_dict_with_lists_tuples(self):
        source = '{0: {"a": [1, (2, 3)]}, 1: None, 2: True, 3: 1+2j}'
        result = validate_literal(source)
        self.assertEqual(result['type'], 'dict')
        self.assertEqual(eval(result['expression']), {0: {'a': [1, (2, 3)]}, 1: None, 2: True, 3: 1 + 2j})

    def test_malicious_and_unsupported_rejected(self):
        cases = [
            ('{1, 2}', '集合'),                     # set 字面量
            ('{k: 1 for k in "ab"}', ''),           # 字典推导式
            ('[x for x in []]', ''),                # 列表推导式
            ('dict(a=1)', ''),                      # 函数调用
            ('eval("1")', ''),                      # eval 调用
            ('__import__("os")', ''),               # 导入调用
            ('{**{1: 2}}', ''),                     # 字典展开
            ('{[1]: "x"}', '不可哈希'),              # 列表键
            ('{{1}: 2}', '不可哈希'),                # 集合键
            ('{1: [1, 2, 3][0]}', ''),              # 值里的下标表达式
            ("{1e999: 'x'}", 'Non-finite'),         # 非有限键
            ('{0: 1e999}', 'Non-finite'),           # 非有限值
            ("{0: float('nan')}", ''),              # 调用构造非有限
            ('1 + 2', ''),                          # 表达式
            ('{', ''),                              # 括号不匹配
        ]
        for source, fragment in cases:
            with self.assertRaises((ValueError, SyntaxError), msg=source) as caught:
                validate_literal(source)
            if fragment:
                self.assertIn(fragment, str(caught.exception), source)

    def test_complexity_budgets(self):
        # 深度：20 层容器仍可用（与旧行为一致），21 层拒绝
        self.assertEqual(validate_literal('[' * 20 + '1' + ']' * 20)['type'], 'list')
        with self.assertRaises(ValueError) as deep:
            validate_literal('[' * 21 + '1' + ']' * 21)
        self.assertIn('嵌套超过', str(deep.exception))
        with self.assertRaises(ValueError) as deep_dict:
            validate_literal('{' + '"k": {' * 20 + '1: 2' + '}' * 20 + '}')
        self.assertIn('嵌套超过', str(deep_dict.exception))
        ok_dict = '{' + '"k": {' * 19 + '1: 2' + '}' * 19 + '}'
        self.assertEqual(validate_literal(ok_dict)['type'], 'dict')
        # 条目预算：512 以内可用，超出拒绝（错误中文可理解）
        ok_entries = '{' + ','.join(f'{i}: {i}' for i in range(512)) + '}'
        self.assertEqual(validate_literal(ok_entries)['type'], 'dict')
        big_entries = '{' + ','.join(f'{i}: {i}' for i in range(513)) + '}'
        with self.assertRaises(ValueError) as budget:
            validate_literal(big_entries)
        self.assertIn('512', str(budget.exception))


class NodeSpecLiteralParamTest(unittest.TestCase):
    """NodeSpec literal 参数镜像：同一 validate_literal 契约，dict 候选可用。"""

    def test_render_param_value_with_dict(self):
        param = {'key': 'mapping', 'label': '映射', 'type': 'literal', 'def': ID_TO_TOKEN, 'help': ''}
        rendered = nodespecs.render_param_value(param, None, validate_literal)
        self.assertEqual(rendered, repr(eval(ID_TO_TOKEN)))
        self.assertIn('0:', rendered)
        with self.assertRaises(nodespecs.SpecError) as rejected:
            nodespecs.render_param_value({**param, 'def': '{1, 2}'}, None, safe_literal)
        self.assertIn('集合', str(rejected.exception))

    def test_check_default_accepts_dict_def(self):
        spec = {
            'schema': 1, 'id': 'dictlookup', 'version': 1, 'label': '字典查找', 'description': '',
            'category': '自定义',
            'output': {'name': 'out', 'type': 'Any'},
            'inputs': [],
            'params': [{'key': 'mapping', 'label': '映射', 'type': 'literal', 'def': TOKEN_TO_ID, 'help': ''}],
            'call': {'kind': 'function', 'target': 'torch.zeros_like',
                     'args': [{'source': 'param', 'key': 'mapping', 'keyword': None, 'fallbackParam': None}]},
        }
        canonical = nodespecs.validate_spec(spec, safe_literal)
        self.assertEqual(canonical['params'][0]['def'], TOKEN_TO_ID)
        # 恶意默认值在 schema 校验阶段就被拒绝
        with self.assertRaises(nodespecs.SpecError):
            nodespecs.validate_spec({**spec, 'params': [{**spec['params'][0], 'def': '{1, 2}'}]}, safe_literal)


class DictArchiveHttpTest(unittest.TestCase):
    """隔离 HTTP：/api/literal + 含字典字面量的图保存与真实运行（缺键 KeyError 不静默）。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self._previous_data_dir = server.DATA_DIR
        server.DATA_DIR = Path(self.temp.name)
        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = 'http://127.0.0.1:' + str(self.httpd.server_port)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()
        server.DATA_DIR = self._previous_data_dir
        self.temp.cleanup()

    def request(self, route, data=None):
        request = Request(self.base + route, data=json.dumps(data).encode() if data is not None else None,
                          headers={'Content-Type': 'application/json'})
        with urlopen(request) as response:
            return json.load(response)

    def default_project(self):
        listing = self.request('/api/projects')
        return self.request('/api/project?id=' + listing['default_id'])['project']

    def test_literal_endpoint_returns_dict(self):
        result = self.request('/api/literal', {'source': ID_TO_TOKEN})
        self.assertTrue(result['ok'])
        self.assertEqual(result['type'], 'dict')
        self.assertIn('0:', result['expression'])

    def test_dict_literal_save_run_and_missing_key_keyerror(self):
        project = self.default_project()
        graph = {'version': 3, 'seed': 1234,
                 'main': {'1': {'data': {'type': 'literal', 'name': 'id_to_token', 'value': ID_TO_TOKEN}},
                          '2': {'data': {'type': 'literal', 'name': 'token_to_id', 'value': TOKEN_TO_ID}}},
                 'functions': []}
        code = (f'id_to_token = {validate_literal(ID_TO_TOKEN)["expression"]}\n'
                f'token_to_id = {validate_literal(TOKEN_TO_ID)["expression"]}\n'
                'print(id_to_token[0], token_to_id["知"])\n')
        saved = self.request('/api/save', {'project_id': project['id'], 'version': project['version'],
                                           'graph': graph, 'code': code})['project']
        run = self.request('/api/run', {'project_id': saved['id'], 'version': saved['version'],
                                        'revision': saved['revision']})
        self.assertEqual(run['returncode'], 0, run['stderr'])
        self.assertIn('知 0', run['stdout'])
        # 存档镜像里字典字面量原样保留（graph JSON 不 stringify 整数键）
        archived = json.loads((Path(self.temp.name) / 'projects' / saved['id'] / 'project.json').read_text(encoding='utf-8'))
        self.assertEqual(archived['graph']['main']['1']['data']['value'], ID_TO_TOKEN)
        # 缺键：真实执行抛 KeyError，不静默回退 None
        bad_code = code + 'print(id_to_token[999])\n'
        bumped = self.request('/api/save', {'project_id': saved['id'], 'version': saved['version'],
                                            'graph': graph, 'code': bad_code})['project']
        failed = self.request('/api/run', {'project_id': bumped['id'], 'version': bumped['version'],
                                           'revision': bumped['revision']})
        self.assertNotEqual(failed['returncode'], 0)
        self.assertIn('KeyError', failed['stderr'])
        self.assertIn('KeyError', failed['trace']['error'])

    def test_save_as_copy_keeps_dict_graph(self):
        project = self.default_project()
        graph = {'version': 3, 'seed': 1,
                 'main': {'1': {'data': {'type': 'literal', 'name': 'mapping', 'value': ID_TO_TOKEN}}},
                 'functions': []}
        saved = self.request('/api/save', {'project_id': project['id'], 'version': project['version'],
                                           'graph': graph, 'code': None})['project']
        copy = self.request('/api/projects/save-as', {'project_id': saved['id'], 'name': 'DictCopy',
                                                      'graph': graph})['project']
        self.assertEqual(copy['graph']['main']['1']['data']['value'], ID_TO_TOKEN)


class RuntimeDictPreviewTest(unittest.TestCase):
    """运行查看器：dict 的有界键值预览，非字符串键保留类型面貌。"""

    def preview_of(self, value):
        return runtime_trace.Recorder(io.StringIO()).preview(value)

    def test_mapping_preview_bounded_pairs(self):
        preview = self.preview_of({0: '知', 1: '之', '知': 0})
        self.assertEqual(preview['kind'], 'mapping')
        self.assertEqual(preview['type'], 'dict')
        self.assertEqual(preview['count'], 3)
        self.assertFalse(preview['truncated'])
        self.assertEqual(preview['entries'][0], ['0', {'kind': 'value', 'type': 'str', 'repr': "'知'", 'truncated': False}])
        self.assertEqual(preview['entries'][2][0], "'知'")   # 字符串键带引号，非字符串键不带
        self.assertEqual(preview['entries'][2][1]['repr'], '0')

    def test_mapping_preview_truncates_at_8(self):
        preview = self.preview_of({i: i for i in range(20)})
        self.assertEqual(len(preview['entries']), 8)
        self.assertTrue(preview['truncated'])
        self.assertEqual(preview['count'], 20)

    def test_mapping_nested_and_depth_fallback(self):
        preview = self.preview_of({'a': {1: [2, 3.5]}, (1, 2): 'tuple'})
        self.assertEqual(preview['entries'][0][1]['kind'], 'mapping')
        # 与序列同一深度预算：第 3 层列表回退安全 repr（序列套序列的行为一致）
        self.assertEqual(preview['entries'][0][1]['entries'][0][1],
                         {'kind': 'value', 'type': 'list', 'repr': '<list>'})
        self.assertEqual(preview['entries'][1][0], '(1, 2)')
        deep = self.preview_of({1: {2: {3: {4: 5}}}})
        inner = deep['entries'][0][1]['entries'][0][1]
        self.assertEqual(inner['kind'], 'value')
        self.assertEqual(inner['repr'], '<dict>')   # 超出深度回退安全 repr，与序列一致
        # emit 序列化路径安全（allow_nan=False）
        stream = io.StringIO()
        recorder = runtime_trace.Recorder(stream)
        recorder.record(('main', '1'), [{0: '知'}])
        event = json.loads(stream.getvalue())
        self.assertEqual(event['outputs'][0]['kind'], 'mapping')

    def test_empty_dict_preview(self):
        preview = self.preview_of({})
        self.assertEqual(preview['kind'], 'mapping')
        self.assertEqual(preview['count'], 0)
        self.assertEqual(preview['entries'], [])


if __name__ == '__main__':
    unittest.main(verbosity=1)
