"""NodeSpec schema v1 validation and rendering (mirror of static/nodespec.js).

Shared rules (target syntax, imports dedupe, param typing, rendering) are kept
character-for-character identical to the browser core; canvas/test_nodespec_backend.py
cross-checks both. The backend validates specs independently and generates
candidate-run code itself. Trusted local execution only: free call targets and
declared imports are syntax-validated (never whitelist-hidden) and run as the
user's own trusted local Python — this project never claims a sandbox.
"""
import json
import math
import re

SCHEMA = 1
MAX_INPUTS = 8
MAX_PARAMS = 16
MAX_ARGS = 16
MAX_SELECT_OPTIONS = 20
MAX_IMPORTS = 8

# 候选建议（制作器 datalist 选项），不是白名单：target 可自由填写任何语法合法的名字。
FUNCTION_TARGETS = [
    'torch.sqrt', 'torch.sin', 'torch.cos', 'torch.abs', 'torch.neg',
    'torch.exp', 'torch.log', 'torch.relu', 'torch.tanh', 'torch.sigmoid',
    'torch.softmax', 'torch.triu', 'torch.tril', 'torch.ones_like',
    'torch.zeros_like', 'torch.transpose', 'torch.flatten',
    'math.sqrt', 'builtins.len',
]
METHOD_TARGETS = [
    'masked_fill', 'transpose', 'permute', 'reshape', 'view', 't', 'to',
    'float', 'double', 'half', 'bool', 'int', 'long', 'softmax', 'sigmoid',
    'tanh', 'relu', 'clamp', 'squeeze', 'unsqueeze', 'flatten',
    'abs', 'sqrt', 'exp', 'log', 'sin', 'cos', 'neg', 'triu', 'tril',
    'item', 'tolist',
]
PARAM_TYPES = ('int', 'number', 'bool', 'string', 'literal', 'select')
# 输入/输出共用的展示类型提示：只提示不转换；扩展项用于返回 Python 标量/容器的自由目标。
INPUT_TYPE_HINTS = ('Tensor', 'BoolTensor', 'Any', 'number', 'int',
                    'string', 'bool', 'list', 'tuple', 'dict')

ID_RE = re.compile(r'^[a-z][a-z0-9_]{0,63}$')
KEY_RE = re.compile(r'^[a-z][a-z0-9_]{0,30}$')
# 目标 / 模块名 / 别名的单段标识符：Python 标识符规则（下划线或字母开头，≤64 位）。
PY_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,63}$')
TARGET_MAX_LENGTH = 200    # 点分目标 / 模块名全长上限
TARGET_MAX_SEGMENTS = 16   # 点分段数上限
# 默认导入已占用的绑定名：生成代码头部固定有 import torch 与 import torch.nn as nn。
DEFAULT_IMPORT_NAMES = ('torch', 'nn')
CUSTOM_TYPE_RE = re.compile(r'^custom\.([a-z][a-z0-9_]{0,63})@(\d+)$')
INT_RE = re.compile(r'^-?\d+$')
NUMBER_RE = re.compile(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$')

PY_KEYWORDS = frozenset(('False None True and as assert async await break class continue def del elif else '
                         'except finally for from global if import in is lambda nonlocal not or pass raise return '
                         'try while with yield print self torch nn').split())
# Python 真实保留字（软关键字 match/case/type 合法标识符，不在其中）。
# 目标 / 模块名 / 别名的每一段都用它校验；与 PY_KEYWORDS（参数 key 用、额外含
# print/self/torch/nn）区分：torch.sqrt 的段 torch 必须放行。
PY_TRUE_KEYWORDS = frozenset(('False None True and as assert async await break class continue def del elif else '
                              'except finally for from global if import in is lambda nonlocal not or pass raise '
                              'return try while with yield').split())


class SpecError(ValueError):
    pass


def encode_type(spec):
    return f"custom.{spec['id']}@{spec['version']}"


def parse_custom_type(type_name):
    match = CUSTOM_TYPE_RE.match(str(type_name or ''))
    return {'id': match.group(1), 'version': int(match.group(2))} if match else None


def is_custom_type(type_name):
    return isinstance(type_name, str) and type_name.startswith('custom.')


def _fail(message):
    raise SpecError(message)


def _text(value, field, max_len, min_len=1):
    if not isinstance(value, str):
        _fail(f'{field} 必须是文本')
    text = value.strip()
    if len(text) < min_len or len(text) > max_len:
        _fail(f'{field} 长度需在 {min_len}-{max_len} 个字符之间')
    if '<' in text or '>' in text:
        _fail(f'{field} 不允许包含 < 或 >')
    return text


def _identifier(value, field):
    if not isinstance(value, str) or not KEY_RE.match(value):
        _fail(f'{field} 只允许小写字母开头的字母/数字/下划线（1-31 位）')
    if value in PY_KEYWORDS:
        _fail(f'{field}「{value}」是保留字，不能使用')
    return value


def _py_name(value, field):
    """单段 Python 标识符（目标 / 方法名 / 别名用）：允许下划线与大写，拒绝保留字。"""
    if not isinstance(value, str) or not PY_NAME_RE.match(value):
        _fail(f'{field}「{value if value is not None else ""}」必须是单个 Python 标识符'
              f'（字母/数字/下划线，不能以数字开头，不能含点号、空格、括号、分号等）')
    if value in PY_TRUE_KEYWORDS:
        _fail(f'{field}「{value}」是 Python 保留字，不能使用')
    return value


def _dotted_target(value, field='调用目标'):
    """自由目标语法校验：合法点分标识符（math.sqrt / torch.nn.functional.relu / builtins.len）。
    只做语法校验，没有任何隐藏白名单；括号/分号/换行/空格/通配符一律拒绝，
    保证生成的代码里 target 不可能拼接出额外语句。"""
    if not isinstance(value, str):
        _fail(f'{field} 必须是文本')
    text = value.strip()
    if not 0 < len(text) <= TARGET_MAX_LENGTH:
        _fail(f'{field} 长度需在 1-{TARGET_MAX_LENGTH} 个字符之间')
    segments = text.split('.')
    if len(segments) > TARGET_MAX_SEGMENTS:
        _fail(f'{field} 最多 {TARGET_MAX_SEGMENTS} 段')
    for segment in segments:
        if not PY_NAME_RE.match(segment):
            _fail(f'{field}「{text}」不是合法的点分 Python 标识符（形如 math.sqrt、torch.nn.functional.relu；'
                  f'每段只能是字母/数字/下划线，不能以数字开头，不能含空格、括号、分号、通配符等）')
        if segment in PY_TRUE_KEYWORDS:
            _fail(f'{field}「{text}」的段「{segment}」是 Python 保留字，不能使用')
    return text


def _normalize_imports(raw, field='imports'):
    """声明式 imports：{module, alias?}，对应 import <module> [as <alias>]。
    默认头部已有 import torch 与 import torch.nn as nn：重复声明被静默去重；
    绑定名（alias 或 module 首段，import a.b 绑定的是 a）与 torch/nn 冲突、
    或两条声明绑定同名 → 明确拒绝。"""
    if raw is None:
        return []
    if not isinstance(raw, list):
        _fail(f'{field} 必须是数组')
    if len(raw) > MAX_IMPORTS:
        _fail(f'导入最多 {MAX_IMPORTS} 条')
    out = []
    bound_names = {}   # 绑定名 -> 「module [as alias]」文本（用于冲突报错）
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            _fail(f'导入 {index + 1} 必须是对象')
        for key in item:
            if key not in ('module', 'alias'):
                _fail(f'导入 {index + 1} 未知字段「{key}」')
        module = _dotted_target(item.get('module'), f'导入 {index + 1} 的模块名')
        alias = item.get('alias')
        if alias is not None and alias != '':
            alias = _py_name(alias, f'导入 {index + 1} 的别名')
        else:
            alias = None
        text = f'{module} as {alias}' if alias else module
        if any(e['module'] == module and (e['alias'] or None) == alias for e in out):
            continue   # 完全相同 → 去重
        # 默认导入已存在：裸 torch.*（绑定名 torch）与 torch.nn as nn 不重复生成。
        if alias is None and (module == 'torch' or module.startswith('torch.')):
            continue
        if module == 'torch.nn' and alias == 'nn':
            continue
        bound = alias or module.split('.')[0]
        if bound in DEFAULT_IMPORT_NAMES:
            _fail(f'导入「{text}」的绑定名「{bound}」与默认导入 torch/nn 冲突，请改用其他别名')
        if bound in bound_names:
            _fail(f'导入绑定名「{bound}」冲突：「{bound_names[bound]}」与「{text}」不能同时绑定同一个名字')
        bound_names[bound] = text
        out.append({'module': module, 'alias': alias})
    return out


def _enum(value, field, options):
    if not isinstance(value, str) or value not in options:
        _fail(f"{field}「{value}」只允许：{'、'.join(options)}")
    return value


def _bool(value, field):
    if not isinstance(value, bool):
        _fail(f'{field} 必须是 true 或 false')
    return value


def render_number(text, field='数字参数'):
    raw = str(text if text is not None else '').strip()
    lowered = raw.lower()
    if lowered.startswith('+'):
        lowered = lowered[1:]
    if lowered in ('inf', 'infinity'):
        return 'float("inf")'
    if lowered in ('-inf', '-infinity'):
        return 'float("-inf")'
    if lowered == 'nan':
        return 'float("nan")'
    if not NUMBER_RE.match(raw):
        _fail(f'{field}「{raw}」不是合法数字（支持十进制、科学计数、inf、-inf、nan）')
    if not math.isfinite(float(raw)):
        _fail(f'{field}「{raw}」数值越界')
    return raw


def render_int(text, field='整数参数'):
    raw = str(text if text is not None else '').strip()
    if not INT_RE.match(raw):
        _fail(f'{field}「{raw}」必须是整数')
    return raw


def render_string(value, field='字符串参数'):
    if not isinstance(value, str):
        _fail(f'{field} 必须是文本')
    if len(value) > 500:
        _fail(f'{field} 超过 500 字符')
    return json.dumps(value, ensure_ascii=False)


def render_param_value(param, value, literal):
    """literal: callable source -> {'expression': ...} | {'error': ...} | None."""
    key = f"参数 {param['label'] or param['key']}"
    has_value = value is not None and str(value) != ''
    kind = param['type']
    if kind == 'int':
        return render_int(value if has_value else param['def'], key)
    if kind == 'number':
        return render_number(value if has_value else param['def'], key)
    if kind == 'bool':
        flag = value if has_value else param['def']
        if flag is True or flag == 'true':
            return 'True'
        if flag is False or flag == 'false':
            return 'False'
        _fail(f'{key} 必须是布尔值')
    if kind == 'string':
        return render_string(str(value) if has_value else str(param.get('def', '')), key)
    if kind == 'select':
        text = str(value) if has_value else str(param.get('def', ''))
        if text not in param.get('options', []):
            _fail(f'{key}「{text}」不在可选项中')
        return json.dumps(text, ensure_ascii=False)
    if kind == 'literal':
        source = str(value if has_value else param.get('def', '')).strip()
        if not source:
            _fail(f'{key} 缺少字面量')
        checked = literal(source) if literal else None
        if not checked:
            _fail(f'{key}「{source[:40]}」尚未通过 Literal 校验')
        if checked.get('error'):
            _fail(f"{key}「{source[:40]}」无效：{checked['error']}")
        return checked['expression']
    _fail(f'{key} 类型未知')


def _check_default(item, param, literal):
    field = f"参数 {param['key']} 默认值"
    kind = param['type']
    default = item.get('def')
    if kind == 'int':
        if isinstance(default, bool):
            _fail(f'{field} 必须是整数')
        if isinstance(default, int):
            return str(default)
        return render_int(default, field)
    if kind == 'number':
        if isinstance(default, bool):
            _fail(f'{field} 必须是数字')
        if isinstance(default, (int, float)):
            if not math.isfinite(default):
                _fail(f'{field} 数值越界')
            return str(default)
        render_number(default, field)   # 校验；默认值保留原始写法（如 -inf），渲染时再转换
        return str(default).strip()
    if kind == 'bool':
        if not isinstance(default, bool):
            _fail(f'{field} 必须是 true / false')
        return default
    if kind == 'string':
        return _text(default, field, 500, 0)
    if kind == 'select':
        if not isinstance(default, str) or default not in param.get('options', []):
            _fail(f'{field} 必须是 options 之一')
        return default
    if kind == 'literal':
        if not isinstance(default, str) or not default.strip() or len(default.strip()) > 500:
            _fail(f'{field} 必须是 1-500 字符的字面量源码')
        checked = literal(default.strip()) if literal else None
        if checked and checked.get('error'):
            _fail(f"{field}「{default.strip()}」无效：{checked['error']}")
        return default.strip()
    _fail(f'{field} 类型未知')


def validate_spec(raw, literal=None):
    """Strict schema validation; returns a canonical dict with fixed key order."""
    if not isinstance(raw, dict):
        _fail('节点规范必须是一个 JSON 对象')
    allowed = ['schema', 'id', 'version', 'label', 'description', 'category', 'imports', 'output', 'inputs', 'params', 'call']
    for key in raw:
        if key not in allowed:
            _fail(f'未知字段「{key}」')
    if raw.get('schema') != SCHEMA:
        _fail(f'仅支持 nodespec schema {SCHEMA}')
    spec_id = _identifier(raw.get('id'), '规范 id')
    version = raw.get('version')
    if type(version) is not int or not 1 <= version <= 1000000:
        _fail('版本号必须是 1 以上的整数')
    label = _text(raw.get('label'), '显示名称', 40)
    description = _text(raw.get('description', ''), '说明', 500, 0)
    category = _text(raw.get('category') or '自定义', '分类', 20) if raw.get('category') not in (None, '') else '自定义'
    imports = _normalize_imports(raw.get('imports'))

    output_raw = raw.get('output')
    if not isinstance(output_raw, dict):
        _fail('缺少 output（自定义节点只有一个输出）')
    for key in output_raw:
        if key not in ('name', 'type'):
            _fail(f'output 未知字段「{key}」')
    output = {'name': _text(output_raw.get('name'), '输出名称', 30),
              'type': _enum(output_raw.get('type', 'Tensor'), '输出类型', INPUT_TYPE_HINTS)}

    inputs_raw = raw.get('inputs')
    if inputs_raw is None:
        _fail('缺少 inputs 数组（可以为空）')
    if not isinstance(inputs_raw, list):
        _fail('inputs 必须是数组')
    if len(inputs_raw) > MAX_INPUTS:
        _fail(f'输入最多 {MAX_INPUTS} 个')
    seen_names = set()
    inputs = []
    for index, item in enumerate(inputs_raw):
        if not isinstance(item, dict):
            _fail(f'输入 {index + 1} 必须是对象')
        for key in item:
            if key not in ('name', 'type', 'optional', 'fallbackParam', 'help'):
                _fail(f'输入 {index + 1} 未知字段「{key}」')
        name = _text(item.get('name'), f'输入 {index + 1} 名称', 30)
        if name in seen_names:
            _fail(f'输入名称「{name}」重复')
        seen_names.add(name)
        optional = _bool(item.get('optional', False), f'输入 {name} 的 optional')
        help_text = _text(item.get('help', ''), f'输入 {name} 说明', 200, 0)
        if item.get('fallbackParam') is not None and not optional:
            _fail(f'输入 {name} 不是可选输入，不能配置 fallbackParam')
        inputs.append({'name': name,
                       'type': _enum(item.get('type', 'Tensor'), f'输入 {name} 类型', INPUT_TYPE_HINTS),
                       'optional': optional,
                       'fallbackParam': item.get('fallbackParam'),
                       'help': help_text})

    params_raw = raw.get('params')
    if params_raw is None:
        _fail('缺少 params 数组（可以为空）')
    if not isinstance(params_raw, list):
        _fail('params 必须是数组')
    if len(params_raw) > MAX_PARAMS:
        _fail(f'参数最多 {MAX_PARAMS} 个')
    seen_keys = set()
    params = []
    for index, item in enumerate(params_raw):
        if not isinstance(item, dict):
            _fail(f'参数 {index + 1} 必须是对象')
        for key in item:
            if key not in ('key', 'label', 'type', 'def', 'help', 'options'):
                _fail(f'参数 {index + 1} 未知字段「{key}」')
        key = _identifier(item.get('key'), f'参数 {index + 1} key')
        if key in seen_keys:
            _fail(f'参数 key「{key}」重复')
        seen_keys.add(key)
        param = {'key': key,
                 'label': _text(item.get('label'), f'参数 {key} 显示名', 40),
                 'type': _enum(item.get('type'), f'参数 {key} 类型', PARAM_TYPES),
                 'def': None,
                 'help': _text(item.get('help', ''), f'参数 {key} 说明', 300, 0)}
        if param['type'] == 'select':
            options = item.get('options')
            if not isinstance(options, list) or not 1 <= len(options) <= MAX_SELECT_OPTIONS:
                _fail(f'参数 {key} 的 options 需要 1-{MAX_SELECT_OPTIONS} 个选项')
            options = [_text(option, f'参数 {key} 选项', 30) for option in options]
            if len(set(options)) != len(options):
                _fail(f'参数 {key} 的选项重复')
            param['options'] = options
        param['def'] = _check_default(item, param, literal)
        params.append(param)
    param_keys = {p['key'] for p in params}

    for entry in inputs:
        fallback = entry['fallbackParam']
        if fallback is None:
            continue
        if not isinstance(fallback, str) or fallback not in param_keys:
            _fail(f"输入 {entry['name']} 的 fallbackParam「{fallback}」不是已定义的参数 key")

    call_raw = raw.get('call')
    if not isinstance(call_raw, dict):
        _fail('缺少 call 调用模板')
    for key in call_raw:
        if key not in ('kind', 'target', 'args'):
            _fail(f'call 未知字段「{key}」')
    kind = _enum(call_raw.get('kind'), '调用类型', ('function', 'method'))
    # 目标自由填写：function 允许任意合法点分标识符，method 只允许单个标识符（方法名）。
    # 只做语法校验，没有任何隐藏白名单；执行信任边界见模块 docstring。
    if kind == 'function':
        target = _dotted_target(call_raw.get('target'), '调用目标')
    else:
        target = _py_name(call_raw.get('target'), '方法名（单个方法标识符，不能含点号）')
    args_raw = call_raw.get('args')
    if not isinstance(args_raw, list) or (not args_raw and kind != 'method'):
        _fail('call.args 至少需要一个绑定（方法调用可以只靠接收者，支持无参数方法如 item/tolist）')
    if len(args_raw) > MAX_ARGS:
        _fail(f'绑定最多 {MAX_ARGS} 个')
    used_inputs, used_params = set(), set()
    args = []
    for index, item in enumerate(args_raw):
        if not isinstance(item, dict):
            _fail(f'绑定 {index + 1} 必须是对象')
        for key in item:
            if key not in ('source', 'index', 'key', 'keyword', 'fallbackParam'):
                _fail(f'绑定 {index + 1} 未知字段「{key}」')
        keyword = None
        if item.get('keyword') not in (None,):
            keyword = _identifier(item['keyword'], f'绑定 {index + 1} 关键字名')
        if item.get('source') == 'input':
            index_value = item.get('index')
            if type(index_value) is not int or not 1 <= index_value <= len(inputs):
                _fail(f'绑定 {index + 1} 的输入序号超出范围（1-{max(len(inputs), 1)}）')
            if index_value in used_inputs:
                _fail(f'输入 {index_value} 被绑定了多次')
            used_inputs.add(index_value)
            if kind == 'method' and index_value == 1:
                _fail('方法调用的输入 1 是接收者（receiver），不能重复绑定到实参')
            fallback = item.get('fallbackParam')
            if fallback is not None:
                if not isinstance(fallback, str) or fallback not in param_keys:
                    _fail(f'绑定 {index + 1} 的 fallbackParam「{fallback}」不是已定义的参数 key')
            args.append({'source': 'input', 'index': index_value, 'keyword': keyword, 'fallbackParam': fallback})
        elif item.get('source') == 'param':
            key = item.get('key')
            if not isinstance(key, str) or key not in param_keys:
                _fail(f'绑定 {index + 1} 引用的参数「{key}」未定义')
            if key in used_params:
                _fail(f'参数 {key} 被绑定了多次')
            used_params.add(key)
            args.append({'source': 'param', 'key': key, 'keyword': keyword, 'fallbackParam': None})
        else:
            _fail(f'绑定 {index + 1} 的 source 必须是 input 或 param')

    if kind == 'method':
        if not inputs:
            _fail('方法调用至少需要一个输入作为接收者')
        if inputs[0]['optional']:
            _fail('方法调用的接收者（输入 1）不能是可选输入')
        used_inputs.add(1)
    for number in range(1, len(inputs) + 1):
        if number not in used_inputs:
            _fail(f'输入 {number}（{inputs[number - 1]["name"]}）没有被绑定到调用；每个输入都必须参与调用')
    for param in params:
        consumed = param['key'] in used_params or any(e['fallbackParam'] == param['key'] for e in inputs) \
            or any(a['fallbackParam'] == param['key'] for a in args)
        if not consumed:
            _fail(f"参数 {param['key']} 未被任何绑定使用")

    return {'schema': SCHEMA, 'id': spec_id, 'version': version, 'label': label, 'description': description,
            'category': category, 'imports': imports, 'output': output, 'inputs': inputs, 'params': params,
            'call': {'kind': kind, 'target': target, 'args': args}}


def canonical_json(spec):
    return json.dumps(validate_spec(spec), ensure_ascii=False, separators=(',', ':'))


def render_call(spec, input_exprs, param_values=None, literal=None):
    """Char-for-char identical to NS.renderCall in static/nodespec.js.

    input_exprs[i] is the expression for input i+1; None means not connected."""
    values = {}
    for param in spec['params']:
        supplied = (param_values or {}).get(param['key'])
        values[param['key']] = param['def'] if supplied is None or str(supplied) == '' else supplied
    param_exprs = {p['key']: render_param_value(p, values[p['key']], literal) for p in spec['params']}
    args = []
    for binding in spec['call']['args']:
        if binding['source'] == 'input':
            expr = input_exprs[binding['index'] - 1] if input_exprs else None
            if expr is None or expr == '':
                if binding['fallbackParam']:
                    expr = param_exprs[binding['fallbackParam']]
                else:
                    _fail(f"输入 {binding['index']} 未连线且未配置参数回退（fallbackParam）")
        else:
            expr = param_exprs[binding['key']]
        args.append(f"{binding['keyword']}={expr}" if binding['keyword'] else expr)
    if spec['call']['kind'] == 'method':
        receiver = input_exprs[0] if input_exprs else None
        if receiver is None or receiver == '':
            _fail('接收者（输入 1）未连线')
        return f"{receiver}.{spec['call']['target']}({', '.join(args)})"
    return f"{spec['call']['target']}({', '.join(args)})"


# ---------------------------------------------------------------- candidates
TENSOR_DTYPES = ('float32', 'float64', 'float16', 'bfloat16', 'int64', 'int32', 'bool')
MAX_DIMS = 8
MAX_ELEMENTS = 1_000_000


def validate_samples(raw, spec):
    """Per-input candidate samples: literal source or explicit tensor options."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        _fail('试运行 samples 必须是对象')
    samples = []
    for index, entry in enumerate(spec['inputs']):
        supplied = raw.get(str(index + 1), raw.get(index + 1))
        if supplied is None:
            if entry['optional']:
                samples.append(None)   # simulates an unconnected optional input
                continue
            _fail(f"输入 {index + 1}（{entry['name']}）缺少试运行样本")
        if not isinstance(supplied, dict):
            _fail(f'输入 {index + 1} 的样本必须是对象')
        kind = _enum(supplied.get('kind'), f'输入 {index + 1} 样本类型', ('literal', 'tensor'))
        if kind == 'literal':
            source = supplied.get('source')
            if not isinstance(source, str) or not source.strip() or len(source) > 10000:
                _fail(f'输入 {index + 1} 的字面量必须是 1-10000 字符')
            samples.append({'kind': 'literal', 'source': source.strip()})
        else:
            shape = supplied.get('shape')
            if not isinstance(shape, list) or not shape or len(shape) > MAX_DIMS or any(type(n) is not int or n < 1 or n > 10000 for n in shape):
                _fail(f'输入 {index + 1} 的 shape 需要 1-{MAX_DIMS} 个 1-10000 的整数')
            total = 1
            for n in shape:
                total *= n
            if total > MAX_ELEMENTS:
                _fail(f'输入 {index + 1} 的样本元素个数超过 {MAX_ELEMENTS}')
            dtype = _enum(supplied.get('dtype', 'float32'), f'输入 {index + 1} dtype', TENSOR_DTYPES)
            fill = _enum(supplied.get('fill', 'constant'), f'输入 {index + 1} 填充方式', ('constant', 'rand'))
            sample = {'kind': 'tensor', 'shape': shape, 'dtype': dtype, 'fill': fill}
            if fill == 'constant':
                value = supplied.get('value')
                if not isinstance(value, str) or not value.strip() or len(value) > 10000:
                    _fail(f'输入 {index + 1} 的填充值必须是 1-10000 字符字面量')
                sample['value'] = value.strip()
            samples.append(sample)
    return samples


def _sample_expr(sample, literal):
    if sample['kind'] == 'literal':
        return _literal_expression(sample['source'], literal, f"样本「{sample['source'][:40]}」")
    dtype = f"torch.{sample['dtype']}"
    sizes = ', '.join(str(n) for n in sample['shape'])
    if sample['fill'] == 'rand':
        return f'torch.randn({sizes}, dtype={dtype})'
    value = _literal_expression(sample['value'], literal, f"填充值「{sample['value'][:40]}」")
    return f'torch.full(({sizes},), {value}, dtype={dtype})'


def _literal_expression(source, literal, label):
    checked = literal(source)
    if not checked or checked.get('error'):
        detail = checked.get('error') if checked else '校验服务不可用'
        _fail(f'{label}无效：{detail}')
    return checked['expression']


def _import_lines(imports):
    """规范里的声明式 imports → 真实 import 语句（validate_spec 已去重/去默认冲突）。"""
    lines = []
    for entry in imports or []:
        lines.append(f"import {entry['module']} as {entry['alias']}" if entry.get('alias') else f"import {entry['module']}")
    return lines


# 候选预览：有界的非 Tensor 处理，与 runtime_trace.Recorder.preview 同一套有界策略
# （标量 repr 截断、序列/映射只取前 8 项、嵌套 2 层），Tensor 分支保持旧版
# shape/dtype/values 语义（≤256 个元素全量、其余截断）。
PREVIEW_CODE = '''def _ns_preview(value, depth=0):
    kind = type(value).__name__
    if isinstance(value, torch.Tensor):
        shape = list(value.shape)
        numel = value.numel()
        if numel <= 256:
            values = value.detach().cpu().reshape(-1).tolist()
            truncated = False
        else:
            values = value.detach().cpu().reshape(-1)[:256].tolist()
            truncated = True
        return {'kind': 'tensor', 'type': kind, 'shape': shape, 'dtype': str(value.dtype),
                'numel': numel, 'values': values, 'truncated': truncated,
                'requires_grad': bool(value.requires_grad)}
    if isinstance(value, torch.nn.Module):
        return {'kind': 'value', 'type': kind, 'repr': '<' + kind + ' Module>'}
    if value is None or isinstance(value, (bool, int, float, complex)):
        return {'kind': 'value', 'type': kind, 'repr': repr(value)[:200]}
    if isinstance(value, str):
        return {'kind': 'value', 'type': kind, 'repr': repr(value[:300]), 'truncated': len(value) > 300}
    if isinstance(value, (list, tuple)) and depth < 2:
        return {'kind': 'sequence', 'type': kind, 'count': len(value),
                'items': [_ns_preview(x, depth + 1) for x in list(value)[:8]],
                'truncated': len(value) > 8}
    if isinstance(value, dict) and depth < 2:
        entries = []
        for index, (key, item) in enumerate(value.items()):
            if index >= 8:
                break
            entries.append([repr(key)[:100], _ns_preview(item, depth + 1)])
        return {'kind': 'mapping', 'type': kind, 'count': len(value),
                'entries': entries, 'truncated': len(value) > 8}
    return {'kind': 'value', 'type': kind, 'repr': repr(value)[:300]}


try:
    _payload = _ns_preview(out)
except BaseException as _exc:
    _payload = {'kind': 'value', 'type': 'preview error', 'repr': str(_exc)[:300]}
print(json.dumps(_payload, ensure_ascii=False, default=lambda o: repr(o)[:200]))
'''


def build_try_code(spec, samples, literal):
    """Backend generates the candidate code itself; the client never ships code.

    头部 import 与正式生成代码一致（默认 torch / torch.nn as nn + 规范声明的
    imports，validate_spec 已去重）；预览对任意返回值给出有界 type/repr/预览。"""
    head = ['import json, torch', 'import torch.nn as nn', 'torch.manual_seed(1234)']
    head += _import_lines(spec.get('imports'))
    body = []
    names = []
    for index, sample in enumerate(samples):
        if sample is None:
            names.append(None)
            continue
        name = f'in{index + 1}'
        body.append(f'{name} = {_sample_expr(sample, literal)}')
        names.append(name)
    input_exprs = [name if name is not None else None for name in names]
    call = render_call(spec, input_exprs, None, literal)
    return ''.join(line + '\n' for line in head) + ''.join(line + '\n' for line in body) + f'out = {call}\n' + PREVIEW_CODE


def validate_graph_specs(graph):
    """Project-snapshot validation: nodeSpecs schema + every custom node reference."""
    raw_specs = graph.get('nodeSpecs')
    if raw_specs is None:
        return []
    if not isinstance(raw_specs, list):
        _fail('nodeSpecs 必须是数组')
    by_encoded = {}
    specs = []
    for raw in raw_specs:
        spec = validate_spec(raw)
        encoded = encode_type(spec)
        text = json.dumps(spec, ensure_ascii=False, sort_keys=True)
        if encoded in by_encoded:
            if by_encoded[encoded] != text:
                _fail(f'版本冲突：{encoded} 在 nodeSpecs 中出现了不同内容')
            continue
        by_encoded[encoded] = text
        specs.append(spec)
    scopes = [graph.get('main')]
    for function in graph.get('functions', []):
        scopes.append(function.get('data') if isinstance(function, dict) else None)
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        for node in scope.values():
            type_name = (node or {}).get('data', {}).get('type') if isinstance(node, dict) else None
            if is_custom_type(type_name) and type_name not in by_encoded:
                _fail(f'节点引用的 {type_name} 不在本项目的 nodeSpecs 中；请先恢复或重新发布该定义')
    return specs
