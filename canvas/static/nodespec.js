/*
 * nodespec.js —— 自定义节点规范（NodeSpec）schema v1 纯核心
 * 职责：规范 normalize/校验、注册表（发布定义深不可变、同 id+version 不可覆盖不同内容）、
 *       call 目标语法校验（自由目标：合法点分 Python 标识符，不做隐藏白名单）、
 *       声明式 imports 校验与去重、参数渲染规则、调用表达式生成。
 * 本文件保持无 DOM 依赖，浏览器与 node 测试（require）共用同一套规则；
 * Python 侧镜像实现见 nodespecs.py（两侧规则由测试保证一致）。
 * 类型编码：custom.<id>@<version>，例如 custom.sqrt@1；旧节点永远绑定其发布时的版本。
 * 信任边界：目标与 import 只做语法校验，执行的是用户本机 Python 里真实可见的名字，
 * 其副作用（文件/网络/系统）由用户自己负责；本模块保证绝不生成 target 之外的额外语句。
 */
(function (global) {
  'use strict';

  const SCHEMA = 1;
  const MAX_INPUTS = 8;
  const MAX_PARAMS = 16;
  const MAX_ARGS = 16;
  const MAX_SELECT_OPTIONS = 20;
  const MAX_IMPORTS = 8;

  /* 候选建议（制作器 datalist 选项），不是白名单：target 可自由填写任何语法合法的名字。 */
  const FUNCTION_TARGETS = [
    'torch.sqrt', 'torch.sin', 'torch.cos', 'torch.abs', 'torch.neg',
    'torch.exp', 'torch.log', 'torch.relu', 'torch.tanh', 'torch.sigmoid',
    'torch.softmax', 'torch.triu', 'torch.tril', 'torch.ones_like',
    'torch.zeros_like', 'torch.transpose', 'torch.flatten',
    'math.sqrt', 'builtins.len',
  ];
  const METHOD_TARGETS = [
    'masked_fill', 'transpose', 'permute', 'reshape', 'view', 't', 'to',
    'float', 'double', 'half', 'bool', 'int', 'long', 'softmax', 'sigmoid',
    'tanh', 'relu', 'clamp', 'squeeze', 'unsqueeze', 'flatten',
    'abs', 'sqrt', 'exp', 'log', 'sin', 'cos', 'neg', 'triu', 'tril',
    'item', 'tolist',
  ];
  const PARAM_TYPES = ['int', 'number', 'bool', 'string', 'literal', 'select'];
  /* 输入/输出共用的展示类型提示：只提示不转换；string/bool/list/tuple/dict 用于
   * 返回 Python 标量或容器的自由目标（item/tolist/math.sqrt 等）。 */
  const INPUT_TYPE_HINTS = ['Tensor', 'BoolTensor', 'Any', 'number', 'int', 'string', 'bool', 'list', 'tuple', 'dict'];

  /* Python 真实保留字（软关键字 match/case/type 合法标识符，不在其中）。
   * 目标 / 模块名 / 别名的每一段都用它校验；与 PY_KEYWORDS（参数 key 用、额外含
   * print/self/torch/nn）区分：torch.sqrt 的段 torch 必须放行。 */
  const PY_TRUE_KEYWORDS = new Set(('False None True and as assert async await break class continue def del elif else '
    + 'except finally for from global if import in is lambda nonlocal not or pass raise return '
    + 'try while with yield').split(' '));

  /* 与 app.js PY_KEYWORDS 一致的 Python 保留字（参数 key / 关键字实参名校验用）。 */
  const PY_KEYWORDS = new Set(('False None True and as assert async await break class continue def del elif else '
    + 'except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield '
    + 'print self torch nn').split(' '));

  const ID_RE = /^[a-z][a-z0-9_]{0,63}$/;
  const KEY_RE = /^[a-z][a-z0-9_]{0,30}$/;
  /* 目标 / 模块名 / 别名的单段标识符：Python 标识符规则（下划线或字母开头，≤64 位）。 */
  const PY_NAME_RE = /^[A-Za-z_][A-Za-z0-9_]{0,63}$/;
  const TARGET_MAX_LENGTH = 200;    // 点分目标 / 模块名全长上限
  const TARGET_MAX_SEGMENTS = 16;   // 点分段数上限
  /* 默认导入已占用的绑定名：生成的代码头部固定有 import torch 与 import torch.nn as nn。 */
  const DEFAULT_IMPORT_NAMES = ['torch', 'nn'];
  const TYPE_RE = /^custom\.([a-z][a-z0-9_]{0,63})@(\d+)$/;
  const INT_RE = /^-?\d+$/;
  const NUMBER_RE = /^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/;

  const encode = (id, version) => `custom.${id}@${version}`;
  const isCustomType = t => typeof t === 'string' && /^custom\./.test(t);
  const parseType = t => {
    const m = TYPE_RE.exec(String(t ?? ''));
    return m ? { id: m[1], version: Number(m[2]) } : null;
  };

  class SpecError extends Error {}
  class VersionConflict extends SpecError {}

  function fail(message) { throw new SpecError(message); }

  function checkText(value, field, max, { min = 1 } = {}) {
    if (typeof value !== 'string') fail(`${field} 必须是文本`);
    const text = value.trim();
    if (text.length < min || text.length > max) fail(`${field} 长度需在 ${min}-${max} 个字符之间`);
    if (/[<>]/.test(text)) fail(`${field} 不允许包含 < 或 >`);
    return text;
  }

  function checkIdentifier(value, field) {
    if (typeof value !== 'string' || !KEY_RE.test(value)) fail(`${field} 只允许小写字母开头的字母/数字/下划线（1-31 位）`);
    if (PY_KEYWORDS.has(value)) fail(`${field}「${value}」是保留字，不能使用`);
    return value;
  }

  /* 单段 Python 标识符（目标 / 方法名 / 别名用）：允许下划线与大写，拒绝保留字。 */
  function checkPyName(value, field) {
    if (typeof value !== 'string' || !PY_NAME_RE.test(value)) {
      fail(`${field}「${value ?? ''}」必须是单个 Python 标识符（字母/数字/下划线，不能以数字开头，不能含点号、空格、括号、分号等）`);
    }
    if (PY_TRUE_KEYWORDS.has(value)) fail(`${field}「${value}」是 Python 保留字，不能使用`);
    return value;
  }

  /* 自由目标语法校验：合法点分标识符（math.sqrt / torch.nn.functional.relu / builtins.len）。
   * 只做语法校验，没有任何隐藏白名单；括号/分号/换行/空格/通配符一律拒绝，
   * 保证生成的代码里 target 不可能拼接出额外语句。 */
  function checkDottedTarget(value, field = '调用目标') {
    if (typeof value !== 'string') fail(`${field} 必须是文本`);
    const text = value.trim();
    if (!text.length || text.length > TARGET_MAX_LENGTH) fail(`${field} 长度需在 1-${TARGET_MAX_LENGTH} 个字符之间`);
    const segments = text.split('.');
    if (segments.length > TARGET_MAX_SEGMENTS) fail(`${field} 最多 ${TARGET_MAX_SEGMENTS} 段`);
    for (const segment of segments) {
      if (!PY_NAME_RE.test(segment)) {
        fail(`${field}「${text}」不是合法的点分 Python 标识符（形如 math.sqrt、torch.nn.functional.relu；每段只能是字母/数字/下划线，不能以数字开头，不能含空格、括号、分号、通配符等）`);
      }
      if (PY_TRUE_KEYWORDS.has(segment)) fail(`${field}「${text}」的段「${segment}」是 Python 保留字，不能使用`);
    }
    return text;
  }

  /* ---- 声明式 imports：{module, alias?}，对应 import <module> [as <alias>] ----
   * 默认头部已有 import torch 与 import torch.nn as nn：重复声明被静默去重；
   * 绑定名（alias 或 module 首段，import a.b 绑定的是 a）与 torch/nn 冲突、
   * 或两条声明绑定同名 → 明确拒绝。 */
  function normalizeImports(raw, field = 'imports') {
    if (raw === undefined || raw === null) return [];
    if (!Array.isArray(raw)) fail(`${field} 必须是数组`);
    if (raw.length > MAX_IMPORTS) fail(`导入最多 ${MAX_IMPORTS} 条`);
    const out = [];
    const boundNames = new Map();   // 绑定名 -> 「module [as alias]」文本（用于冲突报错）
    for (const [i, item] of raw.entries()) {
      if (!item || typeof item !== 'object' || Array.isArray(item)) fail(`导入 ${i + 1} 必须是对象`);
      for (const key of Object.keys(item)) if (!['module', 'alias'].includes(key)) fail(`导入 ${i + 1} 未知字段「${key}」`);
      const module = checkDottedTarget(item.module, `导入 ${i + 1} 的模块名`);
      let alias = null;
      if (item.alias !== undefined && item.alias !== null && item.alias !== '') {
        alias = checkPyName(item.alias, `导入 ${i + 1} 的别名`);
      }
      const text = alias ? `${module} as ${alias}` : module;
      if (out.some(e => e.module === module && (e.alias || null) === alias)) continue;   // 完全相同 → 去重
      /* 默认导入已存在：裸 torch.*（绑定名 torch）与 torch.nn as nn 不重复生成。 */
      if (!alias && (module === 'torch' || module.startsWith('torch.'))) continue;
      if (module === 'torch.nn' && alias === 'nn') continue;
      const bound = alias || module.split('.')[0];
      if (DEFAULT_IMPORT_NAMES.includes(bound)) {
        fail(`导入「${text}」的绑定名「${bound}」与默认导入 torch/nn 冲突，请改用其他别名`);
      }
      if (boundNames.has(bound)) fail(`导入绑定名「${bound}」冲突：「${boundNames.get(bound)}」与「${text}」不能同时绑定同一个名字`);
      boundNames.set(bound, text);
      out.push({ module, alias });
    }
    return out;
  }

  /* 数字字面量共享规则：十进制（含科学计数）或 inf/-inf/nan；
   * 渲染结果两侧必须逐字符一致（JS renderNumber / Python render_number）。 */
  function renderNumber(text, field = '数字参数') {
    const raw = String(text ?? '').trim();
    const lowered = raw.toLowerCase().replace(/^\+/, '');
    if (['inf', 'infinity'].includes(lowered)) return 'float("inf")';
    if (lowered === '-inf' || lowered === '-infinity') return 'float("-inf")';
    if (lowered === 'nan') return 'float("nan")';
    if (!NUMBER_RE.test(raw)) fail(`${field}「${raw}」不是合法数字（支持十进制、科学计数、inf、-inf、nan）`);
    const value = Number(raw);
    if (!Number.isFinite(value)) fail(`${field}「${raw}」数值越界`);
    return raw;
  }

  /* int 参数只接受整数字面量；不支持构造参数替换（自定义节点参数是节点自身的数据）。 */
  function renderInt(text, field = '整数参数') {
    const raw = String(text ?? '').trim();
    if (!/^-?\d+$/.test(raw)) fail(`${field}「${raw}」必须是整数`);
    return raw;
  }

  function renderString(value, field = '字符串参数') {
    if (typeof value !== 'string') fail(`${field} 必须是文本`);
    if (value.length > 500) fail(`${field} 超过 500 字符`);
    return JSON.stringify(value);
  }

  /* 参数渲染：int/number/bool/string/select 就地校验；literal 必须经
   * literalResolver（后端 /api/literal 的 ast.literal_eval 白名单）取回规范表达式。 */
  function renderParamValue(param, value, literalResolver) {
    const key = `参数 ${param.label || param.key}`;
    const hasValue = value !== undefined && value !== null && String(value) !== '';
    switch (param.type) {
      case 'int': return renderInt(hasValue ? value : param.def, key);
      case 'number': return renderNumber(hasValue ? value : param.def, key);
      case 'bool': {
        const flag = hasValue ? value : param.def;
        if (flag === true || flag === 'true') return 'True';
        if (flag === false || flag === 'false') return 'False';
        return fail(`${key} 必须是布尔值`);
      }
      case 'string': return renderString(hasValue ? String(value) : String(param.def ?? ''), key);
      case 'select': {
        const text = String(hasValue ? value : param.def ?? '');
        if (!(param.options || []).includes(text)) fail(`${key}「${text}」不在可选项中`);
        return JSON.stringify(text);
      }
      case 'literal': {
        const source = String(hasValue ? value : param.def ?? '').trim();
        if (!source) fail(`${key} 缺少字面量`);
        const checked = literalResolver ? literalResolver(source) : null;
        if (!checked) fail(`${key}「${source.slice(0, 40)}」尚未通过 Literal 校验`);
        if (checked.error) fail(`${key}「${source.slice(0, 40)}」无效：${checked.error}`);
        return checked.expression;
      }
      default: return fail(`${key} 类型未知`);
    }
  }

  /* ---- normalize：严格 schema，未知字段一律拒绝，输出键序固定的规范副本 ---- */
  function normalize(raw, hooks = {}) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) fail('节点规范必须是一个 JSON 对象');
    const allowed = ['schema', 'id', 'version', 'label', 'description', 'category', 'imports', 'output', 'inputs', 'params', 'call'];
    for (const key of Object.keys(raw)) if (!allowed.includes(key)) fail(`未知字段「${key}」`);
    if (raw.schema !== SCHEMA) fail(`仅支持 nodespec schema ${SCHEMA}`);
    checkIdentifier(raw.id, '规范 id');
    if (Number.isInteger(raw.version) === false || raw.version < 1 || raw.version > 1000000) fail('版本号必须是 1 以上的整数');
    const label = checkText(raw.label, '显示名称', 40);
    const description = raw.description === undefined ? '' : checkText(raw.description, '说明', 500, { min: 0 });
    const category = raw.category === undefined || raw.category === '' ? '自定义' : checkText(raw.category, '分类', 20);
    const imports = normalizeImports(raw.imports);

    /* 单输出 */
    if (!raw.output || typeof raw.output !== 'object' || Array.isArray(raw.output)) fail('缺少 output（自定义节点只有一个输出）');
    for (const key of Object.keys(raw.output)) if (!['name', 'type'].includes(key)) fail(`output 未知字段「${key}」`);
    const output = {
      name: checkText(raw.output.name, '输出名称', 30),
      type: raw.output.type === undefined ? 'Tensor' : checkEnum(raw.output.type, '输出类型', INPUT_TYPE_HINTS),
    };

    /* 固定 inputs */
    if (raw.inputs === undefined || !Array.isArray(raw.inputs)) fail('缺少 inputs 数组（可以为空）');
    if (raw.inputs.length > MAX_INPUTS) fail(`输入最多 ${MAX_INPUTS} 个`);
    const seenInput = new Set();
    const inputs = raw.inputs.map((item, i) => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) fail(`输入 ${i + 1} 必须是对象`);
      for (const key of Object.keys(item)) if (!['name', 'type', 'optional', 'fallbackParam', 'help'].includes(key)) fail(`输入 ${i + 1} 未知字段「${key}」`);
      const name = checkText(item.name, `输入 ${i + 1} 名称`, 30);
      if (seenInput.has(name)) fail(`输入名称「${name}」重复`);
      seenInput.add(name);
      const type = item.type === undefined ? 'Tensor' : checkEnum(item.type, `输入 ${name} 类型`, INPUT_TYPE_HINTS);
      const optional = item.optional === undefined ? false : requireBool(item.optional, `输入 ${name} 的 optional`);
      const help = item.help === undefined ? '' : checkText(item.help, `输入 ${name} 说明`, 200, { min: 0 });
      if (item.fallbackParam !== undefined && item.fallbackParam !== null && !optional) {
        fail(`输入 ${name} 不是可选输入，不能配置 fallbackParam`);
      }
      return { name, type, optional, fallbackParam: item.fallbackParam === undefined ? null : item.fallbackParam, help };
    });

    /* params */
    if (raw.params === undefined) fail('缺少 params 数组（可以为空）');
    if (!Array.isArray(raw.params)) fail('params 必须是数组');
    if (raw.params.length > MAX_PARAMS) fail(`参数最多 ${MAX_PARAMS} 个`);
    const seenKey = new Set();
    const params = raw.params.map((item, i) => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) fail(`参数 ${i + 1} 必须是对象`);
      for (const key of Object.keys(item)) if (!['key', 'label', 'type', 'def', 'help', 'options'].includes(key)) fail(`参数 ${i + 1} 未知字段「${key}」`);
      const key = checkIdentifier(item.key, `参数 ${i + 1} key`);
      if (seenKey.has(key)) fail(`参数 key「${key}」重复`);
      seenKey.add(key);
      const plabel = checkText(item.label, `参数 ${key} 显示名`, 40);
      const type = checkEnum(item.type, `参数 ${key} 类型`, PARAM_TYPES);
      const help = item.help === undefined ? '' : checkText(item.help, `参数 ${key} 说明`, 300, { min: 0 });
      const param = { key, label: plabel, type, def: null, help };
      if (type === 'select') {
        if (!Array.isArray(item.options) || !item.options.length || item.options.length > MAX_SELECT_OPTIONS) {
          fail(`参数 ${key} 的 options 需要 1-${MAX_SELECT_OPTIONS} 个选项`);
        }
        const options = item.options.map(o => checkText(o, `参数 ${key} 选项`, 30));
        if (new Set(options).size !== options.length) fail(`参数 ${key} 的选项重复`);
        param.options = options;
      }
      param.def = checkDefault(item, param, hooks);
      return param;
    });
    const paramKeys = new Set(params.map(p => p.key));

    /* fallbackParam 引用检查 */
    for (const input of inputs) {
      if (input.fallbackParam === undefined || input.fallbackParam === null) { input.fallbackParam = null; continue; }
      if (typeof input.fallbackParam !== 'string' || !paramKeys.has(input.fallbackParam)) {
        fail(`输入 ${input.name} 的 fallbackParam「${input.fallbackParam}」不是已定义的参数 key`);
      }
    }

    /* call：receiver（method 时为输入 1）+ 显式绑定。
     * 目标自由填写：function 允许任意合法点分标识符，method 只允许单个标识符（方法名）。 */
    if (!raw.call || typeof raw.call !== 'object' || Array.isArray(raw.call)) fail('缺少 call 调用模板');
    for (const key of Object.keys(raw.call)) if (!['kind', 'target', 'args'].includes(key)) fail(`call 未知字段「${key}」`);
    const kind = checkEnum(raw.call.kind, '调用类型', ['function', 'method']);
    let target;
    if (kind === 'function') {
      target = checkDottedTarget(raw.call.target, '调用目标');
    } else {
      target = checkPyName(raw.call.target, '方法名（单个方法标识符，不能含点号）');
    }
    if (!Array.isArray(raw.call.args) || (!raw.call.args.length && kind !== 'method')) {
      fail('call.args 至少需要一个绑定（方法调用可以只靠接收者，支持无参数方法如 item/tolist）');
    }
    if (raw.call.args.length > MAX_ARGS) fail(`绑定最多 ${MAX_ARGS} 个`);
    const usedInputs = new Set(), usedParams = new Set();
    const args = raw.call.args.map((item, i) => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) fail(`绑定 ${i + 1} 必须是对象`);
      for (const key of Object.keys(item)) if (!['source', 'index', 'key', 'keyword', 'fallbackParam'].includes(key)) fail(`绑定 ${i + 1} 未知字段「${key}」`);
      const keyword = item.keyword === undefined || item.keyword === null ? null : checkIdentifier(item.keyword, `绑定 ${i + 1} 关键字名`);
      let binding;
      if (item.source === 'input') {
        if (!Number.isInteger(item.index) || item.index < 1 || item.index > inputs.length) {
          fail(`绑定 ${i + 1} 的输入序号超出范围（1-${Math.max(inputs.length, 1)}）`);
        }
        if (usedInputs.has(item.index)) fail(`输入 ${item.index} 被绑定了多次`);
        usedInputs.add(item.index);
        if (kind === 'method' && item.index === 1) fail('方法调用的输入 1 是接收者（receiver），不能重复绑定到实参');
        let fallback = null;
        if (item.fallbackParam !== undefined && item.fallbackParam !== null) {
          if (typeof item.fallbackParam !== 'string' || !paramKeys.has(item.fallbackParam)) {
            fail(`绑定 ${i + 1} 的 fallbackParam「${item.fallbackParam}」不是已定义的参数 key`);
          }
          fallback = item.fallbackParam;
        }
        binding = { source: 'input', index: item.index, keyword, fallbackParam: fallback };
      } else if (item.source === 'param') {
        if (typeof item.key !== 'string' || !paramKeys.has(item.key)) fail(`绑定 ${i + 1} 引用的参数「${item.key}」未定义`);
        if (usedParams.has(item.key)) fail(`参数 ${item.key} 被绑定了多次`);
        usedParams.add(item.key);
        binding = { source: 'param', key: item.key, keyword, fallbackParam: null };
      } else {
        fail(`绑定 ${i + 1} 的 source 必须是 input 或 param`);
      }
      return binding;
    });

    /* 完整性：每个输入必须被消费（method 的输入 1 由 receiver 消费）；每个参数至少被使用一次。 */
    if (kind === 'method') {
      if (!inputs.length) fail('方法调用至少需要一个输入作为接收者');
      if (inputs[0].optional) fail('方法调用的接收者（输入 1）不能是可选输入');
      usedInputs.add(1);
    }
    for (let i = 1; i <= inputs.length; i++) {
      if (!usedInputs.has(i)) fail(`输入 ${i}（${inputs[i - 1].name}）没有被绑定到调用；每个输入都必须参与调用`);
    }
    for (const p of params) {
      const used = usedParams.has(p.key) || inputs.some(x => x.fallbackParam === p.key) || args.some(a => a.fallbackParam === p.key);
      if (!used) fail(`参数 ${p.key} 未被任何绑定使用`);
    }

    return { schema: SCHEMA, id: raw.id, version: raw.version, label, description, category, imports, output, inputs, params, call: { kind, target, args } };
  }

  function requireBool(value, field) {
    if (typeof value !== 'boolean') fail(`${field} 必须是 true 或 false`);
    return value;
  }
  function checkEnum(value, field, options) {
    if (typeof value !== 'string' || !options.includes(value)) fail(`${field}「${value}」只允许：${options.join('、')}`);
    return value;
  }
  function checkDefault(item, param, hooks) {
    const field = `参数 ${param.key} 默认值`;
    const def = item.def;
    switch (param.type) {
      case 'int': {
        if (typeof def === 'number') { if (!Number.isInteger(def)) fail(`${field} 必须是整数`); return String(def); }
        const text = renderInt(def, field);
        return text;
      }
      case 'number': {
        if (typeof def === 'number') {
          if (!Number.isFinite(def)) fail(`${field} 数值越界`);
          return String(def);
        }
        renderNumber(def, field);   // 校验；默认值保留用户原始写法（如 -inf），渲染时再转换
        return String(def).trim();
      }
      case 'bool': {
        if (typeof def !== 'boolean') fail(`${field} 必须是 true / false`);
        return def;
      }
      case 'string': return typeof def === 'string' ? checkText(def, field, 500, { min: 0 }) : fail(`${field} 必须是文本`);
      case 'select': {
        if (typeof def !== 'string' || !param.options.includes(def)) fail(`${field} 必须是 options 之一`);
        return def;
      }
      case 'literal': {
        if (typeof def !== 'string' || !def.trim() || def.trim().length > 500) fail(`${field} 必须是 1-500 字符的字面量源码`);
        const checked = hooks.literal ? hooks.literal(def.trim()) : null;
        if (checked && checked.error) fail(`${field}「${def.trim()}」无效：${checked.error}`);
        return def.trim();
      }
      default: return fail(`${field} 类型未知`);
    }
  }

  function canonicalJson(spec) {
    return JSON.stringify(normalize(spec));
  }
  function deepCopy(value) {
    return value === undefined ? value : JSON.parse(JSON.stringify(value));
  }
  function deepFreeze(value) {
    if (value && typeof value === 'object') {
      for (const key of Object.getOwnPropertyNames(value)) deepFreeze(value[key]);
      Object.freeze(value);
    }
    return value;
  }

  /* ---- 调用表达式生成（与 nodespecs.py render_call 逐字符一致） ----
   * inputExprs[i] 是输入 i+1 的表达式；null/undefined 表示未连线。
   * paramValues 缺省时使用规范默认值。 */
  function renderCall(spec, inputExprs, paramValues, literalResolver) {
    const values = {};
    for (const p of spec.params) values[p.key] = paramValues?.[p.key] !== undefined && paramValues?.[p.key] !== null && String(paramValues[p.key]) !== '' ? paramValues[p.key] : p.def;
    const paramExprs = {};
    for (const p of spec.params) paramExprs[p.key] = renderParamValue(p, values[p.key], literalResolver);
    const args = [];
    for (const binding of spec.call.args) {
      let expr = null;
      if (binding.source === 'input') {
        expr = inputExprs ? inputExprs[binding.index - 1] : null;
        if (expr === null || expr === undefined || expr === '') {
          if (binding.fallbackParam) expr = paramExprs[binding.fallbackParam];
          else fail(`输入 ${binding.index} 未连线且未配置参数回退（fallbackParam）`);
        }
      } else {
        expr = paramExprs[binding.key];
      }
      args.push(binding.keyword ? `${binding.keyword}=${expr}` : expr);
    }
    if (spec.call.kind === 'method') {
      const receiver = inputExprs ? inputExprs[0] : null;
      if (receiver === null || receiver === undefined || receiver === '') fail('接收者（输入 1）未连线');
      return `${receiver}.${spec.call.target}(${args.join(', ')})`;
    }
    return `${spec.call.target}(${args.join(', ')})`;
  }

  /* 制作器实时预览：in1/in2… 代表输入端口。 */
  function previewStatement(spec) {
    const names = spec.inputs.map((_, i) => `in${i + 1}`);
    try {
      return `out = ${renderCall(spec, names, null, null)}`;
    } catch (e) {
      if (e instanceof SpecError) return `# 预览不可用：${e.message}`;
      throw e;
    }
  }

  /* ---- 注册表：发布的定义深不可变；同 id+version 不同内容不可覆盖 ---- */
  function createRegistry() {
    const byEncoded = new Map();
    const byId = new Map();
    const registry = {
      has(encoded) { return byEncoded.has(String(encoded)); },
      get(encoded) { return byEncoded.get(String(encoded)) || null; },
      versions(id) { return [...(byId.get(id)?.keys() || [])]; },
      maxVersion(id) { return Math.max(0, ...(byId.get(id)?.keys() || [])); },
      nextVersion(id) { return registry.maxVersion(id) + 1; },
      list() { return [...byEncoded.values()]; },
      ids() { return [...byId.keys()]; },
      /* 严格发布：同 id+version 内容相同 → 幂等；不同 → VersionConflict。 */
      publish(raw, hooks) {
        const spec = normalize(raw, hooks);
        const encoded = encode(spec.id, spec.version);
        const existing = byEncoded.get(encoded);
        if (existing) {
          if (canonicalJson(existing) === canonicalJson(spec)) return { created: false, spec: existing, encoded };
          throw new VersionConflict(`版本冲突：${encoded} 已发布过不同内容，不能覆盖；请改用版本 ${registry.nextVersion(spec.id)} 或修改规范 id`);
        }
        deepFreeze(spec);
        if (!byId.has(spec.id)) byId.set(spec.id, new Map());
        byId.get(spec.id).set(spec.version, spec);
        byEncoded.set(encoded, spec);
        return { created: true, spec, encoded };
      },
      /* 更新式发布：内容冲突时自动递增版本，旧节点仍绑旧版本。 */
      publishUpdate(raw, hooks) {
        try {
          return registry.publish(raw, hooks);
        } catch (e) {
          if (!(e instanceof VersionConflict)) throw e;
          const probe = normalize(raw, hooks);
          return registry.publish({ ...probe, version: registry.nextVersion(probe.id) }, hooks);
        }
      },
      serialize() { return registry.list().map(deepCopy); },
      size: () => byEncoded.size,
    };
    return registry;
  }

  /* 图中引用了但注册表缺失的 custom 类型（加载时“保留原图报错禁止运行”的依据）。 */
  function scanMissing(graph, registry) {
    const missing = new Set();
    const scopes = [graph?.main, ...(Array.isArray(graph?.functions) ? graph.functions.map(f => f?.data) : [])];
    for (const scope of scopes) {
      if (!scope || typeof scope !== 'object') continue;
      for (const node of Object.values(scope)) {
        const type = node?.data?.type;
        if (isCustomType(type) && !registry.has(type)) missing.add(String(type));
      }
    }
    return [...missing];
  }

  /* ---- 规范 → 现有 NODE_DEFS 结构（复用参数表单/端口/摘要机制） ----
   * number 用文本框：要允许手填 inf / -inf / nan（number 输入框会拒绝这些词）。 */
  const FORM_TYPE = { int: 'int', number: 'text', bool: 'bool', string: 'text', literal: 'text', select: 'select' };
  function specToDef(spec, hooks = {}) {
    const def = {
      label: spec.label,
      defaults: { name: spec.id },
      inputs: spec.inputs.length, outputs: 1,
      inNames: spec.inputs.map(i => i.name + (i.optional ? ' (可选)' : '')),
      outNames: [spec.output.name],
      optionalInputs: spec.inputs.map((i, idx) => (i.optional ? idx : -1)).filter(i => i >= 0),
      params: spec.params.map(p => ({
        key: p.key, label: p.label, type: FORM_TYPE[p.type], def: p.def,
        help: p.help + (p.type === 'int' ? '（只接受整数）' : p.type === 'number' ? '（支持数字/inf/-inf/nan）' : p.type === 'literal' ? '（Python 字面量，经后端校验）' : ''),
        options: p.options, spectype: p.type,
      })),
      summary: d => describeCall(spec, d, hooks),
      spec,
    };
    return def;
  }
  /* 卡片参数摘要：输入用端口名占位，参数用已填值/默认值；渲染失败显示原因，不抛出。 */
  function describeCall(spec, data, hooks = {}) {
    try {
      const paramExpr = {};
      for (const p of spec.params) {
        const raw = data?.[p.key] !== undefined && String(data[p.key]) !== '' ? String(data[p.key]) : String(p.def ?? '');
        try { paramExpr[p.key] = renderParamValue(p, raw, hooks.literal); } catch (e) { paramExpr[p.key] = `「${raw.trim()}」`; }
      }
      const args = spec.call.args.map(b => {
        const expr = b.source === 'input' ? spec.inputs[b.index - 1].name : paramExpr[b.key];
        return b.keyword ? `${b.keyword}=${expr}` : expr;
      });
      const head = spec.call.kind === 'method' ? `${spec.inputs[0].name}.${spec.call.target}` : spec.call.target;
      return `${head}(${args.join(', ')})`;
    } catch (e) {
      return e instanceof SpecError ? e.message : '';
    }
  }

  /* ---- 内置模板：可一键载入制作器的 4 个示例 ---- */
  const TEMPLATES = {
    sqrt: {
      schema: 1, id: 'sqrt', version: 1, label: '平方根 sqrt', description: '逐元素开平方 torch.sqrt(x)，输入必须非负。',
      category: '自定义', output: { name: 'out', type: 'Tensor' },
      inputs: [{ name: 'x', type: 'Tensor', optional: false, fallbackParam: null, help: '非负张量' }],
      params: [],
      call: { kind: 'function', target: 'torch.sqrt', args: [{ source: 'input', index: 1, keyword: null, fallbackParam: null }] },
    },
    sin: {
      schema: 1, id: 'sin', version: 1, label: '正弦 sin', description: '逐元素正弦 torch.sin(x)，弧度制。',
      category: '自定义', output: { name: 'out', type: 'Tensor' },
      inputs: [{ name: 'x', type: 'Tensor', optional: false, fallbackParam: null, help: '任意形状张量' }],
      params: [],
      call: { kind: 'function', target: 'torch.sin', args: [{ source: 'input', index: 1, keyword: null, fallbackParam: null }] },
    },
    triu: {
      schema: 1, id: 'triu', version: 1, label: '上三角 triu', description: 'torch.triu(input, diagonal=…)：保留对角线及上方；diagonal 可连线覆盖（连线优先）。',
      category: '自定义', output: { name: 'out', type: 'Tensor' },
      inputs: [
        { name: 'input', type: 'Tensor', optional: false, fallbackParam: null, help: '至少二维张量，作用于最后两个维度' },
        { name: 'diagonal', type: 'int', optional: true, fallbackParam: 'diagonal', help: '可连线 Literal 整数，未连线时用参数值' },
      ],
      params: [{ key: 'diagonal', label: 'diagonal', type: 'int', def: '0', help: '0 保留主对角线及上方；1 只保留严格上三角' }],
      call: {
        kind: 'function', target: 'torch.triu',
        args: [
          { source: 'input', index: 1, keyword: null, fallbackParam: null },
          { source: 'input', index: 2, keyword: 'diagonal', fallbackParam: 'diagonal' },
        ],
      },
    },
    masked_fill: {
      schema: 1, id: 'masked_fill', version: 1, label: '掩码填充 masked_fill', description: 'x.masked_fill(mask, value=…)：mask 为 True 的位置替换为 value；演示 Tensor 方法 + 接收者 + 关键字参数回退。',
      category: '自定义', output: { name: 'out', type: 'Tensor' },
      inputs: [
        { name: 'input', type: 'Tensor', optional: false, fallbackParam: null, help: '接收者：被填充的张量' },
        { name: 'mask', type: 'BoolTensor', optional: false, fallbackParam: null, help: 'bool 掩码，可广播' },
        { name: 'value', type: 'number', optional: true, fallbackParam: 'value', help: '可连线 Literal 数字；未连线时用参数值' },
      ],
      params: [{ key: 'value', label: 'value', type: 'number', def: '-inf', help: 'mask 为 True 的位置替换为此标量；支持数字、inf、-inf、nan' }],
      call: {
        kind: 'method', target: 'masked_fill',
        args: [
          { source: 'input', index: 2, keyword: null, fallbackParam: null },
          { source: 'input', index: 3, keyword: 'value', fallbackParam: 'value' },
        ],
      },
    },
  };

  const NS = {
    SCHEMA, MAX_INPUTS, MAX_PARAMS, MAX_ARGS, MAX_SELECT_OPTIONS, MAX_IMPORTS,
    FUNCTION_TARGETS, METHOD_TARGETS, PARAM_TYPES, INPUT_TYPE_HINTS,
    PY_TRUE_KEYWORDS, DEFAULT_IMPORT_NAMES,
    encode, parseType, isCustomType,
    SpecError, VersionConflict,
    checkPyName, checkDottedTarget, normalizeImports,
    renderNumber, renderInt, renderString, renderParamValue, renderCall,
    normalize, canonicalJson, deepCopy, deepFreeze,
    createRegistry, scanMissing, specToDef, previewStatement, describeCall,
    TEMPLATES,
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = NS;
  if (typeof global !== 'undefined') global.NS = NS;
})(typeof window !== 'undefined' ? window : globalThis);
