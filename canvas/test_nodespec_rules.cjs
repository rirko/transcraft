/* NodeSpec 纯核心规则测试（node test_nodespec_rules.cjs）。
 * 覆盖：模板合法性、自由目标语法校验（任意合法点分标识符接受，注入语法拒绝）、
 * 声明式 imports（去重/默认 torch+nn 去重/冲突拒绝）、无参数方法 item/tolist、
 * 错误参数绑定拒绝、参数渲染共享规则、输出类型提示扩展、
 * 注册表不可变性与版本更新、缺失规范扫描。
 * `node test_nodespec_rules.cjs dump` 输出规则 JSON，供 test_nodespec_backend.py 做两侧一致性比对。
 * 全程内存对象，不接触任何项目数据。 */
const assert = require('assert');
const path = require('path');
const NS = require(path.join(__dirname, 'static', 'nodespec.js'));

/* 后端 /api/literal 的测试替身：与 ast.literal_eval 白名单同规则的最小子集。 */
const literalOk = source => {
  try {
    const value = JSON.parse(source.replace(/'/g, '"')
      .replace(/\bTrue\b/g, 'true').replace(/\bFalse\b/g, 'false').replace(/\bNone\b/g, 'null'));
    return { expression: source, type: typeof value };
  } catch (e) { return { error: 'invalid literal' }; }
};

function rejects(fn, pattern) {
  assert.throws(fn, e => e instanceof NS.SpecError && (!pattern || pattern.test(e.message)),
    `expected SpecError${pattern ? ' /' + pattern.source + '/' : ''}`);
}

/* ---- 模板：四个模板全部合法且可渲染 ---- */
for (const [name, template] of Object.entries(NS.TEMPLATES)) {
  const spec = NS.normalize(template, { literal: literalOk });
  assert.equal(spec.id, name);
  assert.equal(spec.schema, 1);
  assert(spec.output && spec.inputs.length >= 1);
}
assert.equal(NS.renderCall(NS.TEMPLATES.sqrt, ['x']), 'torch.sqrt(x)');
assert.equal(NS.renderCall(NS.TEMPLATES.sin, ['t']), 'torch.sin(t)');
assert.equal(NS.renderCall(NS.TEMPLATES.triu, ['x', '2']), 'torch.triu(x, diagonal=2)');
assert.equal(NS.renderCall(NS.TEMPLATES.triu, ['x', null], { diagonal: '1' }), 'torch.triu(x, diagonal=1)');
assert.equal(NS.renderCall(NS.TEMPLATES.masked_fill, ['x', 'm', '-2.5']), 'x.masked_fill(m, value=-2.5)');
assert.equal(NS.renderCall(NS.TEMPLATES.masked_fill, ['x', 'm', null], {}, literalOk),
  'x.masked_fill(m, value=float("-inf"))');
console.log('PASS: 四个模板合法，函数/方法/连线优先渲染正确');

/* ---- 自由目标：任意合法点分标识符接受（真开放，无隐藏 torch.* 白名单）----
 * 信任边界：这些目标会真实执行本机 Python（含 os.system / torch.load / __import__），
 * 规范层只做语法校验，不做权限裁剪；副作用由用户自己负责。 */
const base = () => NS.deepCopy(NS.TEMPLATES.sqrt);
const fnCall = target => ({ kind: 'function', target, args: [{ source: 'input', index: 1, keyword: null, fallbackParam: null }] });
for (const target of ['math.sqrt', 'builtins.len', 'torch.sqrt', 'torch.load', 'torch.save',
  'os.system', 'subprocess.run', 'torch._C._nn.gelu', 'torch.utils.data', 'torch.nn.functional.softmax',
  'eval', 'exec', '__import__', 'compile', 'globals', 'open', '_private.name', 'A.Uppercase.seg']) {
  const draft = base();
  draft.call = fnCall(target);
  const spec = NS.normalize(draft);
  assert.equal(spec.call.target, target, `自由目标 ${target} 必须原样保留`);
}
console.log('PASS: 任意合法点分标识符目标全部接受（math/builtins/普通模块/双下划线/大写段，无隐藏白名单）');

/* ---- 目标语法校验：非法语法一律拒绝（防拼接额外语句），与危险性无关 ---- */
for (const target of ['torch.*', 'sqrt.', '.sqrt', 'torch..sqrt', 'a b', 'a;b', 'a\nb', 'eval(x',
  'x) + y', '2bad', 'a-b', 'a.class.b', 'a.import.b', 'out=x', 'f()', 'a.b[0]', 'import', 'None',
  'True', 'and', 42, null, '']) {
  const draft = base();
  draft.call = fnCall(target);
  rejects(() => NS.normalize(draft));
}
console.log('PASS: 目标语法校验（空段/通配符/空格/分号/换行/括号/下标/保留字段/非字符串全部拒绝）');

/* ---- method 目标：单个合法方法名；无参数方法（item/tolist）允许零绑定 ---- */
const methodDraft = (target, args) => {
  const draft = base();
  if (args.length) draft.inputs.push({ name: 'm', type: 'BoolTensor', optional: false, fallbackParam: null, help: '' });
  draft.call = { kind: 'method', target, args };
  return draft;
};
for (const target of ['item', 'tolist', 'masked_fill', 'relu_', 'fill_', '__setattr__', 'numpy', 'cpu']) {
  const args = target === 'item' || target === 'tolist' ? [] : [{ source: 'input', index: 2, keyword: null, fallbackParam: null }];
  const spec = NS.normalize(methodDraft(target, args));
  assert.equal(spec.call.target, target);
}
assert.equal(NS.renderCall(NS.normalize(methodDraft('item', [])), ['x']), 'x.item()');
assert.equal(NS.renderCall(NS.normalize(methodDraft('tolist', [])), ['x']), 'x.tolist()');
for (const target of ['a.b', 'class', 'for', 'item(', 'x y', '']) {
  rejects(() => NS.normalize(methodDraft(target, [{ source: 'input', index: 2, keyword: null, fallbackParam: null }])),
    new RegExp(target.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
}
/* 函数 kind 仍要求至少一个绑定；method 可以只靠接收者 */
rejects(() => NS.normalize({ ...base(), call: { kind: 'function', target: 'math.sqrt', args: [] } }), /至少需要一个绑定/);
console.log('PASS: 方法目标语法校验 + 无参数 item/tolist 零绑定渲染 x.item() / x.tolist()');

/* ---- 错误参数绑定拒绝 ---- */
rejects(() => NS.normalize({ ...base(), inputs: [], call: { kind: 'function', target: 'torch.sqrt', args: [{ source: 'input', index: 1 }] } }), /序号超出范围/);
rejects(() => NS.normalize({ ...base(), call: { kind: 'function', target: 'torch.sqrt', args: [{ source: 'input', index: 1 }, { source: 'input', index: 1 }] } }), /多次/);
rejects(() => NS.normalize({ ...base(), call: { kind: 'function', target: 'torch.sqrt', args: [{ source: 'input', index: 2 }] } }), /序号超出范围/);
rejects(() => NS.normalize({ ...base(), call: { kind: 'function', target: 'torch.sqrt', args: [{ source: 'param', key: 'nope' }] } }), /未定义/);
rejects(() => NS.normalize({ ...base(), call: { kind: 'function', target: 'torch.sqrt', args: [{ source: 'input', index: 1, keyword: 'for' }] } }), /保留字/);
rejects(() => NS.normalize({ ...base(), call: { kind: 'function', target: 'torch.sqrt', args: [{ source: 'input', index: 1, keyword: '1bad' }] } }), /关键字名/);
rejects(() => NS.normalize({ ...base(), call: { kind: 'function', target: 'torch.sqrt', args: [{ source: 'input', index: 1, fallbackParam: 'ghost' }] } }), /fallbackParam/);
rejects(() => NS.normalize({ ...base(), call: { kind: 'function', target: 'torch.sqrt', args: [] } }), /至少需要一个绑定/);
rejects(() => NS.normalize({ ...base(), inputs: [], call: { kind: 'function', target: 'torch.sqrt', args: [{ source: 'input', index: 1 }] } }), /序号超出范围|没有被绑定/);
/* method：输入 1 是 receiver，不得重复绑定；receiver 不可选 */
rejects(() => NS.normalize({ ...base(), call: { kind: 'method', target: 'masked_fill', args: [{ source: 'input', index: 1 }] } }), /接收者/);
const optionalReceiver = base();
optionalReceiver.inputs[0].optional = true;
optionalReceiver.inputs.push({ name: 'm', type: 'BoolTensor', optional: false, fallbackParam: null, help: '' });
optionalReceiver.call = { kind: 'method', target: 'masked_fill', args: [{ source: 'input', index: 2, keyword: null, fallbackParam: null }] };
rejects(() => NS.normalize(optionalReceiver), /接收者.*不能是可选输入/);
/* 未被调用的输入/参数必须报错 */
rejects(() => NS.normalize({ ...base(), params: [{ key: 'k', label: 'k', type: 'int', def: '0', help: '' }] }),
  /参数 k 未被任何绑定使用/);
const unboundInput = base();
unboundInput.inputs.push({ name: 'y', type: 'Tensor', optional: false, fallbackParam: null, help: '' });
rejects(() => NS.normalize(unboundInput), /输入 2（y）没有被绑定/);
/* 未知字段 / 非法身份 */
rejects(() => NS.normalize({ ...base(), extra: 1 }), /未知字段/);
rejects(() => NS.normalize({ ...base(), schema: 2 }), /schema/);
rejects(() => NS.normalize({ ...base(), id: 'Bad' }), /规范 id/);
rejects(() => NS.normalize({ ...base(), version: 0 }), /版本号/);
rejects(() => NS.normalize({ ...base(), label: '<img>' }), /显示名称/);
rejects(() => NS.normalize({ ...base(), output: { name: 'out', type: 'Dict' } }), /输出类型/);
/* 输出/输入类型提示扩展：string/bool/list/tuple/dict（只是提示，不转换；旧图默认 Tensor 不变） */
for (const hint of ['string', 'bool', 'list', 'tuple', 'dict']) {
  assert.equal(NS.normalize({ ...base(), output: { name: 'out', type: hint } }).output.type, hint);
  const draft = base();
  draft.inputs[0].type = hint;
  assert.equal(NS.normalize(draft).inputs[0].type, hint);
}
assert.equal(NS.normalize(base()).output.type, 'Tensor', '旧规范无扩展类型时默认 Tensor 不变');
console.log('PASS: 错误参数绑定 / receiver 规则 / 未知字段 / 非法身份全部拒绝；输出类型提示扩展并兼容旧图');

/* ---- 声明式 imports：合法列表、去重、默认 torch/nn 去重、冲突拒绝 ---- */
const withImports = imports => NS.normalize({ ...base(), imports });
/* 合法：math / numpy as np / 包.子模块 */
let spec = withImports([{ module: 'math' }, { module: 'numpy', alias: 'np' }, { module: 'collections.abc' }]);
assert.deepEqual(spec.imports, [{ module: 'math', alias: null }, { module: 'numpy', alias: 'np' }, { module: 'collections.abc', alias: null }]);
/* 完全相同条目去重 */
assert.equal(withImports([{ module: 'math' }, { module: 'math' }]).imports.length, 1);
/* 默认 torch / torch.nn as nn 已存在 → 丢弃重复声明；裸 torch.nn（绑定名 torch）也去重 */
assert.deepEqual(withImports([{ module: 'torch' }, { module: 'torch.nn', alias: 'nn' }, { module: 'torch.nn' }, { module: 'math' }]).imports,
  [{ module: 'math', alias: null }]);
/* 无 imports 字段的旧规范兼容：canonical imports 为空数组 */
assert.deepEqual(NS.normalize(base()).imports, []);
/* 冲突：别名/绑定名与 torch、nn 撞 → 拒绝 */
for (const bad of [
  [{ module: 'numpy', alias: 'torch' }], [{ module: 'numpy', alias: 'nn' }],
  [{ module: 'torch.nn', alias: 'torch' }], [{ module: 'math', alias: 'nn' }],
]) { rejects(() => withImports(bad), /与默认导入 torch\/nn 冲突/); }
/* 两条导入绑定同名 → 拒绝 */
rejects(() => withImports([{ module: 'json', alias: 'np' }, { module: 'numpy', alias: 'np' }]), /导入绑定名「np」冲突/);
rejects(() => withImports([{ module: 'math' }, { module: 'math.abc' }]), /导入绑定名「math」冲突/, 'import math 与 import math.abc 都绑定 math');
/* 语法：模块名/别名的注入面（分号/空格/关键字/未知字段/非数组/超量） */
for (const bad of [
  [{ module: 'os; import os' }], [{ module: 'x y' }], [{ module: 'm}' }], [{ module: 'a', alias: 'for' }],
  [{ module: 'a', alias: '1x' }], [{ module: 'a', extra: 1 }], [{ module: 'torch', alias: 't' }, { module: 'json', alias: 't' }, { module: 'math', alias: 't' }],
  'nope', [{ module: 'a' }, { module: 'b' }, { module: 'c' }, { module: 'd' }, { module: 'e' }, { module: 'f' }, { module: 'g' }, { module: 'h' }, { module: 'i' }],
]) { rejects(() => withImports(bad)); }
/* imports 是规范内容的一部分：改动 imports = 不同内容 → 版本冲突 */
{
  const registry2 = NS.createRegistry();
  registry2.publish(base(), { literal: literalOk });
  assert.throws(() => registry2.publish({ ...base(), imports: [{ module: 'math' }] }), e => e instanceof NS.VersionConflict,
    '同 id+version 改 imports 必须版本冲突');
}
console.log('PASS: 声明式 imports（合法/去重/默认 torch+nn 去重/别名冲突/绑定名冲突/注入拒绝/版本冲突）');

/* ---- 参数渲染共享规则（与 nodespecs.py render_param_value 一致） ---- */
const p = (type, def, extra = {}) => ({ key: 'k', label: 'k', type, def, options: extra.options, help: '' });
assert.equal(NS.renderParamValue(p('int', '3'), undefined, null), '3');
assert.equal(NS.renderParamValue(p('int', '3'), '-12'), '-12');
rejects(() => NS.renderParamValue(p('int', '3'), '2.5'), /整数/);
assert.equal(NS.renderParamValue(p('number', '1.5'), undefined), '1.5');
assert.equal(NS.renderParamValue(p('number', '1.5'), 'inf'), 'float("inf")');
assert.equal(NS.renderParamValue(p('number', '1.5'), '-inf'), 'float("-inf")');
assert.equal(NS.renderParamValue(p('number', '1.5'), 'nan'), 'float("nan")');
assert.equal(NS.renderParamValue(p('number', '1.5'), '1e-3'), '1e-3');
rejects(() => NS.renderParamValue(p('number', '1.5'), 'abc'), /不是合法数字/);
rejects(() => NS.renderParamValue(p('number', '1.5'), '1e999'), /越界/);
assert.equal(NS.renderParamValue(p('bool', false), undefined), 'False');
assert.equal(NS.renderParamValue(p('bool', true), 'true'), 'True');
rejects(() => NS.renderParamValue(p('bool', true), 'yes'), /布尔/);
assert.equal(NS.renderParamValue(p('string', 'hi'), undefined), '"hi"');
assert.equal(NS.renderParamValue(p('string', 'hi'), '他"说"'), '"他\\"说\\""');
assert.equal(NS.renderParamValue(p('select', 'a', { options: ['a', 'b'] }), 'b'), '"b"');
rejects(() => NS.renderParamValue(p('select', 'a', { options: ['a', 'b'] }), 'c'), /不在可选项/);
assert.equal(NS.renderParamValue(p('literal', '[1, 2]'), undefined, literalOk), '[1, 2]');
rejects(() => NS.renderParamValue(p('literal', '[1, 2]'), undefined, null), /尚未通过 Literal 校验/);
rejects(() => NS.renderParamValue(p('literal', '[1, 2]'), '1+2', literalOk), /无效/);
console.log('PASS: 参数渲染规则（int/number/inf/bool/string/select/literal）两侧可对齐');

/* ---- 注册表：发布不可变、幂等、版本冲突、更新产生版本 2 ---- */
const registry = NS.createRegistry();
for (const template of Object.values(NS.TEMPLATES)) registry.publish(template, { literal: literalOk });
assert.equal(registry.size(), 4);
assert.equal(registry.publish(NS.TEMPLATES.sqrt, { literal: literalOk }).created, false, '幂等发布');
const conflictDraft = { ...NS.deepCopy(NS.TEMPLATES.sqrt), label: '改过的 sqrt' };
assert.throws(() => registry.publish(conflictDraft), e => e instanceof NS.VersionConflict, '同 id+version 不同内容必须冲突');
assert.throws(() => registry.publish({ ...conflictDraft, description: 'x' }), e => e instanceof NS.VersionConflict, '任意字段差异都算冲突');
const updated = registry.publishUpdate(conflictDraft, { literal: literalOk });
assert.equal(updated.encoded, 'custom.sqrt@2');
assert.equal(updated.created, true);
assert.equal(registry.get('custom.sqrt@1').label, '平方根 sqrt', '旧节点绑定的版本 1 保持原内容');
assert.equal(registry.get('custom.sqrt@2').label, '改过的 sqrt');
assert.equal(registry.nextVersion('sqrt'), 3);
assert(Object.isFrozen(registry.get('custom.sqrt@1')) && Object.isFrozen(registry.get('custom.sqrt@1').inputs)
  && Object.isFrozen(registry.get('custom.sqrt@1').call.args), '发布定义深不可变');
assert.throws(() => { 'use strict'; registry.get('custom.sqrt@1').id = 'hacked'; }, TypeError);
assert.deepEqual(registry.versions('sqrt'), [1, 2]);
/* serialize → 全新注册表可重建且引用闭合 */
const restored = NS.createRegistry();
for (const spec of registry.serialize()) restored.publish(spec, { literal: literalOk });
assert.equal(restored.size(), 5);
assert.equal(restored.get('custom.sqrt@2').label, '改过的 sqrt');
console.log('PASS: 注册表幂等/冲突/版本 2/深冻结/序列化重建');

/* ---- 缺失规范扫描（保留原图报错禁止运行的依据） ---- */
const graph = { main: { 1: { data: { type: 'custom.sqrt@1' } }, 2: { data: { type: 'literal' } } },
  functions: [{ name: 'M', data: { 3: { data: { type: 'custom.ghost@7' } } } }] };
assert.deepEqual(NS.scanMissing(graph, registry), ['custom.ghost@7']);
assert.deepEqual(NS.scanMissing({ main: {}, functions: [] }, NS.createRegistry()), []);
assert.equal(NS.isCustomType('custom.a@1'), true);
assert.equal(NS.isCustomType('literal'), false);
assert.deepEqual(NS.parseType('custom.my_op_2@12'), { id: 'my_op_2', version: 12 });
assert.equal(NS.parseType('custom.A@1'), null);
console.log('PASS: 缺失规范扫描与类型编码解析');

/* ---- dump 模式：供 Python 侧做共享规则一致性比对 ---- */
if (process.argv[2] === 'dump') {
  const spec = NS.normalize({ ...NS.TEMPLATES.masked_fill, params: [{ key: 'value', label: 'value', type: 'number', def: '-inf', help: '' }] }, { literal: literalOk });
  const greet = { ...NS.TEMPLATES.sqrt, id: 'greet', label: 'greet',
    params: [
      { key: 'sep', label: 'sep', type: 'string', def: '他说"hi"', help: '' },
      { key: 'flag', label: 'flag', type: 'bool', def: true, help: '' },
      { key: 'mode', label: 'mode', type: 'select', def: 'fast', options: ['fast', 'slow'], help: '' },
      { key: 'bias', label: 'bias', type: 'literal', def: '[1, 2.5, "x"]', help: '' },
    ],
    call: { kind: 'function', target: 'torch.sqrt', args: [
      { source: 'input', index: 1, keyword: null, fallbackParam: null },
      { source: 'param', key: 'sep', keyword: null, fallbackParam: null },
      { source: 'param', key: 'flag', keyword: 'flag', fallbackParam: null },
      { source: 'param', key: 'mode', keyword: 'mode', fallbackParam: null },
      { source: 'param', key: 'bias', keyword: 'bias', fallbackParam: null },
    ] } };
  const fnArg = [{ source: 'input', index: 1, keyword: null, fallbackParam: null }];
  const decisions = [];
  const attempt = draft => {
    try { NS.normalize(draft, { literal: literalOk }); return 'ok'; }
    catch (e) { return e instanceof NS.SpecError ? 'rejected' : 'threw'; }
  };
  const good = NS.TEMPLATES.triu;
  decisions.push(['triu-template', attempt(good)]);
  /* 自由目标：合法语法一律接受（不再有白名单），非法语法一律拒绝 */
  decisions.push(['free-target-math', attempt({ ...NS.TEMPLATES.sqrt, call: { kind: 'function', target: 'math.sqrt', args: fnArg } })]);
  decisions.push(['free-target-torch-load', attempt({ ...NS.TEMPLATES.sqrt, call: { kind: 'function', target: 'torch.load', args: fnArg } })]);
  decisions.push(['free-target-subprocess', attempt({ ...NS.TEMPLATES.sqrt, call: { kind: 'function', target: 'subprocess.run', args: fnArg } })]);
  decisions.push(['wildcard-target', attempt({ ...NS.TEMPLATES.sqrt, call: { kind: 'function', target: 'torch.*', args: fnArg } })]);
  decisions.push(['target-trailing-dot', attempt({ ...NS.TEMPLATES.sqrt, call: { kind: 'function', target: 'sqrt.', args: fnArg } })]);
  decisions.push(['target-paren-injection', attempt({ ...NS.TEMPLATES.sqrt, call: { kind: 'function', target: 'eval(x', args: fnArg } })]);
  decisions.push(['target-semicolon-injection', attempt({ ...NS.TEMPLATES.sqrt, call: { kind: 'function', target: 'os.system; import os', args: fnArg } })]);
  decisions.push(['target-keyword-segment', attempt({ ...NS.TEMPLATES.sqrt, call: { kind: 'function', target: 'a.class.b', args: fnArg } })]);
  /* 方法：单个方法名；无参数 item 允许零绑定；点分/保留字/in-place 全部按语法判定 */
  decisions.push(['method-inplace-allowed', attempt({ ...NS.TEMPLATES.sqrt, call: { kind: 'method', target: 'relu_', args: [] } })]);
  decisions.push(['method-item-noargs-ok', attempt({ ...NS.TEMPLATES.sqrt, call: { kind: 'method', target: 'item', args: [] } })]);
  decisions.push(['method-dotted-rejected', attempt({ ...NS.TEMPLATES.sqrt, call: { kind: 'method', target: 'a.b', args: [] } })]);
  decisions.push(['method-keyword-rejected', attempt({ ...NS.TEMPLATES.sqrt, call: { kind: 'method', target: 'class', args: [] } })]);
  /* imports：合法声明 / 默认去重 / 冲突拒绝 */
  decisions.push(['imports-math-ok', attempt({ ...NS.TEMPLATES.sqrt, imports: [{ module: 'math', alias: null }] })]);
  decisions.push(['imports-default-dedupe', attempt({ ...NS.TEMPLATES.sqrt, imports: [{ module: 'torch', alias: null }, { module: 'torch.nn', alias: 'nn' }, { module: 'numpy', alias: 'np' }] })]);
  decisions.push(['imports-alias-conflict', attempt({ ...NS.TEMPLATES.sqrt, imports: [{ module: 'json', alias: 'np' }, { module: 'numpy', alias: 'np' }] })]);
  decisions.push(['imports-torch-alias-rejected', attempt({ ...NS.TEMPLATES.sqrt, imports: [{ module: 'numpy', alias: 'torch' }] })]);
  decisions.push(['imports-injection-rejected', attempt({ ...NS.TEMPLATES.sqrt, imports: [{ module: 'os; import os', alias: null }] })]);
  decisions.push(['dup-input', attempt({ ...NS.TEMPLATES.triu, call: { kind: 'function', target: 'torch.triu', args: [{ source: 'input', index: 1, keyword: null, fallbackParam: null }, { source: 'input', index: 1, keyword: 'diagonal', fallbackParam: null }] } })]);
  decisions.push(['keyword-reserved', attempt({ ...NS.TEMPLATES.triu, params: [{ key: 'for', label: 'f', type: 'int', def: '0', help: '' }], inputs: [NS.TEMPLATES.triu.inputs[0], { ...NS.TEMPLATES.triu.inputs[1], fallbackParam: 'for' }], call: { kind: 'function', target: 'torch.triu', args: [{ source: 'input', index: 1 }, { source: 'input', index: 2, keyword: 'for', fallbackParam: 'for' }] } })]);
  decisions.push(['missing-binding', attempt({ ...NS.TEMPLATES.triu, call: { kind: 'function', target: 'torch.triu', args: [{ source: 'input', index: 1 }] } })]);
  const renderings = [];
  const ren = (name, specObj, inputs, params) => {
    try { renderings.push([name, NS.renderCall(NS.normalize(specObj, { literal: literalOk }), inputs, params, literalOk)]); }
    catch (e) { renderings.push([name, 'ERROR: ' + e.message]); }
  };
  ren('sqrt', NS.TEMPLATES.sqrt, ['x'], {});
  ren('triu-connected', NS.TEMPLATES.triu, ['x', '2'], {});
  ren('triu-fallback', NS.TEMPLATES.triu, ['x', null], { diagonal: '1' });
  ren('masked-fill', spec, ['x', 'm', '0.5'], {});
  ren('masked-fill-inf', spec, ['x', 'm', null], {});
  ren('greet-params', greet, ['x'], {});
  ren('free-math', { ...NS.TEMPLATES.sqrt, call: { kind: 'function', target: 'math.sqrt', args: fnArg } }, ['x'], {});
  ren('method-item', { ...NS.TEMPLATES.sqrt, call: { kind: 'method', target: 'item', args: [] } }, ['x'], {});
  ren('method-tolist', { ...NS.TEMPLATES.sqrt, call: { kind: 'method', target: 'tolist', args: [] } }, ['x'], {});
  ren('number-bad', { ...NS.TEMPLATES.masked_fill, params: [{ key: 'value', label: 'value', type: 'number', def: 'abc', help: '' }] }, ['x', 'm', null], {});
  console.log(JSON.stringify({ functionTargets: NS.FUNCTION_TARGETS, methodTargets: NS.METHOD_TARGETS,
    paramTypes: NS.PARAM_TYPES, inputTypeHints: NS.INPUT_TYPE_HINTS, decisions, renderings }, null, 1));
  return;
}
console.log('ALL PASS: nodespec.js 核心规则');
