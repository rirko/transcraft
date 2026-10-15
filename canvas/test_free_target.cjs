/* NodeSpec 自由目标 + 声明式 imports 隔离回归（node test_free_target.cjs）。
 * 覆盖：任意合法函数目标不再拒绝（math/builtins/普通模块/多级子模块/大写段）、
 * 注入语法拒绝（分号/换行/括号/通配/保留字段）、无参数方法 item/tolist、
 * imports 别名/去重/默认 torch+nn 去重/冲突拒绝、旧规范（无 imports 字段）兼容、
 * 输出类型提示扩展且旧图默认 Tensor、注册表 imports 变化 = 新内容（版本冲突）、
 * 序列化/恢复往返保留 imports。
 * 全程内存对象，不接触任何项目数据；Python 侧镜像由 test_nodespec_backend.py 比对。 */
const assert = require('assert');
const path = require('path');
const NS = require(path.join(__dirname, 'static', 'nodespec.js'));

/* 后端 /api/literal 的测试替身（与 test_nodespec_rules.cjs 同规则）。 */
const literalOk = source => {
  try {
    JSON.parse(source.replace(/'/g, '"')
      .replace(/\bTrue\b/g, 'true').replace(/\bFalse\b/g, 'false').replace(/\bNone\b/g, 'null'));
    return { expression: source, type: 'stub' };
  } catch (e) { return { error: 'invalid literal' }; }
};
const draft = over => ({ ...NS.deepCopy(NS.TEMPLATES.sqrt), ...over });
const fnCall = target => ({ kind: 'function', target, args: [{ source: 'input', index: 1, keyword: null, fallbackParam: null }] });

/* ---- 1. 任意合法函数目标不再拒绝（真开放，无隐藏 torch.* 白名单） ---- */
for (const target of ['math.sqrt', 'builtins.len', 'torch.sqrt', 'os.path.join', 'numpy.linalg.norm',
  'torch.nn.functional.scaled_dot_product_attention', 'collections.OrderedDict', 'subprocess.run',
  'torch.load', 'json.dumps', '_private._helper', 'MyPkg.Module.Func', 'a' + '.b'.repeat(3)]) {
  const spec = NS.normalize(draft({ call: fnCall(target) }), { literal: literalOk });
  assert.equal(spec.call.target, target);
}
/* 长目标边界：16 段 / 200 字符以内合法，超出拒绝 */
assert.equal(NS.normalize(draft({ call: fnCall(Array(16).fill('a').join('.')) })).call.target.split('.').length, 16);
assert.throws(() => NS.normalize(draft({ call: fnCall(Array(17).fill('a').join('.')) })), e => e instanceof NS.SpecError);
assert.throws(() => NS.normalize(draft({ call: fnCall('a.'.repeat(100) + 'a') })), e => e instanceof NS.SpecError);
console.log('PASS: 任意合法点分标识符目标全部接受（math/builtins/普通模块/子模块/大写/下划线开头）');

/* ---- 2. 注入语法拒绝：不可能拼出额外语句 ---- */
for (const target of ['os.system; import os', 'f(); print(1)', 'a\nb', 'eval(x', 'x)', 'torch.*',
  'a b', 'a..b', '.a', 'a.', 'a.class.b', '2f', '-a', 'a#b', 'a"b', 'out=x', 'f(1)', 'x[0]', '']) {
  assert.throws(() => NS.normalize(draft({ call: fnCall(target) })), e => e instanceof NS.SpecError,
    `目标「${target}」必须被拒绝`);
}
/* 方法名：单标识符；item/tolist 零绑定 */
for (const target of ['a.b', 'class', 'lambda', 'item(', 'x y']) {
  assert.throws(() => NS.normalize(draft({ call: { kind: 'method', target, args: [{ source: 'input', index: 1, keyword: null, fallbackParam: null }] } })),
    e => e instanceof NS.SpecError);
}
for (const [target, expected] of [['item', 'in1.item()'], ['tolist', 'in1.tolist()']]) {
  const spec = NS.normalize(draft({ call: { kind: 'method', target, args: [] } }), { literal: literalOk });
  assert.equal(NS.renderCall(spec, ['in1']), expected);
  assert.equal(NS.previewStatement(spec), `out = ${expected}`);
}
assert.throws(() => NS.normalize(draft({ call: { kind: 'function', target: 'math.sqrt', args: [] } })),
  /至少需要一个绑定/, '函数目标仍要求至少一个绑定');
console.log('PASS: 注入语法（分号/换行/括号/通配/保留字段/点号边界）全部拒绝；无参数方法渲染正确');

/* ---- 3. 声明式 imports：普通模块与别名 / 去重 / 默认 torch+nn 去重 / 冲突拒绝 ---- */
const imports = list => NS.normalize(draft({ imports: list }), { literal: literalOk }).imports;
assert.deepEqual(imports([{ module: 'math' }, { module: 'numpy', alias: 'np' }, { module: 'xml.etree.ElementTree' }]),
  [{ module: 'math', alias: null }, { module: 'numpy', alias: 'np' }, { module: 'xml.etree.ElementTree', alias: null }]);
assert.deepEqual(imports([{ module: 'math' }, { module: 'math', alias: null }]), [{ module: 'math', alias: null }], '完全相同条目去重');
assert.deepEqual(imports([{ module: 'torch' }, { module: 'torch.nn', alias: 'nn' }, { module: 'torch.nn' }, { module: 'math' }]),
  [{ module: 'math', alias: null }], '默认 torch / torch.nn as nn / 裸 torch.nn 绑定名 torch 全部去重');
assert.deepEqual(imports([{ module: 'torch', alias: 't' }]), [{ module: 'torch', alias: 't' }], 'import torch as t 不与默认冲突');
assert.deepEqual(imports([{ module: 'a', alias: '' }]), [{ module: 'a', alias: null }], '空字符串别名按无别名处理');
for (const bad of [[{ module: 'numpy', alias: 'torch' }], [{ module: 'numpy', alias: 'nn' }],
  [{ module: 'json', alias: 'np' }, { module: 'numpy', alias: 'np' }], [{ module: 'math' }, { module: 'math.abc' }],
  [{ module: 'os; import os' }], [{ module: 'a', alias: 'for' }], ['math']]) {
  assert.throws(() => imports(bad), e => e instanceof NS.SpecError, `imports ${JSON.stringify(bad)} 必须被拒绝`);
}
assert.equal(NS.MAX_IMPORTS, 8);
console.log('PASS: imports 别名/去重/默认 torch+nn 去重/绑定名冲突/注入拒绝');

/* ---- 4. 旧规范兼容 + 输出类型提示扩展 ---- */
const oldSpec = NS.normalize(draft({}), { literal: literalOk });   // 无 imports 字段、Tensor 输出
assert.deepEqual(oldSpec.imports, []);
assert.equal(oldSpec.output.type, 'Tensor');
for (const hint of ['string', 'bool', 'list', 'tuple', 'dict']) {
  const spec = NS.normalize(draft({ output: { name: 'out', type: hint } }), { literal: literalOk });
  assert.equal(spec.output.type, hint);
  assert.deepEqual(spec.imports, []);
}
for (const hint of NS.INPUT_TYPE_HINTS) {
  const spec = NS.normalize(draft({ inputs: [{ name: 'x', type: hint, optional: false, fallbackParam: null, help: '' }] }), { literal: literalOk });
  assert.equal(spec.inputs[0].type, hint);
}
assert.throws(() => NS.normalize(draft({ output: { name: 'out', type: 'Dict' } })), /输出类型/);
console.log('PASS: 旧规范无 imports/Tensor 默认兼容；string/bool/list/tuple/dict 输出类型提示扩展');

/* ---- 5. 注册表：imports 是规范内容，改动必须产生新版本；序列化往返保留 imports ---- */
const registry = NS.createRegistry();
const mathSpec = draft({ imports: [{ module: 'math', alias: null }] });
const first = registry.publish(mathSpec, { literal: literalOk });
assert.equal(first.created, true);
assert.equal(registry.publish(mathSpec, { literal: literalOk }).created, false, '幂等发布');
assert.throws(() => registry.publish(draft({ imports: [{ module: 'math', alias: null }, { module: 'json' }] }), { literal: literalOk }),
  e => e instanceof NS.VersionConflict, '改 imports = 同 id+version 不同内容 → 版本冲突');
const second = registry.publishUpdate(draft({ imports: [{ module: 'math', alias: null }, { module: 'json' }] }), { literal: literalOk });
assert.equal(second.encoded, 'custom.sqrt@2');
assert.equal(registry.get('custom.sqrt@1').imports.length, 1, '旧版本定义保持原 imports');
const rebuilt = NS.createRegistry();
for (const spec of registry.serialize()) rebuilt.publish(spec, { literal: literalOk });
assert.deepEqual(rebuilt.get('custom.sqrt@2').imports, [{ module: 'math', alias: null }, { module: 'json', alias: null }],
  '序列化 → 重建往返保留 imports');
console.log('PASS: 注册表 imports 不可变（冲突→版本 2）与序列化往返');

/* ---- 6. 自由目标 + imports 的调用表达式与卡片摘要 ---- */
const rendered = NS.renderCall(
  NS.normalize(draft({ call: fnCall('numpy.linalg.norm'), imports: [{ module: 'numpy', alias: 'np' }] }), { literal: literalOk }),
  ['vec'], {}, literalOk);
assert.equal(rendered, 'numpy.linalg.norm(vec)');
const described = NS.describeCall(
  NS.normalize(draft({ call: { kind: 'method', target: 'item', args: [] } }), { literal: literalOk }), {}, {});
assert.equal(described, 'x.item()');
console.log('PASS: 自由目标渲染与卡片摘要');

console.log('ALL PASS: 自由目标 + 声明式 imports 隔离回归');
