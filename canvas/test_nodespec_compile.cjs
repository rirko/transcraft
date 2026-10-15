/* NodeSpec 编译接入集成测试（node test_nodespec_compile.cjs）。
 * 在 vm 中按 index.html 的顺序加载真实脚本，验证：
 *   - 自定义节点经统一 stmtOf 分支编译（无专用 sqrt 特判），真实 torch 执行；
 *   - 可选输入 fallbackParam 连线优先；方法调用 receiver/keyword 渲染；
 *   - 自由目标（math.sqrt 等）+ 声明式 imports 去重生成头部、未使用不生成；
 *   - sourceMap / tracePlan 覆盖自定义节点；节点复制后仍编译一致；
 *   - 缺规范/非法参数 → 保留原图、明确报错、禁止运行；
 *   - 项目快照 nodeSpecs、切项目清理注册表、旧图缺字段兼容；
 *   - 制作器 DOM 合约（模板/实时目标语法校验/发布/版本更新/JSON 导入）。
 * 全程内存 fixtures，不接触任何项目数据。 */
const fs = require('fs'), vm = require('vm'), assert = require('assert'), cp = require('child_process'), path = require('path');

const context = vm.createContext({ console, window: { addEventListener() {} }, setTimeout, clearTimeout });
for (const file of ['editor.js', 'app.js', 'architecture.js', 'nodespec.js', 'nodespec-app.js']) {
  vm.runInContext(fs.readFileSync(path.join(__dirname, 'static', file), 'utf8'), context);
}
const get = expr => vm.runInContext(expr, context);
get(`ED.exportData=key=>fixtures[key]||{};App.fnSignature=name=>({inputs:[],outputs:[]});`);
const NS = get('window.NS');
assert(NS, 'nodespec.js 必须向 window 注入 NS');
const python = path.join(__dirname, '../.venv/Scripts/python.exe');

function node(id, type, data = {}, inputs = [], outputs = 1) {
  return { id, data: { type, name: type.startsWith('custom.') ? type.split('.')[1].split('@')[0] : type, ...data },
    inputs: Object.fromEntries(inputs.map((c, i) => ['input_' + (i + 1), { connections: c && c[0] != null ? [{ node: String(c[0]), input: 'output_' + (c[1] || 1) }] : [] }])),
    outputs: Object.fromEntries(Array.from({ length: outputs }, (_, i) => ['output_' + (i + 1), { connections: [] }])) };
}
function compile(fixtures) {
  context.fixtures = fixtures;
  return get(`App.state.fnOrder=[];App.state.definitions={};({code:App.generatePython(),errors:App.lastErrors})`);
}
const literalCache = get('App.literalCache');

/* ---- 项目注册表：发布 triu / masked_fill / sqrt 模板（模拟项目快照注册） ---- */
for (const template of [NS.TEMPLATES.sqrt, NS.TEMPLATES.triu, NS.TEMPLATES.masked_fill]) {
  context.template = template;
  get(`App.customRegistry.publish(template, {literal: s => App.literalCache.get(s) || null})`);
}
get('App.installNodeSpecs()');
assert(get(`!!NODE_DEFS['custom.sqrt@1']`) && get(`!!NODE_DEFS['custom.triu@1']`) && get(`!!NODE_DEFS['custom.masked_fill@1']`));
assert.equal(get(`App.isStmtType('custom.sqrt@1')`), true, '统一谓词必须覆盖自定义类型');
assert.equal(get(`App.isStmtType('variable_ref')`), true);

/* ---- 1. 函数型自定义节点：编译 + sourceMap + 真实 torch 执行 ---- */
const sqrtGraph = {
  main: {
    1: node(1, 'tensor', { name: 'nums', values: '[4, 9]' }),
    2: node(2, 'custom.sqrt@1', { name: 'roots' }, [[1]]),
    3: node(3, 'print', { name: 'show', label: '开方' }, [[2]], 0),
  },
};
let result = compile(sqrtGraph);
assert.deepEqual(result.errors, []);
assert(result.code.includes('roots = torch.sqrt(nums)'), result.code);
assert(!result.code.includes('未知节点类型'));
assert.equal((result.code.match(/torch\.sqrt/g) || []).length, 1, '无专用 sqrt 编译：只有规范渲染出的一处');
const codeLines = result.code.split('\n');
const mapped = id => Array.from(get(`App.codeSourceMap.main['${id}']`), n => codeLines[n - 1].trim());
assert.deepEqual(mapped(2), ['roots = torch.sqrt(nums)']);
assert.deepEqual(mapped(3), ['print("开方", roots)']);
const traceEntry = get('App.tracePlan').find(p => p.scope === 'main' && p.node === '2');
assert(traceEntry && traceEntry.lines.length === 1 && traceEntry.sources[0].node === '1');
cp.spawnSync(python, ['-c', result.code + '\nassert torch.equal(roots, torch.tensor([2.0, 3.0]))'], { encoding: 'utf8', timeout: 120000 });
console.log('PASS: 函数型自定义节点编译/源码映射/真实 torch 执行');

/* ---- 2. 可选输入 fallbackParam 连线优先 + 参数非法报错 ---- */
const triuGraph = {
  main: {
    1: node(1, 'tensor', { name: 'grid', values: '[[1, 2], [3, 4]]' }),
    2: node(2, 'custom.triu@1', { name: 'upper', diagonal: '1' }, [[1], null]),
  },
};
result = compile(triuGraph);
assert.deepEqual(result.errors, []);
assert(result.code.includes('upper = torch.triu(grid, diagonal=1)'), result.code);
const triuConnected = JSON.parse(JSON.stringify(triuGraph));
triuConnected.main[4] = node(4, 'literal', { name: 'diag', value: '-1' });
triuConnected.main[2].inputs.input_2.connections = [{ node: '4', input: 'output_1' }];
literalCache.set('-1', { expression: '-1', type: 'int' });
result = compile(triuConnected);
assert.deepEqual(result.errors, []);
assert(result.code.includes('diag = -1') && result.code.includes('upper = torch.triu(grid, diagonal=diag)'), '连线必须优先于参数回退：' + result.code);
const badParam = JSON.parse(JSON.stringify(triuGraph));
badParam.main[2].data.diagonal = 'abc';
result = compile(badParam);
assert(result.errors.some(e => e.includes('diagonal')), result.errors);
assert(!result.code.includes('torch.triu'), '非法参数节点必须被跳过（不生成语句、不落默认值）');
console.log('PASS: 可选输入 fallbackParam 连线优先、非法参数明确报错并跳过');

/* ---- 3. Tensor 方法 + receiver + literal 参数 + keyword ---- */
literalCache.set('0.5', { expression: '0.5', type: 'float' });
literalCache.set('True', { expression: 'True', type: 'bool' });
const fillGraph = {
  main: {
    1: node(1, 'tensor', { name: 'scores', values: '[[1.0, 2.0], [3.0, 4.0]]' }),
    2: node(2, 'tensor', { name: 'mask_data', values: '[[True, False], [True, True]]' }),
    3: node(3, 'literal', { name: 'fill_value', value: '0.5' }),
    4: node(4, 'custom.masked_fill@1', { name: 'masked' }, [[1], [2], [3]]),
  },
};
result = compile(fillGraph);
assert.deepEqual(result.errors, []);
assert(result.code.includes('masked = scores.masked_fill(mask_data, value=fill_value)'), result.code);
cp.spawnSync(python, ['-c', result.code + '\nassert torch.equal(masked, torch.tensor([[0.5, 2.0], [0.5, 0.5]]))'], { encoding: 'utf8', timeout: 120000 });
const fillFallback = JSON.parse(JSON.stringify(fillGraph));
fillFallback.main[4].inputs.input_3.connections = [];
result = compile(fillFallback);
assert(result.code.includes('masked = scores.masked_fill(mask_data, value=float("-inf"))'), result.code);
console.log('PASS: 方法模板 receiver/keyword/literal 参数与 -inf 回退，真实 torch 执行');

/* ---- 3.5 自由目标 + 声明式 imports：头部去重生成 import math，真实 python 执行 ---- */
literalCache.set('4', { expression: '4', type: 'int' });
const mathSqrtSpec = { ...NS.deepCopy(NS.TEMPLATES.sqrt), id: 'math_sqrt',
  imports: [{ module: 'math', alias: null }],
  call: { kind: 'function', target: 'math.sqrt', args: [{ source: 'input', index: 1, keyword: null, fallbackParam: null }] } };
context.template = mathSqrtSpec;
get(`App.customRegistry.publish(template, {literal: s => App.literalCache.get(s) || null})`);
get('App.installNodeSpecs()');
const mathGraph = { main: {
  1: node(1, 'literal', { name: 'four', value: '4' }),
  2: node(2, 'custom.math_sqrt@1', { name: 'root' }, [[1]]),
  3: node(3, 'custom.math_sqrt@1', { name: 'root_too' }, [[1]]),
} };
result = compile(mathGraph);
assert.deepEqual(result.errors, []);
assert(result.code.includes('root = math.sqrt(four)') && result.code.includes('root_too = math.sqrt(four)'), result.code);
assert.equal((result.code.match(/import math/g) || []).length, 1, '同一 spec 的 imports 跨节点去重：只生成一次 import math');
assert(result.code.indexOf('import math') > result.code.indexOf('import torch.nn as nn'), '自定义 imports 在默认 torch/nn 之后');
assert(!/import  {2}/.test(result.code), '不会生成空模块导入');
cp.spawnSync(python, ['-c', result.code + '\nassert root == 2.0 and root_too == 2.0'], { encoding: 'utf8', timeout: 120000 });
/* imports 只对实际使用的 custom 节点生成：规范已注册但图未使用 → 不出现 import math */
result = compile({ main: { 1: node(1, 'literal', { name: 'four', value: '4' }) } });
assert.deepEqual(result.errors, []);
assert(!result.code.includes('import math'), '未使用自定义节点时不生成其 imports');
console.log('PASS: 自由目标 math.sqrt 编译 + 声明式 imports 去重生成 + 未使用不生成，真实 python 执行');

/* ---- 4. custom 节点复制：剪贴板只带数据字段，粘贴后编译一致 ---- */
get(`ED.getNode=(key,id)=>fixtures[key]?.[id];ED.hooks.projectId=()=>null;`);
context.fixtures = fillGraph;
assert(get(`ED.copyNode('main','4')`) === true);
const clip = JSON.parse(get(`JSON.stringify(ED.clipboard)`));
assert.equal(clip.data.type, 'custom.masked_fill@1');
assert.deepEqual(Object.keys(clip.data).sort(), ['name', 'type'], '复制只携带 type/name（本节点无参数字段）');
const pastedGraph = { main: {
  1: fillGraph.main[1], 2: fillGraph.main[2], 3: fillGraph.main[3],
  9: node(9, clip.data.type, { ...clip.data, name: 'masked_copy' }, [[1], [2], [3]]),
} };
result = compile(pastedGraph);
assert.deepEqual(result.errors, []);
assert(result.code.includes('masked_copy = scores.masked_fill(mask_data, value=fill_value)'), result.code);
console.log('PASS: custom 节点复制（剪贴板数据字段）与粘贴后编译一致');

/* ---- 5. 缺规范：保留原图、明确报错、禁止运行 ---- */
const broken = { main: { 1: node(1, 'custom.ghost@3', { name: 'x' }, []) } };
const before = JSON.stringify(broken);
context.fixtures = broken;
assert.throws(() => get('App.generatePython()'), /缺少节点规范 custom\.ghost@3/);
assert.equal(JSON.stringify(broken), before, '编译失败不得改动图数据');
get('ED.canvases={main:{}}');   // 模拟已打开的画布，供实时缺失扫描使用
assert.match(get('App.specProblemNow()'), /缺少节点定义：custom\.ghost@3/);
/* 已注册但参数非法 → buildScope 报错 + 节点跳过（不抛异常，走草稿路径） */
result = compile(badParam);
assert(result.errors.some(e => e.includes('diagonal')) && !result.code.includes('torch.triu'));
console.log('PASS: 缺规范抛错且不改图、实时缺失扫描、非法参数走草稿路径');

/* ---- 6. 项目快照 nodeSpecs + 切项目清理 + 旧图兼容 ---- */
vm.runInContext(fs.readFileSync(path.join(__dirname, 'static', 'projects.js'), 'utf8'), context);   // 与 index.html 一致：projects.js 在 nodespec 之后加载
context.fixtures = {};
const specs = JSON.parse(get('JSON.stringify(App.nodeSpecsSnapshot())'));
assert.equal(specs.length, 4);
assert(specs.some(s => s.id === 'masked_fill' && s.version === 1));
/* App.snapshot（projects.js）必须携带 nodeSpecs */
const snap = JSON.parse(get('JSON.stringify(App.snapshot())'));
assert.equal(snap.version, 3);
assert.equal(snap.nodeSpecs.length, 4);
assert(snap.nodeSpecs.every(s => s.schema === 1 && s.id && s.version >= 1));
/* 模拟载入另一个项目：先注册其规范（清空重建），缺失引用要报错 */
context.projectGraph = { version: 3, seed: 1, main: {}, functions: [], nodeSpecs: [NS.TEMPLATES.sin] };
assert.equal(get('App.prepareProjectSpecs(projectGraph)'), '');
assert.equal(get('App.customRegistry.size()'), 1, '切换项目必须清理并重建注册表');
assert(get(`!!NODE_DEFS['custom.sin@1']`) && !get(`!!NODE_DEFS['custom.sqrt@1']`), '旧项目的 NODE_DEFS 安装被清理');
context.brokenProject = { version: 3, seed: 1, main: { 1: { data: { type: 'custom.sin@1' } }, 2: { data: { type: 'custom.lost@1' } } }, functions: [] };
const problem = get('App.prepareProjectSpecs(brokenProject)');
assert(problem.includes('custom.lost@1') && problem.includes('禁止运行'), problem);
context.legacyGraph = { version: 3, seed: 1, main: {}, functions: [] };   // 旧 v2/v3：无 nodeSpecs 字段
assert.equal(get('App.prepareProjectSpecs(legacyGraph)'), '');
assert.equal(get('App.customRegistry.size()'), 0);
console.log('PASS: 快照含未使用定义、切项目清理注册表、缺失引用报错、旧图缺字段为空');

/* ---- 7. 制作器 DOM 合约（离线 stub，不解析 HTML） ---- */
class UIElement {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase(); this.children = []; this.dataset = {}; this.attributes = {};
    this.hidden = false; this.value = ''; this.textContent = ''; this.className = ''; this.style = {};
    this.events = {};
    const self = this;
    this.classList = { add(n) { self.className = (self.className ? self.className + ' ' : '') + n; }, remove() {}, toggle() {}, contains() { return false; } };
  }
  appendChild(child) { this.children.push(child); return child; }
  prepend(child) { this.children.unshift(child); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(k, v) { this.attributes[k] = String(v); }
  addEventListener(name, fn) { (this.events[name] ||= []).push(fn); }
  set type(v) { this._type = v; }
  get type() { return this._type || ''; }
  set checked(v) { this._checked = v; }
  get checked() { return !!this._checked; }
  click() { (this.events.click || []).forEach(fn => fn({ stopPropagation() {} })); }
  before() {}   // 制作器会在状态行前插入试运行代码块；离线 stub 无需真实 DOM 插入
  querySelectorAll() { return []; }   // 离线 stub 不模拟选择器查询；旧分组在本环境不存在
  get text() { return this.textContent + this.children.map(c => c.textContent).join(''); }
}
const elements = new Map();
const byId = id => { if (!elements.has(id)) elements.set(id, new UIElement('div')); return elements.get(id); };
context.document = {
  querySelector: sel => (sel.startsWith('#') ? byId(sel.slice(1)) : null),
  createElement: tag => new UIElement(tag),
  createTextNode: text => ({ textContent: String(text) }),
};
byId('canvas-stack');
get(`App.maker.draft=null;App.maker.built=false;App.openMaker()`);
const stack = byId('canvas-stack');
const panel = stack.children.find(c => c.id === 'nodespec-maker');
assert(panel && panel.children.length > 0, '制作器面板必须构建出列结构');
get(`App.makerLoadTemplate('triu')`);
assert.equal(get('App.maker.draft.id'), 'triu');
assert.equal(get('App.maker.draft.call.target'), 'torch.triu');
assert.equal(get(`App.maker.draft.call.args[1].fallbackParam`), 'diagonal');
assert(byId('maker-code-preview').text.includes('out = torch.triu(in1, diagonal=in2)'), byId('maker-code-preview').text);
/* 验证失败路径与成功路径的状态文案（自由目标：合法语法通过，非法语法实时报红） */
get(`App.maker.draft.call.target="eval(x";App.makerRefreshPreview()`);
assert(byId('maker-code-preview').text.includes('规范暂无效'), '非法目标语法必须实时报红');
get(`App.maker.draft.call.target="torch.load";App.makerRefreshPreview()`);
assert(!byId('maker-code-preview').text.includes('规范暂无效'), '自由目标 torch.load 不再被白名单拒绝（信任边界：真实执行）');
get(`App.maker.draft.call.target="math.sqrt";App.makerRefreshPreview()`);
assert(byId('maker-code-preview').text.includes('out = math.sqrt(in1, diagonal=in2)'), byId('maker-code-preview').text);
get(`App.maker.draft=window.NS.deepCopy(window.NS.TEMPLATES.sqrt);App.makerRefreshPreview()`);
assert(byId('maker-code-preview').text.includes('torch.sqrt'));
/* DOM 回归（事件层）：真实改动类型提示 select 触发 change 事件后，卡片预览
 * mk-pin-label 必须立即跟随（不只改草稿 schema，也不只测 NS.TEMPLATES）。 */
get(`App.makerLoadTemplate('sqrt')`);
const pinLabels = () => {
  const found = [];
  (function walk(el) {
    if (String(el.className).includes('mk-pin-label') && el.textContent) found.push(el.textContent);
    (el.children || []).forEach(walk);
  })(byId('maker-card-preview'));
  return found;
};
const fieldControl = (boxId, label) => {
  const hits = [];
  (function walk(el) {
    if (el.tagName === 'LABEL' && el.children[0] && el.children[0].textContent === label) hits.push(el.children[el.children.length - 1]);
    (el.children || []).forEach(walk);
  })(byId(boxId));
  return hits[0];
};
assert.equal(get('App.maker.draft.inputs[0].type') + '/' + get('App.maker.draft.output.type'), 'Tensor/Tensor');
assert(pinLabels().some(t => t.includes('x: Tensor')) && pinLabels().some(t => t.includes('out: Tensor')), pinLabels());
const inputTypeSelect = fieldControl('maker-inputs', '类型提示');
const outputTypeSelect = fieldControl('maker-meta', '输出类型提示');
assert(inputTypeSelect && inputTypeSelect.tagName === 'SELECT' && outputTypeSelect.tagName === 'SELECT', '输入/输出类型提示下拉必须真实存在');
for (const [sel, expr] of [[inputTypeSelect, 'App.maker.draft.inputs[0].type'], [outputTypeSelect, 'App.maker.draft.output.type']]) {
  sel.value = 'number';
  (sel.events.change || []).forEach(fn => fn());   // 与浏览器一致：用户改值后触发 change 事件
}
assert.equal(get('App.maker.draft.inputs[0].type'), 'number', 'change 事件必须写入输入类型提示');
assert.equal(get('App.maker.draft.output.type'), 'number', 'change 事件必须写入输出类型提示');
const labels = pinLabels();
assert(labels.some(t => t.includes('x: number')) && labels.some(t => t.includes('out: number')), '预览 mk-pin-label 必须显示新类型: ' + labels);
assert(!labels.some(t => t.includes('Tensor')), '预览必须立即刷新，不得残留旧类型: ' + labels);
console.log('PASS: 类型提示 change 事件实时刷新卡片预览 mk-pin-label');
/* 发布 + 同 id 不同内容 → 自动版本 2，旧节点仍绑版本 1 */
(async () => {
  await get('App.makerPublish()');
  assert(get(`!!NODE_DEFS['custom.sqrt@1']`), '发布必须安装 NODE_DEFS 定义');
  assert.match(byId('maker-status').text, /custom\.sqrt@1/);
  get('App.maker.draft.label="改名后的 sqrt"');
  await get('App.makerPublish()');
  assert(get(`!!NODE_DEFS['custom.sqrt@2']`), '更新必须产生版本 2');
  assert(get(`NODE_DEFS['custom.sqrt@1'].spec.label`) !== '改名后的 sqrt', '旧版本定义保持不变');
  assert.match(byId('maker-status').text, /custom\.sqrt@2/);
  /* JSON 高级导入：非法 JSON 报错不覆盖草稿 */
  byId('maker-json-area').value = '{broken json';
  const beforeDraft = JSON.stringify(get('App.maker.draft'));
  const importButtons = [];
  (function walk(el) { if (el.textContent === '导入 JSON') importButtons.push(el); (el.children || []).forEach(walk); })(panel);
  importButtons[0].click();
  assert.match(byId('maker-status').text, /导入失败/);
  assert.equal(JSON.stringify(get('App.maker.draft')), beforeDraft, '导入失败不得改动草稿');
  console.log('PASS: 制作器模板/实时校验/发布/版本更新/JSON 导入防护');
  console.log('ALL PASS: nodespec 编译接入与制作器合约');
})().catch(e => { console.error(e); process.exit(1); });
