/*
 * app.js —— 状态管理 + Python 代码生成 + 同步/运行 + 模型类机制
 * 与 editor.js 的分工：ED 管画布交互，App 管语义（画布 → 代码的翻译器在这里）。
 *
 * 类生成规则（v2）：
 *   - 每个模型画布发布为一个 nn.Module 类
 *   - 画布里的 Embedding / Linear / 单输出模型调用节点 → 提升到 __init__，挂为 self.xxx
 *   - 其余语句留在 forward；「函数输入/输出」节点声明 forward 的参数和返回值
 *   - 主画布的调用节点 → 两行：实例化 + 前向
 */

const NODE_DEFS = ED.NODE_DEFS;

const App = {
  state: { seed: 1234, fnOrder: [] },   // fnOrder: 模型类名，按创建顺序
  selected: null,                        // { key, id }
  drawers: new Map(),                    // JSON [scope, node id] -> independent drawer
  syncTimer: null,
};

/* ================= 工具 ================= */
const $ = sel => document.querySelector(sel);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const asInt = (v, def) => { const n = parseInt(v, 10); return Number.isFinite(n) ? n : def; };
const asBool = v => v === true || v === 'true';

const PY_KEYWORDS = new Set(('False None True and as assert async await break class continue def del elif else '
  + 'except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield '
  + 'print').split(' '));

function toIdent(s, used, fallback) {
  s = String(s ?? '').normalize('NFKC').trim().replace(/\s+/g, '_').replace(/[^\p{ID_Continue}]/gu, '');
  if (s && !/^[_\p{ID_Start}]/u.test(s)) s = '_' + s;
  if (!s) s = fallback;
  if (PY_KEYWORDS.has(s)) s += '_';
  let name = s, i = 2;
  while (used.has(name)) name = `${s}_${i++}`;
  used.add(name);
  return name;
}

/* ================= 初始化 ================= */
App.init = async function () {
  const stack = $('#canvas-stack');
  const mainDiv = document.createElement('div');
  mainDiv.id = 'canvas-main';
  mainDiv.className = 'drawflow dflow-container';
  stack.appendChild(mainDiv);
  ED.makeEditor('main', 'canvas-main');

  ED.hooks.onCanvasChanged = () => App.scheduleSync();
  ED.hooks.onSelect = (key, id) => App.selectNode(key, id);
  ED.hooks.onDrawerToggle = (key, id) => App.toggleNodeDrawer(key, id);
  ED.hooks.onFormMount = (key, id, box) => App.renderInspectorForm(key, id, box);
  ED.hooks.onFormSync = (key, id, box) => App.syncInspectorForm(key, id, box);
  ED.hooks.onNameEdit = (key, id) => App.editNodeName(key, id);
  ED.hooks.onDeselect = () => App.hideInspector();
  ED.hooks.onReset = key => App.closeNodeDrawers(key);
  ED.hooks.onRemove = (key, id) => {
    App.closeNodeDrawer(key, id);
    if (App.selected?.key === key && String(App.selected.id) === String(id)) App.hideInspector();
  };
  ED.hooks.fnMeta = fname => App.fnSignature(fname);
  ED.hooks.projectId = () => App.project?.id ?? null;
  ED.hooks.targetExists = (fname, type) => App.state.fnOrder.includes(fname)
    && !!ED.canvases['fn:' + fname]
    && (type === 'function_call' ? App.state.definitions[fname]?.kind === 'function' : App.state.definitions[fname]?.kind !== 'function');
  ED.hooks.drawerOpen = (key, id) => !!App.nodeDrawer(key, id);
  ED.hooks.copyName = (key, base, used) => toIdent(base + '_copy', new Set([...used,
    'self', 'torch', 'nn', '_canvas_trace', '_canvas_return', 'training', 'forward', '__init__',
    ...App.state.fnOrder, ...String(App.state.definitions?.[key.slice(3)]?.params || '').split(',').map(s => s.trim())]), 'copy');
  ED.hooks.onPaste = (key, id, drawerOpen) => {
    App.selectNode(key, id);
    if (drawerOpen) App.showInspector(key, id);
  };

  App.buildPalette();
  App.bindHeader();
  App.renderTabs();

  await App.initProjects();
};

/* ================= 容器定位 ================= */
App.containerOf = function (key) {
  const id = key === 'main' ? 'canvas-main' : 'canvas-fn-' + key.slice(3);
  return document.getElementById(id);
};

/* ================= 标签页 ================= */
App.switchTab = function (key) {
  ED.activeKey = App.state.activeKey = key;
  document.querySelectorAll('.dflow-container').forEach(d => (d.style.display = 'none'));
  const cont = App.containerOf(key);
  if (cont) cont.style.display = 'block';
  ED.redraw(key);
  document.body.dataset.mode = key === 'main' ? 'main' : 'fn';
  document.querySelectorAll('#tabs .tab').forEach(t =>
    t.classList.toggle('active', t.dataset.key === key));
  App.hideInspector();
};

App.renderTabs = function () {
  const tabs = $('#tabs');
  tabs.innerHTML = '';
  const mk = (label, key, closable) => {
    const t = document.createElement('div');
    t.className = 'tab' + (key === App.state.activeKey ? ' active' : '');
    t.dataset.key = key;
    t.innerHTML = `<span class="tab-label">${esc(label)}</span>` +
      (closable ? `<span class="tab-close" title="删除该模型">×</span>` : '');
    t.addEventListener('click', e => {
      if (e.target.classList.contains('tab-close')) return;
      App.switchTab(key);
    });
    if (closable) {
      t.querySelector('.tab-close').addEventListener('click', e => {
        e.stopPropagation();
        App.closeFunction(key.slice(3));
      });
    }
    tabs.appendChild(t);
  };
  mk('主画布', 'main', false);
  for (const fname of App.state.fnOrder) mk(fname, 'fn:' + fname, true);
  const add = document.createElement('div');
  add.className = 'tab tab-add';
  add.textContent = '＋ 模型';
  add.title = '新建一个模型画布（将生成 class 该名字(nn.Module)）';
  add.addEventListener('click', () => {
    const raw = window.TC?.skipConfirm ? 'MyModel' : prompt('模型类名（将生成 class 该名字(nn.Module)）：', 'MyModel');
    if (raw === null) return;
    App.createFunction(raw);
  });
  tabs.appendChild(add);
};

/* ================= 模型管理 ================= */
App.createFunctionRaw = function (name) {
  const container = document.createElement('div');
  container.id = 'canvas-fn-' + name;
  container.className = 'drawflow dflow-container';
  container.style.display = 'none';
  $('#canvas-stack').appendChild(container);
  ED.makeEditor('fn:' + name, container.id);
  App.state.fnOrder.push(name);
  return name;
};

App.createFunction = function (rawName) {
  const used = new Set([...App.state.fnOrder,'_canvas_trace','_canvas_return']);
  const name = toIdent(rawName, used, 'MyModel');
  App.createFunctionRaw(name);
  App.renderTabs();
  App.renderFnPalette();
  App.switchTab('fn:' + name);
  App.scheduleSync();
  return name;
};

App.removeFunctionRaw = function (name) {
  const key = 'fn:' + name;
  App.closeNodeDrawers(key);
  if (App.selected?.key === key) App.hideInspector();
  const cont = App.containerOf(key);
  ED.canvases[key]?.observer?.disconnect();
  if (cont) cont.remove();
  delete ED.canvases[key];
  App.state.fnOrder = App.state.fnOrder.filter(f => f !== name);
};

App.closeFunction = function (name) {
  if (!confirm(`删除模型「${name}」？\n主画布里调用它的节点将失效（会以注释形式提示）。`)) return;
  App.removeFunctionRaw(name);
  App.renderTabs();
  App.renderFnPalette();
  App.switchTab('main');
  App.scheduleSync();
};

/* 模型接口签名：从模型画布实时推导（fn_input 的参数名 + fn_output 的返回端口） */
App.fnSignature = function (fname) {
  const df = ED.exportData('fn:' + fname);
  const nodes = Object.values(df);
  const byId = (a, b) => Number(a.id) - Number(b.id);
  const inputs = nodes.filter(n => n.data?.type === 'fn_input').sort(byId)
    .map(n => n.data.argname || 'x');
  const outputs = [];
  nodes.filter(n => n.data?.type === 'fn_output').sort(byId).forEach(n => {
    const cnt = Object.keys(n.inputs || {}).length;
    for (let i = 1; i <= cnt; i++) {
      const conn = n.inputs?.['input_' + i]?.connections?.[0];
      const src = conn ? df[conn.node] : null;
      outputs.push(src ? (src.data.type === 'fn_input' ? (src.data.argname || 'x') : (src.data.name || 'ret')) : 'ret');
    }
  });
  return { inputs, outputs };
};

/* ================= 组件栏 ================= */
App.buildPalette = function () {
  const pal = $('#palette-basic');
  pal.innerHTML = '';
  const groups = [
    ['数据 / 输入', ['literal', 'variable_ref', 'tensor']],
    ['模型层', ['embedding', 'linear']],
    ['数学运算', ['matmul', 'binop', 'softmax']],
    ['张量变换', ['transpose', 'getitem', 'dtype']],
    ['遮罩', ['ones_like', 'triu', 'masked_fill']],
    ['输出', ['print', 'print_shape']],
  ];
  for (const [label, types] of groups) {
    const group = document.createElement('details');
    group.className = 'palette-group'; group.open = true;
    const heading = document.createElement('summary'); heading.textContent = label;
    group.appendChild(heading);
    for (const type of types) group.appendChild(App.paletteItem(type));
    pal.appendChild(group);
  }
  const fnPal = $('#palette-fnnodes');
  fnPal.innerHTML = '';
  for (const type of ['fn_input', 'fn_output']) {
    fnPal.appendChild(App.paletteItem(type));
  }
};

App.paletteItem = function (type, fname) {
  const def = NODE_DEFS[type];
  const el = document.createElement('div');
  el.className = 'palette-item';
  el.draggable = true;
  el.dataset.type = type;
  if (fname) el.dataset.fname = fname;
  el.innerHTML = `<div class="pi-name">${esc(fname || def.label)}</div>
    <div class="pi-desc">${esc(fname ? '模型调用节点' : (def.summary?.({ ...Object.fromEntries(def.params.map(p => [p.key, p.def])) }) || ''))}</div>`;
  el.addEventListener('dragstart', ev => {
    ev.dataTransfer.setData('node-type', type);
    if (fname) ev.dataTransfer.setData('fn-name', fname);
    ev.dataTransfer.effectAllowed = 'copy';
  });
  return el;
};

App.renderFnPalette = function () {
  const wrap = $('#palette-fns');
  wrap.innerHTML = '';
  if (!App.state.fnOrder.length) {
    wrap.innerHTML = '<div class="pi-empty">还没有模型。点顶部「＋ 模型」创建，<br>在模型画布里用「函数输入/输出」声明 forward 接口。</div>';
    return;
  }
  for (const fname of App.state.fnOrder) {
    const sig = App.fnSignature(fname);
    const el = App.paletteItem('call', fname);
    el.querySelector('.pi-desc').textContent = `(${sig.inputs.join(', ')}) → ${sig.outputs.join(', ')}`;
    wrap.appendChild(el);
  }
};

/* ================= Code preview ================= */
App.highlightCodeSelection = function (scroll = false) {
  const view = $('#code-view');
  if (!view) return;
  const selected = App.selected;
  const lines = new Set(selected ? App.previewSourceMap?.[selected.key]?.[String(selected.id)] || [] : []);
  let first;
  view.querySelectorAll('.code-line').forEach(line => {
    const active = lines.has(Number(line.dataset.line));
    line.classList.toggle('is-selected', active);
    if (active && !first) first = line;
  });
  if (scroll && first) first.scrollIntoView({ block: 'nearest', inline: 'nearest' });
};

App.renderCodePreview = function (code) {
  const view = $('#code-view'), top = view.scrollTop, left = view.scrollLeft;
  const fragment = document.createDocumentFragment();
  // Tokenize the whole source so multiline strings retain their grammar context.
  const prism = window.Prism;
  const tokens = prism?.languages.python ? prism.tokenize(code, prism.languages.python) : [code];
  let line, lineNumber = 0;
  function appendTokens(value, classes = []) {
    if (Array.isArray(value)) {
      value.forEach(token => appendTokens(token, classes));
    } else if (typeof value !== 'string') {
      appendTokens(value.content, [...classes, ['token', value.type, ...[].concat(value.alias || [])].join(' ')]);
    } else {
      // Split only after tokenization, rebuilding nested token spans on each line.
      for (const text of value.match(/[^\n]*\n|[^\n]+$/g) || []) {
        if (!line) {
          line = document.createElement('span');
          line.className = 'code-line';
          line.dataset.line = String(++lineNumber);
          fragment.appendChild(line);
        }
        let parent = line;
        for (const className of classes) {
          const span = document.createElement('span');
          span.className = className;
          parent.appendChild(span);
          parent = span;
        }
        parent.appendChild(document.createTextNode(text));
        if (text.endsWith('\n')) line = null;
      }
    }
  }
  appendTokens(tokens);
  App.previewSourceMap = App.codeSourceMap;
  view.replaceChildren(fragment);
  App.highlightCodeSelection();
  view.scrollTop = top;
  view.scrollLeft = left;
};

/* ================= 节点内抽屉 ================= */
App.nodeElement = function (key, id) {
  return ED.canvases[key]?.container.querySelector(`[id="node-${id}"]`) || null;
};
App.selectNode = function (key, id) {
  if (!ED.getNode(key, id)) return;
  const changed = App.selected?.key !== key || String(App.selected?.id) !== String(id);
  App.selected = { key, id };
  const canvas = ED.canvases[key], element = App.nodeElement(key, id);
  if (element) {
    canvas.container.querySelectorAll('.drawflow-node.selected').forEach(n => n.classList.remove('selected'));
    element.classList.add('selected');
    canvas.editor.node_selected = element;
  }
  App.highlightCodeSelection(changed);
};
App.drawerKey = (key, id) => JSON.stringify([key, String(id)]);
App.nodeDrawer = (key, id) => App.drawers.get(App.drawerKey(key, id));
App.closeNodeDrawer = function (key, id) {
  const token = App.drawerKey(key, id), drawer = App.drawers.get(token);
  if (!drawer) return;
  const node = App.nodeElement(key, id);
  if (node) {
    node.classList.remove('drawer-open');
    node.querySelector('.nd-drawer').hidden = true;
    const toggle = node.querySelector('.nd-toggle');
    toggle.setAttribute('aria-expanded', 'false');
    toggle.setAttribute('aria-label', '展开运行结果'); toggle.title = '展开运行结果';
    // 关闭运行结果不影响常显参数表单，只清空运行值视图。
    node.querySelector('.nd-runtime-view').replaceChildren();
  }
  App.drawers.delete(token);
  ED.redraw(key);
};
App.closeNodeDrawers = function (key) {
  for (const drawer of [...App.drawers.values()]) {
    if (key === undefined || drawer.key === key) App.closeNodeDrawer(drawer.key, drawer.id);
  }
  if (key !== undefined && App.selected?.key === key) App.hideInspector();
};
App.toggleNodeDrawer = function (key, id) {
  App.selectNode(key, id);
  if (App.nodeDrawer(key, id)) {
    App.closeNodeDrawer(key, id); return;
  }
  App.showInspector(key, id);
};
/* 运行结果抽屉：仅由标题箭头/来源跳转开关，参数表单常驻节点内，与此无关。 */
App.showInspector = function (key, id) {
  const n = ED.getNode(key, id), node = App.nodeElement(key, id);
  if (!n || !node) return;
  if (!App.nodeDrawer(key, id)) App.drawers.set(App.drawerKey(key, id), { key, id: String(id) });
  App.selectNode(key, id);
  node.classList.add('drawer-open');
  const drawer = node.querySelector('.nd-drawer');
  drawer.hidden = false; drawer.id = 'detail-' + encodeURIComponent(key) + '-' + id;
  drawer.dataset.scope = key; drawer.dataset.nodeId = String(id);
  const toggle = node.querySelector('.nd-toggle');
  toggle.setAttribute('aria-expanded', 'true'); toggle.setAttribute('aria-controls', drawer.id);
  toggle.setAttribute('aria-label', '收起运行结果'); toggle.title = '收起运行结果';
  App.renderTensorViewer?.(key, id);
  ED.redraw(key);
};

/* 常显参数表单：挂在节点 .nd-params 内，每个节点独立捕获自己的 scope/id。
 * 节点名/参数名不在这里——点击节点标题原位编辑（App.editNodeName）。 */
App.renderInspectorForm = function (key, id, box) {
  const n = ED.getNode(key, id);
  if (!n || !box) return;
  const def = NODE_DEFS[n.data.type] || { params: [], label: n.data.type };
  let html = '';
  for (const p of def.params) {
    if (n.data.type === 'fn_input' && p.key === 'argname') continue;   // 参数名由标题编辑
    const help=p.help?` <details class="param-help"><summary aria-label="${esc(p.key)} 参数说明" title="参数说明">?</summary><div>${esc(p.help)}</div></details>`:'';
    if (p.type === 'select') {
      html += `<label class="ins-row"><span>${esc(p.label)}${help}</span><select data-pkey="${p.key}" aria-label="${esc(p.label)}">${p.options.map(o=>`<option value="${esc(o)}" ${o===(n.data[p.key]??p.def)?'selected':''}>${esc(o)}</option>`).join('')}</select></label>`;
    } else if (p.type === 'bool') {
      html += `<label class="ins-row"><span>${esc(p.label)}${help}</span>
        <input type="checkbox" data-pkey="${p.key}" ${asBool(n.data[p.key]) ? 'checked' : ''}></label>`;
    } else if (p.type === 'text') {
      // Literal 值（内置 literal 节点与自定义节点 literal 参数）用多行文本域，方便填写字典。
      if (p.multiline || p.spectype === 'literal') {
        html += `<label class="ins-row"><span>${esc(p.label)}${help}</span>
          <textarea data-pkey="${p.key}" rows="3" spellcheck="false" placeholder='如 {0: "知", 1: "之"} 或 ["a", 1]'>${esc(n.data[p.key])}</textarea></label>`;
      } else {
        html += `<label class="ins-row"><span>${esc(p.label)}${help}</span>
          <input type="text" data-pkey="${p.key}" value="${esc(n.data[p.key])}"></label>`;
      }
    } else {
      html += `<label class="ins-row"><span>${esc(p.label)}${help}</span>
        <input type="number" data-pkey="${p.key}" value="${esc(n.data[p.key])}"></label>`;
    }
  }
  if (n.data.type === 'call') {
    html += `<div class="ins-hint">模型「${esc(n.data.fname)}」的调用节点，生成"实例化 + 前向"两行。
      端口是拖入时的签名快照；若模型接口后来改了，删除本节点重新拖一个即可。</div>`;
  }
  if (n.data.type === 'fn_output') {
    html += `<button class="ins-btn" data-ins-action="add-ret">＋ 添加返回值端口</button>`;
  }
  if (n.data.type === 'binop') {
    html += `<div class="ins-row binop-actions">` +
      `<button type="button" class="ins-btn" data-ins-action="add-input" title="在末尾增加一个输入（最多 16 个）；新输入继承上一段运算符">＋ 增加输入</button>` +
      `<button type="button" class="ins-btn" data-ins-action="remove-input" title="移除最后一个输入（最少保留 2 个）；若已连线会先经页面内确认，且不影响前面的连线">－ 移除最后输入</button></div>` +
      `<div class="ins-hint" data-binop-hint></div>`;
  }
  box.innerHTML = html;
  // 参数为空的节点（matmul、fn_input 等）不留空白表单容器；删除节点只走右键菜单。
  box.hidden = !html;

  box.querySelectorAll('[data-pkey]').forEach(inp => {
    inp.addEventListener('input', () => {
      const k = inp.dataset.pkey;
      const v = inp.type === 'checkbox' ? inp.checked : inp.value;
      App.selectNode(key, id);
      ED.updateNode(key, id, { [k]: v });
      App.scheduleSync();
    });
  });
  const addRet = box.querySelector('[data-ins-action="add-ret"]');
  if (addRet) addRet.addEventListener('click', () => {
    App.selectNode(key, id);
    ED.addInputPort(key, id);
    App.scheduleSync();
  });
  const addOperand = box.querySelector('[data-ins-action="add-input"]');
  if (addOperand) addOperand.addEventListener('click', () => {
    App.selectNode(key, id);
    ED.addOperand(key, id);
    App.scheduleSync();
  });
  const removeOperand = box.querySelector('[data-ins-action="remove-input"]');
  if (removeOperand) removeOperand.addEventListener('click', async () => {
    App.selectNode(key, id);
    await ED.removeOperand(key, id);
    App.scheduleSync();
  });
  App.syncBinopForm(box, n);
};

/* 运算 op 参数区：动态同步输入个数提示与增删按钮可用态（表单只挂载一次，不能靠重渲染）。 */
App.syncBinopForm = function (box, n) {
  if (n?.data?.type !== 'binop') return;
  const count = Object.keys(n.inputs || {}).length;
  const hint = box.querySelector('[data-binop-hint]');
  if (hint) hint.textContent = `当前 ${count} 个输入（${ED.binopName(0)}–${ED.binopName(count - 1)}）。` +
    '每个输入可连线或在行内手填字面量（连线优先，断开后恢复手填）；' +
    '相邻输入之间的运算符从上至下左折叠，例如 result = ((a + b) * c)。' +
    '输入既未连线也未填值、运算符为空或非法、字面量非法都会明确报错，不会默认 0。';
  const add = box.querySelector('[data-ins-action="add-input"]');
  const remove = box.querySelector('[data-ins-action="remove-input"]');
  if (add) add.disabled = count >= ED.BINOP_MAX;
  if (remove) remove.disabled = count <= ED.BINOP_MIN;
};

/* 外部数据变化 → 就地同步控件值；元素与监听不动，聚焦中的控件跳过（不丢焦点、不重复绑定）。 */
App.syncInspectorForm = function (key, id, box) {
  const n = ED.getNode(key, id);
  if (!n || !box) return;
  box.querySelectorAll('[data-pkey]').forEach(inp => {
    if (document.activeElement === inp) return;
    const v = n.data[inp.dataset.pkey];
    if (inp.type === 'checkbox') inp.checked = asBool(v);
    else inp.value = v ?? '';
  });
  App.syncBinopForm(box, n);
};

/* 点击节点标题 → 原位编辑节点名；Enter/失焦提交，Esc 取消，空值不提交。
 * 模型输入节点（fn_input）的标题就是 forward 参数名，编辑 argname 字段。 */
App.editNodeName = function (key, id) {
  const n = ED.getNode(key, id);
  const titleElement = App.nodeElement(key, id)?.querySelector('.nd-title');
  if (!n || !titleElement || titleElement.querySelector('.nd-name-input')) return;
  const isArg = n.data.type === 'fn_input';
  const field = isArg ? 'argname' : 'name';
  const current = String((isArg ? n.data.argname : n.data.name)
    || (isArg ? 'x' : NODE_DEFS[n.data.type]?.defaults?.name) || '');
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'nd-name-input';
  input.name = field;
  input.value = current;
  input.setAttribute('aria-label', isArg ? '编辑参数名' : '编辑节点名');
  titleElement.replaceChildren(input);
  input.focus(); input.select();
  let done = false;
  const finish = commit => {
    if (done) return;
    done = true;
    document.removeEventListener('keydown', onKey, true);
    const value = String(input.value ?? '').trim();
    const before = String((isArg ? n.data.argname : n.data.name) || '');
    titleElement.replaceChildren();           // 先移除输入框，refreshSummary 再恢复标题文本
    if (commit && value && value !== before) {
      App.selectNode(key, id);
      ED.updateNode(key, id, { [field]: value });
      App.scheduleSync();
    } else {
      ED.refreshSummary(key, id);
    }
  };
  // 容器在捕获阶段拦截了 input 的 keydown（避免触发 Drawflow 删除），
  // 因此 Enter/Esc 在 document 捕获阶段先行处理。
  const onKey = e => {
    if (e.target !== input) {
      if (input.isConnected === false) { done = true; document.removeEventListener('keydown', onKey, true); }
      return;
    }
    if (!['Enter', 'Escape'].includes(e.key)) return;
    e.preventDefault();
    finish(e.key === 'Enter');
  };
  document.addEventListener('keydown', onKey, true);
  input.addEventListener('blur', () => finish(true));
  input.addEventListener('input', () => input.classList.toggle('invalid', !String(input.value ?? '').trim()));
};

App.hideInspector = function () {
  App.selected = null;
  App.highlightCodeSelection();
};

/* ================= Python 代码生成 ================= */
const STMT_TYPES = ['literal', 'tensor', 'embedding', 'linear', 'matmul', 'binop', 'transpose', 'softmax', 'print', 'print_shape', 'getitem', 'call','ones_like','triu','dtype','masked_fill'];
const MODULE_TYPES = new Set(['embedding', 'linear']);

/* 统一可编译谓词：所有「是否生成语句」的判断都走这里，
 * 自定义节点（custom.*@v）由 nodespec-app.js 扩展进同一入口，避免名单漏新类型。 */
App.isStmtType = t => STMT_TYPES.includes(t);

/* 运算 op 节点允许的二元运算符白名单（与节点内运算符下拉框共用 ED.BIN_OPS） */
const BIN_OPS = ED.BIN_OPS;

function buildScope(df, opts) {
  const errors = [];
  const recs = {};
  const used = new Set(opts.reserved || []);
  const entries = Object.entries(df || {});
  // Allocate producer symbols before aliases so a new reference cannot rename its target.
  const namingEntries = [...entries.filter(([, n]) => n.data?.type !== 'variable_ref'),
    ...entries.filter(([, n]) => n.data?.type === 'variable_ref')];

  // ---- 第 1 遍：命名 ----
  for (const [id, n] of namingEntries) {
    const type = n.data?.type;
    const rec = { id: String(id), type, n, inRefs: [], valid: false, isParam: false, skipReason: '', outVars: [] };
    if (type === 'fn_input') {
      const wanted = (n.data.argname || 'x').trim();
      const arg = toIdent(wanted, used, 'x');
      rec.outVar = arg; rec.outVars = [arg]; rec.isParam = true; rec.valid = true;
      if (arg !== wanted) errors.push(`参数名「${wanted}」不是合法变量名，已改为「${arg}」`);
    } else {
      rec.outVar = toIdent(n.data?.name || NODE_DEFS[type]?.label || type, used, type);
      if (type === 'call') rec.instanceVar = toIdent(`${rec.outVar}_instance`, used, 'instance');
      if (['call','invoke','function_call'].includes(type)) {
        rec.outLabels = n.data?.outLabels || ['out'];
        rec.outVars = rec.outLabels.map((_, i) => {
          const base = i === 0 ? `${rec.outVar}_out` : `${rec.outVar}_out${i + 1}`;
          return toIdent(base, used, 'v');
        });
      } else {
        rec.outVars = [rec.outVar];
      }
    }
    recs[String(id)] = rec;
  }

  // ---- 模块节点判定：模型画布里可挂到 self.xxx 的零件 ----
  const isClass = !!opts.isClass;
  for (const rec of Object.values(recs)) {
    rec.isModule = isClass && !rec.isParam && (
      MODULE_TYPES.has(rec.type) ||
      (rec.type === 'call' && rec.outLabels.length === 1));
  }

  // ---- 第 2 遍：输入连线 → {上游, 输出端口号} ----
  for (const rec of Object.values(recs)) {
    if (rec.isParam) continue;
    const n = rec.n;
    let cnt;
    if (['call','invoke','function_call'].includes(rec.type)) cnt = (n.data.inLabels || []).length;
    else if (NODE_DEFS[rec.type]?.dynamicInputs) cnt = Object.keys(n.inputs || {}).length;
    else cnt = NODE_DEFS[rec.type].inputs;
    for (let i = 1; i <= cnt; i++) {
      const conn = n.inputs?.['input_' + i]?.connections?.[0];
      const up = conn ? recs[String(conn.node)] : null;
      // 0.0.59 把上游输出类名存在 conn.input 字段
      const outPort = conn ? (conn.output ?? conn.input) : null;
      const port = outPort ? (parseInt(String(outPort).replace('output_', ''), 10) - 1) : 0;
      rec.inRefs.push({ up, port, optional:!conn && !!NODE_DEFS[rec.type]?.optionalInputs?.includes(i-1) });
    }
  }

  opts.resolveReferences?.(recs, errors);

  // ---- 第 3 遍：拓扑排序（含模块节点，保证 self 表达式先于使用方） ----
  const nodeIds = Object.values(recs).filter(r => App.isStmtType(r.type) || r.isModule).map(r => r.id);
  const indeg = {}, downs = {};
  nodeIds.forEach(id => { indeg[id] = 0; downs[id] = []; });
  for (const rec of Object.values(recs)) {
    if (!nodeIds.includes(rec.id)) continue;
    for (const ref of rec.inRefs) {
      if (ref.up && nodeIds.includes(ref.up.id)) {
        downs[ref.up.id].push(rec.id);
        indeg[rec.id]++;
      }
    }
  }
  const queue = nodeIds.filter(id => indeg[id] === 0);
  const order = [];
  while (queue.length) {
    const id = queue.shift();
    order.push(id);
    for (const d of downs[id]) if (--indeg[d] === 0) queue.push(d);
  }
  const cyclic = nodeIds.filter(id => !order.includes(id));
  if (cyclic.length) {
    errors.push(`存在循环连线，涉及节点：${cyclic.map(id => '「' + (recs[id].n.data.name || recs[id].type) + '」').join('、')}`);
  }

  // ---- 第 4 遍：拓扑序上传播有效性 + 生成模块表达式 + 解析参数字符串 ----
  const ref = (up, port) => {
    if (!up) return null;
    if (up.isModule) return up.moduleExpr ?? null;
    return up.outVars[port] ?? up.outVar ?? null;
  };
  // 运算 op：先解析每个操作数（连线优先；未连线用手填 Literal，二者皆无则明确报错）。
  // 手填表达式只取后端 /api/literal 校验后的规范形式，绝不对图内文本求值。
  const binopLetter = i => String.fromCharCode(97 + Math.min(i, 25));
  const binopOpList = [...BIN_OPS].join(' ');
  for (const rec of Object.values(recs)) {
    if (rec.type !== 'binop') continue;
    const st = ED.binopState(rec.n);
    const count = rec.inRefs.length;
    rec.binopError = '';
    rec.manualExprs = [];
    for (let i = 0; i < count; i++) {
      const r = rec.inRefs[i];
      if (r.up) { rec.manualExprs.push(null); continue; }
      const raw = String(st.values[i] ?? '');
      const shown = raw.trim().length > 40 ? raw.trim().slice(0, 40) + '…' : raw.trim();
      if (!raw.trim()) {
        rec.binopError = `输入 ${binopLetter(i)} 未连接且未手填值（不会默认为 0）`;
        errors.push(`节点「${dispName(rec)}」输入 ${binopLetter(i)} 未连接且未手填值`);
        rec.manualExprs.push(null);
        continue;
      }
      const literal = App.literalCache.get(raw);
      if (!literal || literal.error) {
        rec.binopError = `输入 ${binopLetter(i)} 手填值「${shown}」不是合法字面量${literal?.error ? '：' + literal.error : '（等待 Literal 校验）'}`;
        errors.push(`节点「${dispName(rec)}」${rec.binopError}`);
        rec.manualExprs.push(null);
        continue;
      }
      rec.manualExprs.push(literal.expression);
      r.staticExpr = literal.expression;   // 静态来源：参与有效性判断与 trace_plan
      r.manualText = raw.trim();
    }
  }
  for (const id of order) {
    const rec = recs[id];
    rec.valid = !rec.referenceError && rec.inRefs.every(r => r.optional || r.staticExpr != null || (ref(r.up, r.port) !== null && (!r.up || r.up.valid)));
    if (!rec.valid) {
      rec.skipReason = rec.skipReason || (rec.binopError || '输入未连接（或上游被跳过）');
      continue;
    }
    // 运算 op 节点：逐段运算符白名单校验 + 左折叠显式括号（result = ((a op1 b) op2 c)）
    if (rec.type === 'binop') {
      const st = ED.binopState(rec.n);
      const count = rec.inRefs.length;
      if (count < ED.BINOP_MIN) {
        rec.valid = false;
        rec.skipReason = `运算 op 至少需要 ${ED.BINOP_MIN} 个输入端口，当前 ${count} 个`;
        errors.push(`节点「${dispName(rec)}」输入端口不足（图可能已损坏），请重建该节点`);
        continue;
      }
      const badOp = st.ops.slice(0, count - 1).find(op => !BIN_OPS.has(String(op)));
      if (badOp !== undefined) {
        rec.valid = false;
        rec.skipReason = `运算符「${String(badOp).trim() === '' ? '（空）' : badOp}」不支持`;
        errors.push(`节点「${dispName(rec)}」运算符「${String(badOp).trim() === '' ? '（空）' : badOp}」不支持（可用：${binopOpList}）`);
        continue;
      }
      const parts = rec.inRefs.map(r => r.staticExpr ?? ref(r.up, r.port));
      if (parts.some(p => p == null)) {
        rec.valid = false;
        rec.skipReason = rec.binopError || '输入未连接（或上游被跳过）';
        continue;
      }
      let expr = parts[0];
      for (let k = 1; k < count; k++) {
        // 两个操作数保持 a op b（与旧图一致）；更多项时每一步都加括号，避免 Python 优先级歧义
        expr = count === 2 ? `${expr} ${st.ops[k - 1]} ${parts[k]}` : `(${expr} ${st.ops[k - 1]} ${parts[k]})`;
      }
      rec.binopExpr = expr;
    }
    if (rec.isModule) {
      const argsStr = rec.inRefs.map(r => ref(r.up, r.port)).join(', ');
      rec.moduleExpr = `self.${rec.outVar}(${argsStr})`;
      rec.outVars = [rec.moduleExpr];
    }
  }
  for (const id of cyclic) {
    recs[id].valid = false;
    recs[id].skipReason = '处于循环连线中';
  }
  for (const rec of Object.values(recs)) {
    rec.argStrs = rec.inRefs.map(r => ref(r.up, r.port));
  }

  return { recs, order, errors, cyclic };
}

/* 模块节点在 __init__ 里的右值 */
function moduleInitOf(rec) {
  const d = rec.n.data;
  const I = asInt;
  switch (rec.type) {
    case 'embedding': return `nn.Embedding(${I(d.vocab_size, 1)}, ${I(d.embedding_dim, 1)})`;
    case 'linear': {
      const b = asBool(d.bias) ? '' : ', bias=False';
      return `nn.Linear(${I(d.in_features, 1)}, ${I(d.out_features, 1)}${b})`;
    }
    case 'call': return `${d.fname}()`;
    default: return `# 未知模块类型: ${rec.type}`;
  }
}

/* forward 里的语句（可能返回多行） */
function stmtOf(rec) {
  const d = rec.n.data;
  const v = rec.outVar;
  const A = i => rec.argStrs[i] ?? '…';
  switch (rec.type) {
    case 'literal': {
      const literal = App.literalCache.get(String(d.value ?? ''));
      if (!literal) throw Error('基础值尚未校验');
      if (literal.error) throw Error('基础值无效: ' + literal.error);
      return `${v} = ${literal.expression}`;
    }
    case 'tensor': return `${v} = torch.tensor(${String(d.values).trim() || '[0]'})`;
    case 'embedding':
      // 主画布上的独立层调用（模型画布里会作为模块提升到 __init__，不走这里）
      return `${v} = nn.Embedding(${asInt(d.vocab_size, 1)}, ${asInt(d.embedding_dim, 1)})(${A(0)})`;
    case 'linear': {
      const b = asBool(d.bias) ? '' : ', bias=False';
      return `${v} = nn.Linear(${asInt(d.in_features, 1)}, ${asInt(d.out_features, 1)}${b})(${A(0)})`;
    }
    case 'matmul': return `${v} = ${A(0)} @ ${A(1)}`;
    case 'binop':
      // rec.binopExpr 已按连线优先 + 手填字面量解析并左折叠加括号（buildScope）
      return `${v} = ${rec.binopExpr ?? '…'}`;
    case 'transpose': return `${v} = ${A(0)}.transpose(${asInt(d.dim0, -2)}, ${asInt(d.dim1, -1)})`;
    case 'softmax': return `${v} = torch.softmax(${A(0)}, dim=${asInt(d.dim, -1)})`;
    case 'print': {
      const arg = A(0);
      return d.label ? `print(${JSON.stringify(d.label)}, ${arg})` : `print(${arg})`;
    }
    case 'print_shape': {
      const arg = A(0);
      return d.label ? `print(${JSON.stringify(d.label)}, tuple(${arg}.shape))` : `print(tuple(${arg}.shape))`;
    }
    case 'getitem': {
      const idx = String(d.index ?? '0').trim() || '0';
      return `${v} = ${A(0)}[${idx}]`;
    }
    case 'call': {
      // 实例化 + 前向 两行：实例名 = 节点名，输出 = 节点名_out[2..]
      const lhs = rec.outVars.length > 1 ? rec.outVars.join(', ') : rec.outVars;
      return [
        `${rec.outVar} = ${d.fname}()`,
        `${lhs} = ${rec.outVar}(${rec.argStrs.join(', ')})`,
      ];
    }
    default: return `# 未知节点类型: ${rec.type}`;
  }
}

const dispName = rec => rec.n.data?.name || NODE_DEFS[rec.type]?.label || rec.type;

App.generatePython = function () {
  const L = [];
  const push = s => L.push(s);
  const errors = [];

  push('"""');
  push('Transformer 画布自动生成 — 请勿手改');
  push('在浏览器画布 (http://127.0.0.1:8000) 上编辑，本文件随之更新。');
  push('"""');
  push('');
  push('import torch');
  push('import torch.nn as nn');
  push('');
  push(`torch.manual_seed(${App.state.seed})`);

  // ---- 模型类定义 ----
  const validFns = new Set();
  if (App.state.fnOrder.length) {
    for (const fname of App.state.fnOrder) {
      const df = ED.exportData('fn:' + fname);
      const scope = buildScope(df, { isClass: true, reserved: [fname] });
      errors.push(...scope.errors.map(e => `[模型 ${fname}] ${e}`));

      const ins = Object.values(df).filter(n => n.data?.type === 'fn_input')
        .sort((a, b) => Number(a.id) - Number(b.id));
      const args = ins.map(n => scope.recs[String(n.id)].outVar);

      // 返回值：按 fn_output 端口顺序解析成字符串
      const retStrs = [];
      Object.values(df).filter(n => n.data?.type === 'fn_output')
        .sort((a, b) => Number(a.id) - Number(b.id))
        .forEach(fo => {
          const rec = scope.recs[String(fo.id)];
          for (const r of rec.inRefs) {
            const s = r.up ? (r.up.isModule ? r.up.moduleExpr : (r.up.outVars[r.port] ?? r.up.outVar)) : null;
            if (s) retStrs.push(s); else errors.push(`[模型 ${fname}] 有返回值端口未连接`);
          }
        });

      const fnValid = scope.errors.length === 0 && retStrs.length > 0 && args.length > 0;
      if (fnValid) validFns.add(fname);

      push('');
      push(`# ---- 模型「${fname}」：由模型画布生成 ----`);
      push(`class ${fname}(nn.Module):`);
      push('    # 初始化模型时调用：把零件挂到 self 上');
      push('    def __init__(self):');
      push('        super().__init__()');
      scope.moduleLines = [];
      for (const id of scope.order) {
        const rec = scope.recs[id];
        if (rec.isModule && rec.valid) scope.moduleLines.push(`        self.${rec.outVar} = ${moduleInitOf(rec)}`);
      }
      scope.moduleLines.forEach(l => push(l));
      push('');
      push('    # 前向传播：数据流过模型的路径');
      push(`    def forward(self, ${args.join(', ')}):`);
      const body = [];
      for (const id of scope.order) {
        const rec = scope.recs[id];
        if (rec.isModule) continue;   // 已提升到 __init__
        if (!rec.valid) {
          body.push(`        # ⚠ 跳过节点「${dispName(rec)}」：${rec.skipReason}`);
          continue;
        }
        const st = stmtOf(rec);
        (Array.isArray(st) ? st : [st]).forEach(s => body.push('        ' + s));
      }
      if (retStrs.length) body.push(`        return ${retStrs.join(', ')}`);
      if (!body.length) body.push('        pass');
      body.forEach(b => push(b));
    }
  }

  // ---- 主画布 ----
  push('');
  push('');
  push('# ================= 主画布 =================');
  const dfMain = ED.exportData('main');
  const mainScope = buildScope(dfMain, { isClass: false, reserved: App.state.fnOrder.slice() });
  errors.push(...mainScope.errors.map(e => `[主画布] ${e}`));
  let any = false;
  for (const id of mainScope.order) {
    const rec = mainScope.recs[id];
    if (rec.type === 'call' && !validFns.has(rec.n.data.fname)) {
      rec.valid = false;
      rec.skipReason = `模型「${rec.n.data.fname}」不完整或未定义`;
    }
    push('');
    if (!rec.valid) {
      push(`# ⚠ 跳过节点「${dispName(rec)}」：${rec.skipReason}`);
      continue;
    }
    push(`# 节点「${dispName(rec)}」`);
    const st = stmtOf(rec);
    (Array.isArray(st) ? st : [st]).forEach(s => push(s));
    any = true;
  }
  if (!any && !mainScope.order.length) push('# （画布是空的：从左侧拖入节点开始）');

  App.lastErrors = errors;
  return L.join('\n') + '\n';
};

/* ================= 同步 & 运行 ================= */
App.scheduleSync = function (immediate = false) {
  clearTimeout(App.syncTimer);
  App.syncTimer = setTimeout(App.syncNow, immediate ? 0 : 400);
};

App.literalCache = new Map();
ED.hooks.literalInfo = source => App.literalCache.get(String(source ?? ''));
/* Literal 节点与运算 op 的手填值共用同一后端安全校验（ast.literal_eval 白名单）。 */
App.validateLiterals = async function () {
  for (const key of Object.keys(ED.canvases)) {
    const sources = [];
    for (const n of Object.values(ED.exportData(key))) {
      if (n.data.type === 'literal') sources.push(String(n.data.value ?? ''));
      else if (n.data.type === 'binop') {
        for (const v of (Array.isArray(n.data.values) ? n.data.values : [])) {
          if (String(v ?? '').trim()) sources.push(String(v));
        }
      }
    }
    for (const source of sources) {
      if (App.literalCache.has(source)) continue;
      const response = await fetch('/api/literal', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({source}),
      });
      const result = await response.json();
      if (!response.ok && !result.error) throw Error('基础值校验服务不可用');
      App.literalCache.set(source, result);
    }
    for (const id of Object.keys(ED.exportData(key))) ED.refreshSummary(key, id);
  }
};

App.syncNow = async function () {
  const status = $('#sync-status');
  if (App.restoreFailed) return false;
  let code;
  try {
    await App.validateLiterals();
    code = App.generatePython();
  } catch (e) {
    status.textContent = '✗ 代码生成失败：' + e.message;
    status.className = 'err';
    console.error('generatePython 失败:', e);
    return;
  }
  App.renderCodePreview(code);
  const errors = App.lastErrors || [];

  try {
    if (errors.length) {
      status.textContent = errors.join('; '); status.className = 'err';
      return false;
    }
    const graph = {
      version: 3,
      seed: App.state.seed,
      main: ED.exportData('main'),
      functions: App.state.fnOrder.map(name => ({
        name, ...App.state.definitions[name], data: ED.exportData('fn:' + name),
      })),
    };
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph, code }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const saved = await res.json();
    if (!saved.ok) throw new Error('保存未确认');
    App.revision = saved.revision;
    const now = new Date().toTimeString().slice(0, 8);
    if (!errors.length) {
      status.textContent = '✓ 已同步 ' + now;
      status.className = 'ok';
    } else {
      status.textContent = `⚠ 已同步（${errors.length} 个问题，悬停查看） ` + now;
      status.className = 'warn';
    }
    return true;
  } catch (e) {
    status.textContent = '✗ 同步失败：' + e.message;
    status.className = 'err';
    return false;
  }
};

App.run = async function () {
  const btn = $('#btn-run');
  btn.disabled = true;
  btn.textContent = '运行中…';
  try {
    if (!(await App.syncNow())) throw new Error('保存或校验失败，已取消运行');
    const res = await fetch('/api/run', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({revision:App.revision}) });
    if (!res.ok) throw new Error('运行被拒绝: HTTP '+res.status);
    const r = await res.json();
    const out = $('#output-view');
    let text = `$ python generated_model.py\n`;
    if (r.stdout) text += r.stdout;
    if (r.stderr) text += '\n[stderr]\n' + r.stderr;
    text += `\n[退出码 ${r.returncode}]`;
    out.textContent = text;
    out.className = r.returncode === 0 ? 'ok' : 'bad';
  } catch (e) {
    $('#output-view').textContent = '运行请求失败: ' + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ 运行';
    $('#output-panel').classList.add('open');
  }
};

/* ================= 示例 ================= */
App.clearAllModels = function () {
  App.closeNodeDrawers();
  App.tensorChoice = {};
  App.hideInspector();
  for (const f of [...App.state.fnOrder]) App.removeFunctionRaw(f);
  ED.clearCanvas('main');
};

App.example1 = function () {   // 第 1 课：ShankeModel（nn.Module 类完整复现）
  if (!window.TC?.skipConfirm && !confirm('用「第 1 课：ShankeModel」示例替换当前所有画布？')) return;
  App.clearAllModels();

  // ---- 模型画布 ShankeModel：forward(ids) → embedding_matrix(ids) ----
  App.createFunctionRaw('ShankeModel');
  const F = 'fn:ShankeModel';
  const xin = ED.api.add(F, 'fn_input', 20, 180);
  ED.api.setParam(F, xin, { argname: 'ids' });
  const emb = ED.api.add(F, 'embedding', 300, 180);
  ED.api.setParam(F, emb, { name: 'embedding_matrix', vocab_size: 50275, embedding_dim: 768 });
  const fout = ED.api.add(F, 'fn_output', 600, 180);
  ED.api.connect(F, xin, 1, emb, 1);
  ED.api.connect(F, emb, 1, fout, 1);

  // ---- 主画布：ids → 模型调用 → 形状/切片打印 ----
  const ids = ED.api.add('main', 'tensor', 40, 200);
  ED.api.setParam('main', ids, { name: 'ids', values: '[1, 3, 5]' });
  const showIn = ED.api.add('main', 'print_shape', 300, 80);
  ED.api.setParam('main', showIn, { name: 'show_in_shape', label: '输入 ids 形状' });
  const model = ED.api.add('main', 'call', 300, 260, 'ShankeModel');
  ED.api.setParam('main', model, { name: 'model' });
  const showOut = ED.api.add('main', 'print_shape', 640, 180);
  ED.api.setParam('main', showOut, { name: 'show_out_shape', label: '输出的 token_embeddings 形状' });
  const pick = ED.api.add('main', 'getitem', 640, 320);
  ED.api.setParam('main', pick, { name: 'token_first', index: '0, :5' });
  const showFirst = ED.api.add('main', 'print', 940, 320);
  ED.api.setParam('main', showFirst, { name: 'show_first5', label: '第 1 个 token 前 5 个数' });

  ED.api.connect('main', ids, 1, showIn, 1);
  ED.api.connect('main', ids, 1, model, 1);
  ED.api.connect('main', model, 1, showOut, 1);
  ED.api.connect('main', model, 1, pick, 1);
  ED.api.connect('main', pick, 1, showFirst, 1);

  App.renderTabs();
  App.renderFnPalette();
  App.switchTab('main');
  App.scheduleSync(true);
};

App.example2 = function () {   // 自注意力（SelfAttention 类）
  if (!window.TC?.skipConfirm && !confirm('用「自注意力」示例替换当前所有画布？')) return;
  App.clearAllModels();

  // ---- 模型画布 self_attention ----
  App.createFunctionRaw('SelfAttention');
  const F = 'fn:SelfAttention';
  const xin = ED.api.add(F, 'fn_input', 20, 200);
  ED.api.setParam(F, xin, { argname: 'x' });

  const wq = ED.api.add(F, 'linear', 260, 60);
  ED.api.setParam(F, wq, { name: 'W_q', in_features: 4, out_features: 4, bias: false });
  const wk = ED.api.add(F, 'linear', 260, 170);
  ED.api.setParam(F, wk, { name: 'W_k', in_features: 4, out_features: 4, bias: false });
  const wv = ED.api.add(F, 'linear', 260, 400);
  ED.api.setParam(F, wv, { name: 'W_v', in_features: 4, out_features: 4, bias: false });

  const kt = ED.api.add(F, 'transpose', 520, 170);
  ED.api.setParam(F, kt, { name: 'K_T', dim0: -2, dim1: -1 });
  const scores = ED.api.add(F, 'matmul', 760, 110);
  ED.api.setParam(F, scores, { name: 'scores' });
  const weights = ED.api.add(F, 'softmax', 1000, 110);
  ED.api.setParam(F, weights, { name: 'weights', dim: -1 });
  const out = ED.api.add(F, 'matmul', 1230, 200);
  ED.api.setParam(F, out, { name: 'out' });

  const fout = ED.api.add(F, 'fn_output', 1470, 200);
  ED.addInputPort(F, fout);

  ED.api.connect(F, xin, 1, wq, 1);
  ED.api.connect(F, xin, 1, wk, 1);
  ED.api.connect(F, xin, 1, wv, 1);
  ED.api.connect(F, wk, 1, kt, 1);
  ED.api.connect(F, wq, 1, scores, 1);
  ED.api.connect(F, kt, 1, scores, 2);
  ED.api.connect(F, scores, 1, weights, 1);
  ED.api.connect(F, weights, 1, out, 1);
  ED.api.connect(F, wv, 1, out, 2);
  ED.api.connect(F, out, 1, fout, 1);
  ED.api.connect(F, weights, 1, fout, 2);

  // ---- 主画布 ----
  const ids = ED.api.add('main', 'tensor', 40, 200);
  ED.api.setParam('main', ids, { name: 'ids', values: '[0, 1, 2]' });
  const emb = ED.api.add('main', 'embedding', 300, 200);
  ED.api.setParam('main', emb, { name: 'embed', vocab_size: 3, embedding_dim: 4 });
  const attn = ED.api.add('main', 'call', 560, 180, 'SelfAttention');
  ED.api.setParam('main', attn, { name: 'attn' });
  const p1 = ED.api.add('main', 'print', 900, 100);
  ED.api.setParam('main', p1, { name: 'show_out', label: 'out' });
  const p2 = ED.api.add('main', 'print', 900, 280);
  ED.api.setParam('main', p2, { name: 'show_weights', label: 'weights' });
  ED.api.connect('main', ids, 1, emb, 1);
  ED.api.connect('main', emb, 1, attn, 1);
  ED.api.connect('main', attn, 1, p1, 1);
  ED.api.connect('main', attn, 2, p2, 1);

  App.renderTabs();
  App.renderFnPalette();
  App.switchTab('main');
  App.scheduleSync(true);
};

/* ================= 顶栏绑定 ================= */
App.bindHeader = function () {
  $('#btn-run').addEventListener('click', App.run);
  $('#btn-example1').addEventListener('click', App.example1);
  $('#btn-example2').addEventListener('click', App.example2);
  $('#btn-clear-output').addEventListener('click', () => {
    $('#output-view').textContent = '（还没有运行过）';
    $('#output-view').className = '';
  });
};

window.addEventListener('DOMContentLoaded', () => {
  App.init();
  window.TC = { App, ED, skipConfirm: false };   // 自动化/调试钩子
});
