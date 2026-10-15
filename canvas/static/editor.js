/*
 * editor.js —— 画布交互层
 * 负责：节点类型定义、Drawflow 实例管理、拖拽落点、节点/连线事件、底层增删改 API。
 * 状态管理、代码生成、同步在 app.js（通过 ED.hooks 回调解耦）。
 */

const ED = {
  hooks: {},            // app.js 注入: onCanvasChanged(), onSelect(key, id), onDeselect()
  canvases: {},         // key -> {editor, container}   key = 'main' | 'fn:<名字>'
  activeKey: 'main',
};

/* ------------------------------------------------------------------
 * 节点类型定义
 * params: 参数表（type: int / text / bool）
 * inNames / outNames: 端口提示文字（显示在节点卡片上）
 * onlyFn: 只能在函数画布中使用
 * ------------------------------------------------------------------ */
ED.NODE_DEFS = {
  literal: {
    label: '基础值 Literal', defaults: { name: 'value' },
    inputs: 0, outputs: 1, inNames: [], outNames: ['value'],
    params: [{ key: 'value', label: 'Python 字面量', type: 'text', multiline: true, def: '0', help: 'Literal 只接受字面量：数字、带引号字符串、True / False、None、列表、元组和字典（键为数字、字符串、布尔、None 或元组），可多行书写，不接受变量名或表达式。引用 dim 等已有变量请使用「变量引用 Variable Reference」。' }],
    summary: d => { const text = String(d.value ?? '0').replace(/\s+/g, ' ').trim(); return text.length > 48 ? text.slice(0, 48) + '…' : text; },
  },
  variable_ref: {
    label: '变量引用 Variable Reference', defaults: { name: 'ref' },
    inputs: 0, outputs: 1, inNames: [], outNames: ['value'],
    params: [{ key: 'variable', label: '变量名', type: 'text', def: '', help: '填写当前作用域的实际变量名（多返回值如 result_out2），不是表达式。普通函数可引用参数及局部输出；模型 forward 可引用输入和图输出。构造参数 dim 被引用时自动保存为 self.dim；self.dim 仅限声明过的构造参数，self.member 仅限已注册模型成员。不可跨模型或引用全局变量。' }],
    summary: d => String(d.variable || '变量名'),
  },
  tensor: {
    label: 'Tensor 输入', defaults: { name: 'ids' },
    inputs: 0, outputs: 1, inNames: [], outNames: ['out'],
    params: [{ key: 'values', label: '张量内容', type: 'text', def: '[1, 3, 5]' }],
    summary: d => String(d.values || ''),
  },
  embedding: {
    label: 'nn.Embedding', defaults: { name: 'embed' },
    inputs: 1, outputs: 1, inNames: ['ids'], outNames: ['向量'],
    params: [
      { key: 'vocab_size', label: 'vocab_size', type: 'int', def: 50275 },
      { key: 'embedding_dim', label: 'embedding_dim', type: 'int', def: 768 },
    ],
    summary: d => `(${d.vocab_size}, ${d.embedding_dim})`,
  },
  linear: {
    label: 'nn.Linear', defaults: { name: 'linear' },
    inputs: 1, outputs: 1, inNames: ['x'], outNames: ['out'],
    params: [
      { key: 'in_features', label: 'in_features', type: 'int', def: 4 },
      { key: 'out_features', label: 'out_features', type: 'int', def: 4 },
      { key: 'bias', label: 'bias', type: 'bool', def: false },
    ],
    summary: d => `(${d.in_features}, ${d.out_features}${d.bias ? '' : ', bias=False'})`,
  },
  matmul: {
    label: '矩阵乘 @', defaults: { name: 'matmul' },
    inputs: 2, outputs: 1, inNames: ['a', 'b'], outNames: ['结果'],
    params: [],
    summary: () => 'A @ B',
  },
  binop: {
    label: '运算 op', defaults: { name: 'result' },
    inputs: 2, outputs: 1, inNames: ['a', 'b'], outNames: ['结果'],
    dynamicInputs: true,   // 操作数个数 = 实际输入端口数（2–16）
    params: [],            // 运算符在相邻输入行之间编辑，不再有底部 op 字段
    summary: d => ED.binopFoldDisplay(d),
  },
  transpose: {
    label: '转置 transpose', defaults: { name: 'transposed' },
    inputs: 1, outputs: 1, inNames: ['x'], outNames: ['out'],
    params: [
      { key: 'dim0', label: 'dim a', type: 'int', def: -2 },
      { key: 'dim1', label: 'dim b', type: 'int', def: -1 },
    ],
    summary: d => `transpose(${d.dim0}, ${d.dim1})`,
  },
  softmax: {
    label: 'Softmax', defaults: { name: 'weights' },
    inputs: 1, outputs: 1, inNames: ['scores'], outNames: ['out'],
    params: [{ key: 'dim', label: 'dim', type: 'int', def: -1 }],
    summary: d => `dim=${d.dim}`,
  },
  ones_like: {
    label:'torch.ones_like', defaults:{name:'ones'}, inputs:1, outputs:1,
    inNames:['input'], outNames:['out'],
    params:[
      {key:'dtype',label:'dtype',type:'select',def:'inherit',options:['inherit','bool','float32','float64','float16','bfloat16','int64','int32'],help:'inherit 沿用输入类型；bool 生成 True，其他类型生成 1。只支持静态选择。'},
      {key:'device',label:'device',type:'select',def:'inherit',options:['inherit','cpu','cuda'],help:'inherit 沿用输入设备。cuda 需要可用的 CUDA GPU；只支持静态选择。'}],
    summary:d=>`dtype=${d.dtype}, device=${d.device}`,
  },
  triu: {
    label:'torch.triu', defaults:{name:'upper'}, inputs:2, outputs:1,
    inNames:['input','diagonal (可选)'],outNames:['out'],optionalInputs:[1],
    params:[{key:'diagonal',label:'diagonal',type:'int',def:0,help:'0 保留主对角线及上方；1 只保留严格上三角；负数保留更下方对角线。可连接 Literal 整数，连线优先；作用于最后两个维度。'}],
    summary:d=>`diagonal=${d.diagonal} (连线优先)`,
  },
  dtype: {
    label:'Tensor.to / bool',defaults:{name:'converted'},inputs:1,outputs:1,inNames:['input'],outNames:['out'],
    params:[{key:'dtype',label:'dtype',type:'select',def:'bool',options:['bool','float32','float64','float16','bfloat16','int64','int32'],help:'bool 转换：0 为 False，非零为 True。不会自动移动设备；只支持静态类型选择。'}],
    summary:d=>`to(torch.${d.dtype})`,
  },
  masked_fill: {
    label:'Tensor.masked_fill',defaults:{name:'masked'},inputs:3,outputs:1,
    inNames:['input','mask: bool','value (可选)'],outNames:['out'],optionalInputs:[2],
    params:[{key:'value',label:'value',type:'text',def:'-inf',help:'mask 为 True 的位置替换为此标量，False 保留原值。支持数字、inf、-inf、nan；可连接 Literal 数字（连线优先）。mask 必须是 bool 且可广播；整数输入不能填 -inf。'}],
    summary:d=>`value=${d.value} (连线优先)`,
  },
  print: {
    label: '打印 print', defaults: { name: 'show' },
    inputs: 1, outputs: 0, inNames: ['x'], outNames: [],
    params: [{ key: 'label', label: '前缀标签', type: 'text', def: '' }],
    summary: d => (d.label ? `"${d.label}" =` : ''),
  },
  print_shape: {
    label: '打印形状 shape', defaults: { name: 'show_shape' },
    inputs: 1, outputs: 0, inNames: ['x'], outNames: [],
    params: [{ key: 'label', label: '前缀标签', type: 'text', def: '' }],
    summary: d => (d.label ? `"${d.label}": ` : '') + 'tuple(x.shape)',
  },
  getitem: {
    label: '索引取值 [..]', defaults: { name: 'picked' },
    inputs: 1, outputs: 1, inNames: ['x'], outNames: ['out'],
    params: [{ key: 'index', label: '索引表达式', type: 'text', def: '0' }],
    summary: d => `x[${d.index}]`,
  },
  fn_input: {
    label: '模型输入', defaults: { name: 'in' }, onlyFn: true,
    inputs: 0, outputs: 1, inNames: [], outNames: ['out'],
    params: [{ key: 'argname', label: '参数名', type: 'text', def: 'x' }],
    summary: d => `参数: ${d.argname}`,
  },
  fn_output: {
    label: '模型输出', defaults: { name: 'return' }, onlyFn: true,
    inputs: 1, outputs: 0, inNames: ['返回值 1'], outNames: [],
    params: [],
    summary: () => '',
    dynamicInputs: true,   // 可在参数面板添加返回值端口
  },
  // 模型调用节点（由"我的模型"生成）: data = { fname, inLabels, outLabels }
  call: {
    label: '模型调用', defaults: { name: 'model' },
    inputs: 1, outputs: 1, inNames: [], outNames: [],   // 实际数量由 data 决定
    params: [],
    summary: d => `${d.fname}(${(d.inLabels || []).join(', ')})`,
    dynamic: true,
  },
};

/* ---------------- 运算 op 多输入工具 ----------------
 * 数据模型：操作数个数 = 实际输入端口数（2–16，从上至下 a,b,c…）；
 * data.ops[k] 是第 k 与第 k+1 个操作数之间的运算符；data.values[i] 是第 i 个
 * 输入的手填字面量（连线优先，未连线时使用）。旧图只有 data.op：所有段沿用它。 */
ED.BIN_OPS = new Set(['+', '-', '*', '/', '//', '%', '@', '**',
  '>', '<', '>=', '<=', '==', '!=', '&', '|', '^', '<<', '>>']);
ED.BINOP_MIN = 2;
ED.BINOP_MAX = 16;
ED.binopName = i => String.fromCharCode(97 + Math.min(i, 25));   // a..p（超 26 不支持）
ED.binopState = function (n) {
  const d = n?.data || {};
  const count = Math.max(ED.BINOP_MIN, Object.keys(n?.inputs || {}).length || ED.BINOP_MIN);
  const legacy = String(d.op ?? '+').trim() || '+';
  const opsIn = Array.isArray(d.ops) ? d.ops : [];
  const ops = [];
  for (let k = 0; k < count - 1; k++) ops.push(opsIn[k] == null ? legacy : String(opsIn[k]));
  const valuesIn = Array.isArray(d.values) ? d.values : [];
  const values = [];
  for (let i = 0; i < count; i++) values.push(valuesIn[i] == null ? '' : String(valuesIn[i]));
  return { count, ops, values };
};
ED.binopFoldDisplay = function (d) {
  const opsIn = Array.isArray(d?.ops) && d.ops.length ? d.ops : [d?.op ?? '+'];
  const ops = opsIn.map(o => String(o ?? '').trim());
  let text = ED.binopName(0);
  for (let k = 0; k < ops.length; k++) {
    const next = ED.binopName(k + 1);
    text = k === 0 && ops.length === 1
      ? `${text} ${ops[k] || '?'} ${next}`
      : `(${text} ${ops[k] || '?'} ${next})`;
  }
  return text;
};

ED.defOf = n => ED.NODE_DEFS[n?.data?.type] || null;

/* Drawflow 0.0.59 没有 getNodeById，直接读内部数据 */
ED.getNode = function (key, id) {
  const ed = ED.canvases[key]?.editor;
  if (!ed) return null;
  return ed.drawflow.drawflow[ed.module].data[id] || null;
};
ED.inCount = n => {
  const def = ED.defOf(n);
  if (def?.dynamic || def?.dynamicInputs) return Object.keys(n.inputs || {}).length; // Actual dynamic ports
  return def ? def.inputs : 0;
};
ED.outCount = n => {
  const def = ED.defOf(n);
  if (n.data?.type === 'call') return (n.data.outLabels || ['out']).length;
  return def ? def.outputs : 0;
};

/* ---------------- 节点卡片 HTML ---------------- */
ED.nodeHTML = function () {
  // Imported HTML is rebuilt from this template; user text is assigned with textContent.
  return '<div class="nd-head"><span class="nd-title"></span>' +
    '<button type="button" class="nd-toggle" aria-label="展开运行结果" title="展开运行结果" aria-expanded="false"><span aria-hidden="true">⌄</span></button></div>' +
    '<div class="nd-summary"><div class="nd-callable"></div><div class="nd-arguments"></div></div><div class="nd-pins"></div><div class="nd-warning"></div>' +
    '<div class="nd-params" aria-label="节点参数"></div>' +
    '<section class="nd-drawer" hidden aria-label="运行结果">' +
    '<div class="nd-runtime"><div class="panel-title">节点运行值 <span class="nd-runtime-status" role="status"></span></div><div class="nd-runtime-view"></div></div></section>';
};

ED.portInfo = function (key, n, side, index) {
  const d = n.data, def = ED.defOf(n);
  const dynamic = ['call', 'invoke', 'function_call'].includes(d.type);
  let labels = def?.[side === 'input' ? 'inNames' : 'outNames'] || [];
  if (dynamic) {
    const sig = ED.hooks.fnMeta?.(d.fname) || {inputs: [], outputs: []};
    labels = side === 'input' ? [...(d.type === 'invoke' ? ['instance'] : []), ...sig.inputs] : sig.outputs;
  } else if (d.type === 'create') labels = side === 'output' ? ['instance'] : [];
  else if (d.type === 'fn_input') labels = [d.argname || 'x'];
  else if (d.type === 'fn_output') labels = Object.keys(n.inputs).map((_, i) => `return ${i + 1}`);
  else if (side === 'input') {
    labels = ({embedding:['input'], linear:['input'], matmul:['input','other'], softmax:['input'], print:['object'], print_shape:['object'], transpose:['self'], getitem:['self']})[d.type] || labels;
    if (d.type === 'binop') labels = Array.from({length: Math.max(Object.keys(n.inputs || {}).length, ED.BINOP_MIN)}, (_, i) => ED.binopName(i));
  }
  const obsolete = dynamic && index >= labels.length;
  const name = obsolete ? `废弃端口 ${index + 1}` : (labels[index] || (side === 'output' ? 'result' : `参数 ${index + 1}`));
  let type = ['embedding','linear','matmul','transpose','softmax','tensor','ones_like','triu','dtype','masked_fill'].includes(d.type) ? 'Tensor' : 'Any';
  if (def?.spec) {
    // 自定义节点：端口类型显示取规范里的类型提示（仅展示，不做编译校验）。
    type = side === 'input' ? (def.spec.inputs[index]?.type || 'Any') : (def.spec.output?.type || 'Tensor');
  }
  if(side==='input' && d.type==='triu' && index===1) type='int';
  if(side==='input' && d.type==='masked_fill' && index===1) type='BoolTensor';
  if(side==='input' && d.type==='masked_fill' && index===2) type='number';
  if (name === 'instance') type = 'Module';
  if (d.type === 'literal') type = ED.hooks.literalInfo?.(d.value)?.type || '未校验';
  if (side === 'input' && !obsolete && type === 'Any') {
    const connection = n.inputs[`input_${index + 1}`]?.connections?.[0];
    const up = connection && ED.getNode(key, connection.node);
    if (up?.data.type === 'literal') type = ED.hooks.literalInfo?.(up.data.value)?.type || '未校验';
  }
  return {name, type, obsolete};
};

/* 黄色功能标题（原参数表单 ins-title，现显示在节点名标题框正下方的 .nd-callable）：
 * 一律用 NODE_DEFS 的准确 label（如「变量引用 Variable Reference」）；
 * call/create/invoke/function_call 的动态类型 label 由 editor.js/architecture.js 的定义给出
 * （模型调用 / 创建模型实例 / 调用已有实例 / 普通函数调用）。 */
ED.functionTitle = function (d) {
  return ED.NODE_DEFS[d.type]?.label || d.type;
};

ED.refreshSummary = function (key, id) {
  const n = ED.getNode(key, id);
  const el = ED.canvases[key].container.querySelector(`[id="node-${id}"]`);
  if (!n || !el) return;
  const canvas = ED.canvases[key];
  if (canvas.observer && !canvas.observedNodes?.has(el)) {
    canvas.observedNodes ||= new WeakSet();
    canvas.observedNodes.add(el); canvas.observer.observe(el);
  }
  const d = n.data, def = ED.defOf(n);
  const title = String(d.type === 'fn_input' ? (d.argname || 'x') : (d.name || def?.defaults?.name || d.type));
  const titleElement = el.querySelector('.nd-title');
  if (!titleElement.querySelector('.nd-name-input')) titleElement.textContent = title;
  titleElement.title = title + '（点击编辑）';
  titleElement.setAttribute('role', 'button');
  titleElement.setAttribute('tabindex', '0');
  titleElement.setAttribute('aria-label', (d.type === 'fn_input' ? '编辑参数名 ' : '编辑节点名 ') + title);
  el.querySelector('.nd-callable').textContent = ED.functionTitle(d);
  let parameters = def?.summary?.(d) || '';
  if (d.type === 'fn_input' || d.type === 'fn_output') parameters = '';
  if (['call', 'invoke', 'function_call'].includes(d.type)) {
    parameters = `(${(d.inLabels || []).join(', ')})`;
    if (d.type === 'call' && d.init_args) parameters += ` · init(${d.init_args})`;
  }
  if (d.type === 'create') parameters = `(${d.init_args || ''})`;
  const argumentsRow = el.querySelector('.nd-arguments');
  argumentsRow.textContent = parameters; argumentsRow.hidden = !parameters;
  const rows = el.querySelector('.nd-pins');
  const count = Math.max(Object.keys(n.inputs || {}).length, Object.keys(n.outputs || {}).length);
  const isBinop = d.type === 'binop';
  const binop = isBinop ? ED.binopState(n) : null;
  let obsolete = false;
  /* 结构签名：端口存在性/连线/标签/废弃状态任一变化才重建行；
   * 仅手填值、运算符变化时就地同步控件，正在编辑的控件不动（不重绘、不失焦）。 */
  const pinSigParts = [d.type, count];
  for (let i = 0; i < count; i++) {
    for (const side of ['input', 'output']) {
      const p = `${side}_${i + 1}`, pin = el.querySelector(`.${side}s > .${p}`);
      if (!pin) { pinSigParts.push('-'); continue; }
      const info = ED.portInfo(key, n, side, i);
      obsolete ||= info.obsolete;   // 同步路径也需要准确的废弃警告
      pinSigParts.push(`${info.name}:${info.type}:${info.obsolete ? 1 : 0}:${n[side + 's'][p].connections.length}`);
    }
  }
  const pinSig = pinSigParts.join('|');
  const buildOperandEditor = i => {
    const input = document.createElement('input');
    input.type = 'text'; input.className = 'nd-arg-value'; input.dataset.operand = String(i);
    input.setAttribute('aria-label', `输入 ${ED.binopName(i)} 手填值`);
    if (n.inputs[`input_${i + 1}`]?.connections?.length) {
      // 接线优先：已连接的输入禁用手填并注明来自上游；手填值保留，断开后恢复。
      input.disabled = true; input.value = '';
      input.placeholder = '来自上游连线';
      input.title = `输入 ${ED.binopName(i)} 已连接，使用上游值；手填值已保留，断开后恢复`;
    } else {
      input.value = binop.values[i] ?? '';
      input.placeholder = '可手填字面量';
      input.title = `输入 ${ED.binopName(i)} 未连线时使用此 Python 字面量（Literal 校验），连线优先`;
      input.addEventListener('input', () => {
        const fresh = ED.getNode(key, id);
        if (!fresh) return;
        const values = ED.binopState(fresh).values.slice();
        values[i] = input.value;
        ED.updateNode(key, id, { values });
        App.scheduleSync();
      });
      input.addEventListener('focus', () => ED.hooks.onSelect?.(key, id));
    }
    return input;
  };
  const buildOpRow = k => {
    const row = document.createElement('div'); row.className = 'nd-op-row';
    const select = document.createElement('select');
    select.className = 'nd-op'; select.dataset.op = String(k);
    select.setAttribute('aria-label', `输入 ${ED.binopName(k)} 与 ${ED.binopName(k + 1)} 之间的运算符`);
    select.title = `输入 ${ED.binopName(k)} 与 ${ED.binopName(k + 1)} 之间的运算符；从上至下按左折叠计算`;
    const current = String(binop.ops[k] ?? '+');
    const options = [...ED.BIN_OPS];
    if (!ED.BIN_OPS.has(current)) options.unshift(current);   // 旧图非法/空运算符仍显示并可改
    for (const item of options) {
      const opt = document.createElement('option');
      opt.value = item; opt.textContent = item === '' ? '（空）' : item;
      select.appendChild(opt);
    }
    select.value = current;
    select.addEventListener('change', () => {
      const fresh = ED.getNode(key, id);
      if (!fresh) return;
      const ops = ED.binopState(fresh).ops.slice();
      ops[k] = select.value;
      ED.updateNode(key, id, { ops, op: ops[0] ?? '+' });   // data.op 同步首段，兼容旧读取方
      App.scheduleSync();
    });
    row.appendChild(select);
    return row;
  };
  const buildRow = i => {
    const row = document.createElement('div'); row.className = 'nd-pin-row';
    for (const side of ['input', 'output']) {
      const label = document.createElement('span'); label.className = 'nd-pin-label pin-' + side;
      const p = `${side}_${i + 1}`, pin = el.querySelector(`.${side}s > .${p}`);
      if (pin) {
        const info = ED.portInfo(key, n, side, i);
        const text = `${info.name}: ${info.type}`;
        label.textContent = text; label.title = text;
        label.classList.toggle('obsolete', info.obsolete); obsolete ||= info.obsolete;
        pin.title = text; pin.setAttribute('aria-label', text);
        pin.classList.toggle('connected', !!n[side + 's'][p].connections.length);
        pin.classList.toggle('obsolete', info.obsolete);
        row.dataset[side === 'input' ? 'in' : 'out'] = p;   // 端口 → 实际行映射，供 redraw 按行中心定位
      }
      if (side === 'input' && isBinop && pin) {
        row.classList.add('nd-arg-row');
        const cell = document.createElement('div'); cell.className = 'nd-pin-input-cell';
        cell.appendChild(label);
        cell.appendChild(buildOperandEditor(i));
        row.appendChild(cell);
      } else {
        row.appendChild(label);
      }
    }
    return row;
  };
  if (rows.dataset.sig === pinSig && rows.children.length) {
    const focused = document.activeElement;
    rows.querySelectorAll('.nd-arg-value').forEach(input => {
      if (input.disabled || input === focused) return;
      input.value = binop?.values[Number(input.dataset.operand)] ?? '';
    });
    rows.querySelectorAll('.nd-op').forEach(select => {
      if (select === focused) return;
      const current = String(binop?.ops[Number(select.dataset.op)] ?? '+');
      if (Array.from(select.options || []).some(option => option.value === current)) select.value = current;
    });
  } else {
    rows.replaceChildren();
    for (let i = 0; i < count; i++) {
      if (isBinop && i > 0) rows.appendChild(buildOpRow(i - 1));   // 运算符居相邻输入行之间
      rows.appendChild(buildRow(i));
    }
    rows.dataset.sig = pinSig;
  }
  el.querySelector('.nd-warning').textContent = obsolete ? '接口已改变：废弃端口及连线已保留，请重连后重建调用节点。' : '';
  const form = el.querySelector('.nd-params');
  if (form && !form.dataset.mounted && ED.hooks.onFormMount) {
    form.dataset.mounted = 'true';
    ED.hooks.onFormMount(key, id, form);
  } else if (form?.dataset.mounted && ED.hooks.onFormSync) {
    // 数据在外部被改（__init__ 面板、示例搭建）时同步控件值，不动正在编辑的焦点。
    ED.hooks.onFormSync(key, id, form);
  }
  ED.redraw(key);
};

ED.redraw = function (key) {
  const canvas = ED.canvases[key];
  if (!canvas || canvas.redrawPending) return;
  canvas.redrawPending = requestAnimationFrame(() => {
    canvas.redrawPending = null;
    if (!canvas.container.isConnected || !canvas.container.getBoundingClientRect().width) return;
    for (const id of Object.keys(canvas.editor.drawflow.drawflow[canvas.editor.module].data)) {
      const node = canvas.container.querySelector(`[id="node-${id}"]`);
      if (node) {
        const nodeRect = node.getBoundingClientRect();
        const scale = nodeRect.height / node.offsetHeight;
        const rows = Array.from(node.querySelectorAll('.nd-pin-row'));
        for (const side of ['inputs', 'outputs']) {
          const attr = side === 'inputs' ? 'in' : 'out';
          Array.from(node.querySelector('.' + side).children).forEach((pin, index) => {
            // 运算 op 在输入行之间插入算子行：必须按各端口实际所在行（data-in/out）定位，
            // 不能假定行下标等于端口下标。
            const wanted = `${side.slice(0, -1)}_${index + 1}`;
            const row = rows.find(r => r.dataset?.[attr] === wanted) || rows[index];
            if (!row || !scale) return;
            const rect = row.getBoundingClientRect();
            pin.style.top = `${(rect.top + rect.height / 2 - nodeRect.top) / scale - node.clientTop - pin.offsetHeight / 2}px`;
          });
        }
      }
      canvas.editor.updateConnectionNodes('node-' + id);
    }
  });
};

/* ---------------- Internal single-node clipboard ---------------- */
ED.clearClipboard = function () {
  ED.clipboard = null;
  ED.closeContextMenu();
};
ED.copyNode = function (key, id) {
  const node = ED.getNode(key, id), def = ED.defOf(node);
  if (!node || !def) return false;
  // Copy only graph settings, never imported HTML, identifiers, edges or runtime values.
  const data = {};
  for (const field of new Set(['type', 'name', 'fname', 'init_args', 'inLabels', 'outLabels', 'values', 'ops', 'op', ...def.params.map(p => p.key)])) {
    if (Object.hasOwn(node.data, field)) data[field] = node.data[field];
  }
  ED.clipboard = JSON.parse(JSON.stringify({data,
    inputs: Object.keys(node.inputs || {}).length, outputs: Object.keys(node.outputs || {}).length,
    drawerOpen: !!ED.hooks.drawerOpen?.(key, id), project: ED.hooks.projectId?.() ?? null}));
  return true;
};
ED.canPaste = function (key) {
  const clip = ED.clipboard, def = ED.NODE_DEFS[clip?.data?.type];
  if (!clip || !def || !ED.canvases[key] || clip.project !== (ED.hooks.projectId?.() ?? null)) return false;
  if (def.onlyFn && !key.startsWith('fn:')) return false;
  if (['call', 'create', 'invoke', 'function_call'].includes(clip.data.type)) {
    if (!ED.hooks.targetExists?.(clip.data.fname, clip.data.type)) return false;
  }
  return true;
};
ED.pasteNode = function (key, x, y) {
  if (!ED.canPaste(key)) return null;
  const clip = ED.clipboard, data = JSON.parse(JSON.stringify(clip.data));
  const field = data.type === 'fn_input' ? 'argname' : 'name';
  const base = String(data[field] || (field === 'argname' ? 'x' : ED.NODE_DEFS[data.type].defaults.name));
  const used = new Set(Object.values(ED.exportData(key)).flatMap(n => [n.data.name, n.data.argname]).filter(Boolean));
  data[field] = ED.hooks.copyName ? ED.hooks.copyName(key, base, used) : base + '_copy';
  const id = ED.canvases[key].editor.addNode(data.type, clip.inputs, clip.outputs,
    Math.round(x), Math.round(y), `node-${data.type}`, data, ED.nodeHTML());
  ED.refreshSummary(key, id);
  ED.hooks.onPaste?.(key, id, clip.drawerOpen);
  return id;
};
ED.closeContextMenu = function () {
  if (!ED.contextMenu) return;
  const {menu, outside, keyboard, dismiss} = ED.contextMenu;
  menu.remove();
  document.removeEventListener('pointerdown', outside, true);
  document.removeEventListener('keydown', keyboard, true);
  window.removeEventListener('resize', dismiss);
  window.removeEventListener('blur', dismiss);
  ED.contextMenu = null;
};
ED.connectionInfo = function (element) {
  const connection = element?.closest?.('svg.connection');
  if (!connection) return null;
  const classes = Array.from(connection.classList);
  const outputNode = classes.find(c => c.startsWith('node_out_node-'));
  const inputNode = classes.find(c => c.startsWith('node_in_node-'));
  const outputPort = classes.find(c => /^output_\d+$/.test(c));
  const inputPort = classes.find(c => /^input_\d+$/.test(c));
  if (!outputNode || !inputNode || !outputPort || !inputPort) return null;
  return {outputId: outputNode.slice('node_out_node-'.length),
    inputId: inputNode.slice('node_in_node-'.length), outputPort, inputPort};
};
ED.openContextMenu = function (key, id, clientX, clientY, connection = null) {
  ED.closeContextMenu();
  const editor = ED.canvases[key].editor;
  const pre = editor.precanvas.getBoundingClientRect();
  const x = (clientX - pre.left) * precanvasW(editor) / pre.width;
  const y = (clientY - pre.top) * precanvasH(editor) / pre.height;
  const menu = document.createElement('div'); menu.className = 'node-context-menu';
  menu.setAttribute('role', 'menu'); menu.setAttribute('aria-label', connection ? '连接线菜单' : '节点菜单');
  const items = [];
  function action(label, disabled, callback) {
    const button = document.createElement('button'); button.type = 'button';
    button.setAttribute('role', 'menuitem'); button.textContent = label; button.disabled = disabled;
    button.onclick = () => { ED.closeContextMenu(); callback(); };
    menu.appendChild(button); items.push(button);
  }
  if (connection) {
    action('删除连接', false, () => editor.removeSingleConnection(
      connection.outputId, connection.inputId, connection.outputPort, connection.inputPort));
  } else {
    if (id !== null) action('复制节点', false, () => ED.copyNode(key, id));
    action('粘贴节点', !ED.canPaste(key), () => ED.pasteNode(key, x, y));
    if (id !== null) action('删除节点', false, () => {
      if (ED.getNode(key, id)) ED.removeNode(key, id);
    });
  }
  document.body.appendChild(menu);
  const bounds = menu.getBoundingClientRect();
  menu.style.left = `${Math.max(4, Math.min(clientX, window.innerWidth - bounds.width - 4))}px`;
  menu.style.top = `${Math.max(4, Math.min(clientY, window.innerHeight - bounds.height - 4))}px`;
  const enabled = items.filter(button => !button.disabled);
  const outside = e => { if (!menu.contains(e.target)) ED.closeContextMenu(); };
  const dismiss = () => ED.closeContextMenu();
  const keyboard = e => {
    if (e.key === 'Escape' || e.key === 'Tab') {
      ED.closeContextMenu();
      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); }
    } else if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key) && enabled.length) {
      e.preventDefault(); e.stopPropagation();
      let index = enabled.indexOf(document.activeElement);
      if (e.key === 'Home') index = 0;
      else if (e.key === 'End') index = enabled.length - 1;
      else index = (index + (e.key === 'ArrowDown' ? 1 : -1) + enabled.length) % enabled.length;
      enabled[index].focus();
    }
  };
  ED.contextMenu = {menu, outside, keyboard, dismiss};
  document.addEventListener('pointerdown', outside, true);
  document.addEventListener('keydown', keyboard, true);
  window.addEventListener('resize', dismiss);
  window.addEventListener('blur', dismiss);
  enabled[0]?.focus();
};

/* ---------------- 创建/获取编辑器 ---------------- */
ED.makeEditor = function (key, containerId) {
  const container = document.getElementById(containerId);
  const editor = new Drawflow(container);
  editor.start();
  editor.draggable_inputs = false;
  // Inline controls must never enter Drawflow's drag, delete or selection handlers.
  // The title bar stays draggable; only the editing input inside it is intercepted.
  const inlineControl = e => e.target.closest?.('.nd-params, .nd-drawer, .nd-toggle, input, select, textarea, button, [contenteditable="true"]');
  // Distinguish a click from a node drag that started on the title bar.
  let titleDown = null;
  container.addEventListener('mousedown', e => {
    if (!inlineControl(e) && e.target.closest?.('.drawflow-node')) titleDown = {x: e.clientX || 0, y: e.clientY || 0};
  }, true);
  for (const event of ['mousedown', 'touchstart', 'pointerdown', 'keydown', 'dblclick', 'contextmenu', 'wheel']) {
    container.addEventListener(event, e => {
      if (inlineControl(e)) e.stopPropagation();
    }, true);
  }
  container.addEventListener('contextmenu', e => {
    // Capture before Drawflow installs its destructive context-menu delete control.
    if (inlineControl(e)) return;
    e.preventDefault(); e.stopImmediatePropagation();
    const node = e.target.closest?.('.drawflow-node');
    const connection = ED.connectionInfo(e.target);
    if (connection) ED.openContextMenu(key, null, e.clientX, e.clientY, connection);
    else ED.openContextMenu(key, node ? node.id.slice(5) : null, e.clientX, e.clientY);
  }, true);
  container.addEventListener('click', e => {
    const toggle = e.target.closest?.('.nd-toggle');
    if (toggle) ED.hooks.onDrawerToggle?.(key, toggle.closest('.drawflow-node').id.slice(5));
    const title = e.target.closest?.('.nd-title');
    const stillDown = titleDown && Math.abs((e.clientX || 0) - titleDown.x) < 4 && Math.abs((e.clientY || 0) - titleDown.y) < 4;
    titleDown = null;
    if (title && !e.target.closest?.('input') && stillDown) ED.hooks.onNameEdit?.(key, title.closest('.drawflow-node').id.slice(5));
    const drawer = e.target.closest?.('.nd-drawer, .nd-params');
    if (drawer) ED.hooks.onSelect?.(key, drawer.closest('.drawflow-node').id.slice(5));
  });
  container.addEventListener('keydown', e => {
    const title = e.target.closest?.('.nd-title');
    if (title && !e.target.closest?.('input') && ['Enter', ' '].includes(e.key)) {
      e.preventDefault();
      ED.hooks.onNameEdit?.(key, title.closest('.drawflow-node').id.slice(5));
    }
  });
  container.addEventListener('focusin', e => {
    const drawer = e.target.closest?.('.nd-drawer, .nd-params');
    if (drawer) ED.hooks.onSelect?.(key, drawer.closest('.drawflow-node').id.slice(5));
  });

  editor.on('nodeCreated', () => ED.hooks.onCanvasChanged && ED.hooks.onCanvasChanged());
  editor.on('nodeRemoved', id => {
    ED.hooks.onRemove?.(key, id);
    ED.hooks.onCanvasChanged && ED.hooks.onCanvasChanged();
  });
  editor.on('nodeMoved', () => { ED.hooks.onCanvasChanged && ED.hooks.onCanvasChanged(); });
  const onConnChange = info => {
    // 连线变化 → 两端卡片的输入/输出展示要跟着更新（如"返回: xxx"）
    if (info && info.output_id != null) ED.refreshSummary(key, info.output_id);
    if (info && info.input_id != null) ED.refreshSummary(key, info.input_id);
    ED.hooks.onCanvasChanged && ED.hooks.onCanvasChanged();
  };
  editor.on('connectionCreated', onConnChange);
  editor.on('connectionRemoved', onConnChange);
  editor.on('nodeSelected', id => ED.hooks.onSelect && ED.hooks.onSelect(key, id));
  editor.on('nodeUnselected', () => {
    if (ED.activeKey === key) ED.hooks.onDeselect?.();
  });
  container.addEventListener('mousedown', e => {
    // 点到画布空白处 → 取消选择
    if (e.target === container || e.target.classList?.contains('drawflow')) {
      ED.hooks.onDeselect && ED.hooks.onDeselect();
    }
  });

  // 拖拽放置
  container.addEventListener('dragover', e => e.preventDefault());
  container.addEventListener('drop', e => {
    e.preventDefault();
    const type = e.dataTransfer.getData('node-type');
    if (!type) return;
    const fname = e.dataTransfer.getData('fn-name') || '';
    const rect = container.getBoundingClientRect();
    const pre = editor.precanvas.getBoundingClientRect();
    const x = (e.clientX - pre.left) * (precanvasW(editor) / pre.width);
    const y = (e.clientY - pre.top) * (precanvasH(editor) / pre.height);
    ED.addNodeTo(key, type, x - 90, y - 40, fname);
  });

  ED.canvases[key] = { editor, container };
  const observer = new ResizeObserver(() => ED.redraw(key));
  observer.observe(container);
  ED.canvases[key].observer = observer;
  return editor;
};

function precanvasW(ed) { return ed.precanvas.clientWidth; }
function precanvasH(ed) { return ed.precanvas.clientHeight; }

/* ---------------- 节点增删改 ---------------- */
ED.addNodeTo = function (key, type, x, y, fname = '') {
  const { editor } = ED.canvases[key];
  const def = ED.NODE_DEFS[type];

  let data, numIn, numOut, cssClass;
  if (type === 'call') {
    const meta = ED.hooks.fnMeta && ED.hooks.fnMeta(fname);
    data = {
      type: 'call', fname,
      name: fname,
      inLabels: meta ? meta.inputs : ['x'],
      outLabels: meta ? meta.outputs : ['out'],
    };
    numIn = data.inLabels.length;
    numOut = Math.max(1, data.outLabels.length);
    cssClass = 'node-call';
  } else {
    data = { type, name: def.defaults.name };
    for (const p of def.params) data[p.key] = p.def;
    numIn = def.inputs;
    numOut = def.outputs;
    cssClass = `node-${type}`;
  }

  const id = editor.addNode(type, numIn, numOut, Math.round(x), Math.round(y), cssClass, data, ED.nodeHTML({ data }));
  ED.refreshSummary(key, id);
  return id;
};

ED.connect = function (key, outId, outPort, inId, inPort) {
  ED.canvases[key].editor.addConnection(String(outId), String(inId), `output_${outPort}`, `input_${inPort}`);
};

ED.updateNode = function (key, id, patch) {
  const { editor } = ED.canvases[key];
  // 0.0.59 的 updateNodeDataFromId 是整体替换而非合并，必须自己带上旧数据
  const cur = ED.getNode(key, id);
  const merged = Object.assign({}, cur?.data, patch);
  editor.updateNodeDataFromId(id, merged);
  ED.refreshSummary(key, id);
};

ED.removeNode = function (key, id) {
  ED.canvases[key].editor.removeNodeId('node-' + id);
};

ED.addInputPort = function (key, id) {
  ED.canvases[key].editor.addNodeInput(id);
  ED.refreshSummary(key, id);
};

/* 页面内 DOM 确认框：破坏性操作（如删除已连线的输入）不用原生 confirm/prompt。 */
ED.confirmDialog = function (title, message) {
  return new Promise(resolve => {
    const dialog = document.createElement('dialog');
    dialog.className = 'ed-confirm';
    dialog.setAttribute('aria-labelledby', 'ed-confirm-title');
    const form = document.createElement('form'); form.method = 'dialog';
    const heading = document.createElement('h3'); heading.id = 'ed-confirm-title'; heading.textContent = title;
    const text = document.createElement('p'); text.textContent = message;
    const actions = document.createElement('div'); actions.className = 'dialog-actions';
    const cancel = document.createElement('button'); cancel.type = 'button'; cancel.textContent = '取消';
    const ok = document.createElement('button'); ok.type = 'submit'; ok.className = 'primary'; ok.textContent = '确定';
    cancel.onclick = () => dialog.close('cancel');
    form.onsubmit = e => { e.preventDefault(); dialog.close('ok'); };
    dialog.onclose = () => { const yes = dialog.returnValue === 'ok'; dialog.remove(); resolve(yes); };
    actions.appendChild(cancel); actions.appendChild(ok);
    form.appendChild(heading); form.appendChild(text); form.appendChild(actions);
    dialog.appendChild(form);
    document.body.appendChild(dialog);
    dialog.showModal();
    ok.focus();
  });
};

/* 运算 op：＋ 末尾增加输入（新输入继承上一段运算符）；－ 移除最后输入（最少 2 个；
 * 已连线的末端输入先经页面内确认，移除时只删除该输入及其连线，不影响前面的连线）。 */
ED.addOperand = function (key, id) {
  const n = ED.getNode(key, id);
  if (!n || n.data.type !== 'binop') return false;
  const count = Object.keys(n.inputs || {}).length;
  if (count >= ED.BINOP_MAX) return false;
  const st = ED.binopState(n);
  ED.canvases[key].editor.addNodeInput(id);
  const ops = st.ops.slice(); ops.push(st.ops[st.ops.length - 1] ?? '+');
  const values = st.values.slice(); values.push('');
  ED.updateNode(key, id, { ops, values, op: ops[0] ?? '+' });
  return true;
};
ED.removeOperand = async function (key, id) {
  const n = ED.getNode(key, id);
  if (!n || n.data.type !== 'binop') return false;
  const count = Object.keys(n.inputs || {}).length;
  if (count <= ED.BINOP_MIN) return false;
  const st = ED.binopState(n);
  const last = 'input_' + count;
  if (n.inputs[last]?.connections?.length) {
    const ok = await ED.confirmDialog(`移除输入 ${ED.binopName(count - 1)}`,
      `输入 ${ED.binopName(count - 1)} 已有连线，移除该输入会同时删除这条连线；` +
      `其他输入的连线不受影响，最少保留 2 个输入。确定移除？`);
    if (!ok) return false;
  }
  // Drawflow 的 removeNodeInput 只删除该输入及其连线；删除末端口不触发其他端口重编号。
  ED.canvases[key].editor.removeNodeInput(id, last);
  const ops = st.ops.slice(0, count - 2);
  const values = st.values.slice(0, count - 1);
  ED.updateNode(key, id, { ops, values, op: ops[0] ?? '+' });
  return true;
};

ED.exportData = function (key) {
  const ex = ED.canvases[key].editor.export();
  const mod = ex.drawflow[Object.keys(ex.drawflow)[0]];
  return mod ? mod.data : {};
};

ED.importData = function (key, data) {
  ED.hooks.onReset?.(key);
  const restored = JSON.parse(JSON.stringify(data || {}));
  for (const node of Object.values(restored)) {
    node.html = ED.nodeHTML();
    node.typenode = false;
    node.class = `node-${node.data.type}`;
  }
  ED.canvases[key].editor.import({ drawflow: { Home: { data: restored } } });
  // 恢复后重绘每张卡片的参数摘要
  const df = ED.exportData(key);
  for (const id of Object.keys(df)) ED.refreshSummary(key, id);
};

ED.clearCanvas = function (key) {
  ED.hooks.onReset?.(key);
  ED.canvases[key].editor.clear();
};

/* 供外部批量搭图用（示例/自动化测试） */
ED.api = {
  add: (key, type, x, y, fname) => ED.addNodeTo(key, type, x, y, fname),
  connect: ED.connect,
  setParam: (key, id, patch) => ED.updateNode(key, id, patch),
  exportData: ED.exportData,
};
