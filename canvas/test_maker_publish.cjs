/* NodeSpec 发布-保存集成测试（node test_maker_publish.cjs）。
 * 在 vm 中按 index.html 顺序加载真实脚本（editor/app/architecture/nodespec/nodespec-app/projects），
 * 用真实 App.makerPublish 验证审批要求「未被画布使用的定义也保存」：
 *   - 发布成功（新定义 / 版本冲突自动新版本）→ App.scheduleSync 触发 editRevision + keepDraft
 *     浏览器草稿 + 保存队列 syncNow → /api/save 携带 nodeSpecs 快照；
 *   - 刷新恢复：真实 App.loadProject 从保存的图恢复注册表与 NODE_DEFS；
 *   - 幂等发布（内容未变化）不增版本、不改状态、不进保存队列；
 *   - 发布失败（非法规范 / literal 服务不可用）不改注册表、不排队、不写草稿；
 *   - 保存冲突回报：/api/save 失败 → keepDraft 保留 + 状态行报错，存档版本不变；
 *   - 跨项目 async：等待 literal 校验期间切换项目 → 发布取消，不把定义写进另一个项目；
 *   - 跨项目 async 试运行：/api/nodespec/try 等待期间切换项目 → 旧结果丢弃不污染；
 *   - 草稿归属绑定：切项目清理旧草稿，同项目重载保留。
 * 全程内存 stub，不访问浏览器、不接触任何项目数据。 */
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict'), path = require('path');

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
  before() {}
  /* 试运行代码框按 id / 标签查找子节点（makerTryRun 用 $('#maker-try-code') 与 querySelector('pre')）。 */
  querySelector(sel) {
    if (!sel) return null;
    const walk = (el, test) => {
      for (const child of (el.children || [])) {
        if (test(child)) return child;
        const found = walk(child, test);
        if (found) return found;
      }
      return null;
    };
    if (sel.startsWith('#')) { const want = sel.slice(1); return walk(this, el => el.id === want); }
    const tag = String(sel).toUpperCase();
    return walk(this, el => el.tagName === tag);
  }
  querySelectorAll() { return []; }
  get text() { return this.textContent + this.children.map(c => c.textContent).join(''); }
}

function fixture() {
  const elements = new Map();
  const byId = id => { if (!elements.has(id)) elements.set(id, new UIElement('div')); return elements.get(id); };
  const storage = new Map();
  const calls = [];        // 全部 fetch：{path, body}
  const saves = [];        // /api/save 请求体
  const pending = new Map();   // 捕获的 setTimeout 回调（保存队列去抖）
  let timerSeq = 0, literalGate = null, tryGate = null;
  const flags = { literal: 'ok', save: 'ok', tryRun: 'ok' };   // 'ok' | 'defer' | 'fail'（literal/tryRun），'ok' | 'fail'（save）

  const context = vm.createContext({
    console,
    window: { addEventListener() {} },
    setTimeout: (fn) => { pending.set(++timerSeq, fn); return timerSeq; },
    clearTimeout: id => { pending.delete(id); },
    localStorage: {
      setItem: (k, v) => storage.set(k, v),
      getItem: k => storage.has(k) ? storage.get(k) : null,
      removeItem: k => storage.delete(k),
    },
    fetch: async (url, opts) => {
      const body = opts && opts.body ? JSON.parse(opts.body) : null;
      calls.push({ path: url, body });
      if (url === '/api/literal') {
        if (flags.literal === 'defer') await new Promise(resolve => { literalGate = resolve; });
        if (flags.literal === 'fail') throw new Error('literal 服务不可用');
        return { ok: true, json: async () => ({ expression: body.source, type: 'list' }) };
      }
      if (url === '/api/nodespec/try') {
        if (flags.tryRun === 'defer') await new Promise(resolve => { tryGate = resolve; });
        if (flags.tryRun === 'fail') return { ok: true, json: async () => ({ ok: false, error: '候选失败', returncode: 1 }) };
        return { ok: true, json: async () => ({ ok: true, preview: { kind: 'value', type: 'float', repr: '2.0' }, code: 'out = math.sqrt(in1)' }) };
      }
      if (url === '/api/save') {
        saves.push(body);
        if (flags.save === 'fail') return { ok: false, json: async () => ({ error: '保存冲突' }) };
        return { ok: true, json: async () => ({ ok: true, project: { id: body.project_id, version: body.version + 1, name: 'P' }, revision: 'r' + saves.length }) };
      }
      return { ok: true, json: async () => ({}) };
    },
  });
  for (const file of ['editor.js', 'app.js', 'architecture.js', 'nodespec.js', 'nodespec-app.js', 'projects.js']) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, 'static', file), 'utf8'), context);
  }
  context.document = {
    querySelector: sel => (sel.startsWith('#') ? byId(sel.slice(1)) : null),
    createElement: tag => new UIElement(tag),
    createTextNode: text => ({ textContent: String(text) }),
  };
  const get = expr => vm.runInContext(expr, context);
  const App = get('App');
  get(`ED.exportData = key => fixtures[key] || {}; ED.hooks.projectId = () => null;`);
  /* 画布数据面与渲染 UI 与本测试无关，stub 掉；保存管线（scheduleSync/keepDraft/syncNow/request）保持真实。 */
  get(`ED.clearClipboard = () => {}; ED.clearCanvas = () => {}; ED.importData = () => {};`);
  get(`App.refreshSignatures = () => {}; App.validateLiterals = async () => {};`
    + `App.generatePython = () => 'print(1)'; App.renderCodePreview = () => {};`
    + `App.renderTabs = () => {}; App.renderFnPalette = () => {}; App.switchTab = () => {};`
    + `App.closeNodeDrawers = () => {}; App.hideInspector = () => {};`);
  context.fixtures = {};
  const NS = get('window.NS');
  const status = () => byId('maker-status').textContent;
  const projectStatus = () => byId('sync-status').textContent;
  const flush = async () => {   // 触发去抖的保存队列并等真实 saveQueue 排空
    for (let i = 0; i < 5 && pending.size; i++) {
      const batch = [...pending.values()]; pending.clear();
      batch.forEach(fn => fn());
      await App.saveQueue;
    }
  };
  const asProject = (id, version, graph) => ({ id, version, name: 'P', code: 'print(1)', graph });
  return { context, get, NS, storage, calls, saves, pending, flags, byId, status, projectStatus, flush, asProject,
    releaseLiteral: () => { if (literalGate) { const r = literalGate; literalGate = null; r(); } },
    releaseTryRun: () => { if (tryGate) { const r = tryGate; tryGate = null; r(); } },
    open: id => { get(`App.project={id:'${id}',version:1,name:'P'}`); get('App.openMaker()'); },
    literalDraft() {
      const draft = NS.deepCopy(NS.TEMPLATES.sqrt);
      draft.id = 'scaled_sqrt';
      draft.params = [{ key: 'scale', label: 'scale', type: 'literal', def: '[1, 2]', help: '' }];
      draft.call.args.push({ source: 'param', key: 'scale', keyword: null, fallbackParam: null });
      return draft;
    } };
}

(async () => {
  /* ---- 1. 新定义发布 → 保存队列（editRevision + keepDraft + /api/save 携带 nodeSpecs）→ 刷新 load 恢复 ---- */
  {
    const f = fixture();
    f.open('p1');
    f.get('App.makerBindDraft(window.NS.deepCopy(window.NS.TEMPLATES.sqrt))');
    assert.equal(f.get('App.maker.projectId'), 'p1', '草稿必须绑定当前项目');
    await f.get('App.makerPublish()');

    assert.equal(f.get(`App.customRegistry.has('custom.sqrt@1')`), true, '注册表必须已有定义');
    assert.equal(f.get(`!!NODE_DEFS['custom.sqrt@1']`), true, '发布必须安装 NODE_DEFS');
    assert.match(f.status(), /已发布 custom\.sqrt@1/);
    assert.match(f.status(), /保存队列/);
    assert.equal(f.get('App.editRevision'), 1, '发布必须递增 editRevision');
    const draftBackup = JSON.parse(f.storage.get('canvas-draft:p1'));
    assert.equal(draftBackup.graph.nodeSpecs.length, 1, 'keepDraft 草稿必须携带 nodeSpecs');
    assert.equal(draftBackup.graph.nodeSpecs[0].id, 'sqrt');

    await f.flush();
    assert.equal(f.saves.length, 1, '保存队列必须发出一次 /api/save');
    const saveBody = f.saves[0];
    assert.equal(saveBody.project_id, 'p1');
    assert.equal(saveBody.graph.nodeSpecs.length, 1);
    assert.equal(saveBody.graph.nodeSpecs[0].id, 'sqrt');
    assert.equal(saveBody.graph.nodeSpecs[0].version, 1);
    assert.equal(saveBody.graph.version, 3);
    assert.equal(f.storage.has('canvas-draft:p1'), false, '保存确认后草稿备份应清除');

    /* 刷新恢复：用保存的图走真实 loadProject（规范注册走真实 prepareProjectSpecs） */
    f.context.restored = f.asProject('p1', 2, saveBody.graph);
    f.get('App.loadProject(restored)');
    assert.equal(f.get('App.specProblem'), '');
    assert.equal(f.get(`App.customRegistry.has('custom.sqrt@1')`), true, '刷新后注册表必须恢复定义');
    assert.equal(f.get(`!!NODE_DEFS['custom.sqrt@1']`), true, '刷新后 NODE_DEFS 必须恢复');
    assert.equal(f.get('App.customRegistry.size()'), 1);
    assert.equal(f.get('App.editRevision'), 0, '载入项目不触发保存');
    console.log('PASS: 新定义发布触发保存队列（快照含 nodeSpecs），刷新 loadProject 恢复定义');
  }

  /* ---- 2. 幂等发布：内容未变化 → 不增版本、不递增 editRevision、不排队 ---- */
  {
    const f = fixture();
    f.open('p1');
    f.get('App.makerBindDraft(window.NS.deepCopy(window.NS.TEMPLATES.sqrt))');
    await f.get('App.makerPublish()');
    await f.flush();
    const before = { rev: f.get('App.editRevision'), saves: f.saves.length, pending: f.pending.size };
    await f.get('App.makerPublish()');   // 草稿已是已发布内容
    assert.match(f.status(), /内容未变化，无需重复发布/);
    assert.equal(f.get('App.customRegistry.maxVersion("sqrt")'), 1, '幂等发布不得产生新版本');
    assert.equal(f.get('App.editRevision'), before.rev, '幂等发布不得改项目状态');
    assert.equal(f.saves.length, before.saves, '幂等发布不得触发保存');
    assert.equal(f.pending.size, before.pending, '幂等发布不得排队');
    console.log('PASS: 幂等发布不增版本、不改状态、不进保存队列');
  }

  /* ---- 3. 版本冲突：同 id+version 不同内容 → 自动新版本并保存，旧版本保持不变 ---- */
  {
    const f = fixture();
    f.open('p1');
    f.get('App.makerBindDraft(window.NS.deepCopy(window.NS.TEMPLATES.sqrt))');
    await f.get('App.makerPublish()');
    await f.flush();
    f.get('App.maker.draft.label = "改名后的 sqrt"');
    await f.get('App.makerPublish()');
    assert.match(f.status(), /custom\.sqrt@2/);
    assert.match(f.status(), /新版本/);
    assert.equal(f.get(`!!NODE_DEFS['custom.sqrt@2']`), true);
    assert.notEqual(f.get(`NODE_DEFS['custom.sqrt@1'].spec.label`), '改名后的 sqrt', '旧版本定义保持不变');
    assert.equal(f.get('App.customRegistry.maxVersion("sqrt")'), 2);
    assert.equal(f.get('App.editRevision'), 2, '新版本发布必须再次递增 editRevision');
    await f.flush();
    assert.equal(f.saves.length, 2);
    assert.deepEqual(f.saves[1].graph.nodeSpecs.map(s => `${s.id}@${s.version}`).sort(), ['sqrt@1', 'sqrt@2'], '保存快照必须同时携带新旧版本');
    console.log('PASS: 版本冲突自动发布新版本并进入保存队列，旧版本保持不变');
  }

  /* ---- 4. 发布失败：不改注册表、不排队、不写草稿 ---- */
  {
    const f = fixture();
    f.open('p1');
    f.context.bad = f.NS.deepCopy(f.NS.TEMPLATES.sqrt);
    f.context.bad.call.target = 'os.system; import os';   // 非法目标语法（自由目标只做语法校验）
    f.get('App.makerBindDraft(bad)');
    await f.get('App.makerPublish()');
    assert.match(f.status(), /发布失败/);
    assert.match(f.status(), /存档未改动/);
    assert.equal(f.get('App.customRegistry.size()'), 0, '失败不得写入注册表');
    assert.equal(f.get('App.editRevision'), 0, '失败不得递增 editRevision');
    assert.equal(f.saves.length, 0, '失败不得触发保存');
    assert.equal(f.pending.size, 0, '失败不得排队');
    assert.equal(f.storage.has('canvas-draft:p1'), false, '失败不得写浏览器草稿');
  }
  {
    const f = fixture();   // literal 校验服务不可用：warm 阶段失败同样不动存档
    f.open('p1');
    f.context.ld = f.literalDraft();
    f.get('App.makerBindDraft(ld)');
    f.flags.literal = 'fail';
    await f.get('App.makerPublish()');
    assert.match(f.status(), /发布失败/);
    assert.equal(f.get('App.customRegistry.size()'), 0);
    assert.equal(f.saves.length, 0);
    assert.equal(f.storage.has('canvas-draft:p1'), false);
    console.log('PASS: 非法规范 / literal 服务失败 → 发布失败且注册表、队列、草稿、存档全部未动');
  }

  /* ---- 5. 保存冲突回报：/api/save 失败 → keepDraft 保留草稿、状态行报错、注册表仍在 ---- */
  {
    const f = fixture();
    f.open('p1');
    f.get('App.makerBindDraft(window.NS.deepCopy(window.NS.TEMPLATES.sqrt))');
    f.flags.save = 'fail';
    await f.get('App.makerPublish()');
    await f.flush();
    assert.match(f.projectStatus(), /保存失败/);
    assert.equal(f.storage.has('canvas-draft:p1'), true, '保存失败必须保留浏览器草稿备份');
    assert.equal(f.get(`App.customRegistry.has('custom.sqrt@1')`), true, '内存定义保留，可重试保存');
    assert.equal(f.get('App.project.version'), 1, '保存失败不得推进存档版本');
    console.log('PASS: 保存冲突回报且浏览器草稿保留、存档版本未推进');
  }

  /* ---- 6. 跨项目 async：等待 literal 校验期间切换项目 → 发布取消，定义不进另一个项目 ---- */
  {
    const f = fixture();
    f.open('p1');
    f.context.ld = f.literalDraft();
    f.get('App.makerBindDraft(ld)');
    f.flags.literal = 'defer';
    const publishing = f.get('App.makerPublish()');   // 挂起在 /api/literal
    await Promise.resolve();
    f.context.other = f.asProject('p2', 1, { version: 3, seed: 1, main: {}, functions: [] });
    f.get('App.loadProject(other)');   // 切换项目：注册表重建、旧草稿作废
    assert.equal(f.get('App.customRegistry.size()'), 0);
    assert.equal(f.get('App.maker.draft'), null, '切项目必须作废前项目草稿');
    assert.equal(f.get('App.maker.projectId'), 'p2');
    f.releaseLiteral();   // 放行挂起中的发布
    await publishing;
    assert.match(f.status(), /项目已切换/);
    assert.match(f.status(), /取消/);
    assert.equal(f.get('App.customRegistry.size()'), 0, '取消后新项目注册表必须保持为空');
    assert.equal(Object.keys(f.get('NODE_DEFS')).filter(t => f.NS.isCustomType(t)).length, 0, '不得给新项目安装定义');
    assert.equal(f.saves.length, 0, '取消发布不得触发保存');
    assert.equal(f.pending.size, 0);
    assert.equal(f.get('App.editRevision'), 0);
    console.log('PASS: await 期间项目切换 → 发布取消，定义与保存都不会进入另一个项目');
  }

  /* ---- 7. 草稿归属：切项目清理、同项目重载保留、旧签名调用兼容 ---- */
  {
    const f = fixture();
    f.open('p1');
    f.get('App.makerLoadTemplate("sin")');
    assert.equal(f.get('App.maker.draft.id'), 'sin');
    assert.equal(f.get('App.maker.projectId'), 'p1');
    f.context.g2 = { version: 3, seed: 1, main: {}, functions: [] };
    f.get('App.prepareProjectSpecs(g2, "p2")');
    assert.equal(f.get('App.maker.draft'), null, '载入不同项目必须清理草稿');
    assert.equal(Object.keys(f.get('App.maker.samples')).length, 0, '试运行样本必须一并清理');
    assert.equal(f.get('App.maker.projectId'), 'p2');
    f.get('App.project={id:"p2",version:1,name:"P2"}');   // loadProject 在 prepareProjectSpecs 之后才赋值 App.project
    f.get('App.openMaker()');   // 新项目里重建草稿
    assert.equal(f.get('App.maker.draft.id'), 'my_op');
    assert.equal(f.get('App.maker.projectId'), 'p2');
    f.get('App.prepareProjectSpecs(g2, "p2")');   // 同项目重载：保留进行中的草稿
    assert.equal(f.get('App.maker.draft.id'), 'my_op');
    f.get('App.prepareProjectSpecs(g2)');   // 旧调用（无 projectId）：行为不变
    assert.equal(f.get('App.maker.draft.id'), 'my_op');
    console.log('PASS: 草稿归属绑定（切项目清理、同项目保留、旧调用兼容）');
  }

  /* ---- 8. 跨项目 async 试运行：/api/nodespec/try 等待期间切换项目 → 旧结果丢弃 ---- */
  {
    const f = fixture();
    f.open('p1');
    // 预置试运行代码框结构（真实 DOM 中由首次试运行创建；离线 stub 需手动建好）
    f.get('const cb0 = document.querySelector("#maker-try-code");'
      + 'if (!cb0.querySelector("#maker-try-code-view")) { const p = document.createElement("pre"); p.id = "maker-try-code-view"; cb0.appendChild(p); }');
    f.get('App.makerBindDraft(window.NS.deepCopy(window.NS.TEMPLATES.sqrt))');
    f.get('App.maker.samples[0] = { mode: "literal", source: "4" }');
    f.flags.tryRun = 'defer';
    const running = f.get('App.makerTryRun()');   // 挂起在 /api/nodespec/try
    for (let i = 0; i < 50 && !f.calls.some(c => c.path === '/api/nodespec/try'); i++) await Promise.resolve();
    assert(f.calls.some(c => c.path === '/api/nodespec/try'), '试运行请求必须已发出并挂起');
    f.context.other = f.asProject('p2', 1, { version: 3, seed: 1, main: {}, functions: [] });
    f.get('App.loadProject(other)');   // 切换项目：草稿/样本作废
    assert.equal(f.get('App.maker.draft'), null, '切项目必须作废旧草稿');
    f.releaseTryRun();
    await running;
    assert.match(f.status(), /项目已切换/, '试运行结果必须标记为已丢弃');
    assert.match(f.status(), /丢弃/);
    assert.equal(f.get('App.maker.busy'), false, '试运行结束后必须释放 busy');
    /* 新项目里的正常试运行不受旧结果污染 */
    f.flags.tryRun = 'ok';
    f.get('App.openMaker()');
    f.get('App.maker.draft.label = "试";App.maker.samples[0] = { mode: "literal", source: "9" }');
    await f.get('App.makerTryRun()');
    assert.match(f.status(), /试运行成功/);
    assert.match(f.status(), /float · 2\.0/, '新项目的试运行结果正常显示');
    assert.equal(f.byId('maker-try-code').querySelector('#maker-try-code-view').textContent, 'out = math.sqrt(in1)');
    console.log('PASS: await 期间项目切换 → 试运行结果丢弃不污染，新项目试运行正常');
  }

  console.log('ALL PASS: makerPublish 发布-保存合约（真实 App.makerPublish + 真实保存管线）');
})().catch(e => { console.error(e); process.exit(1); });
