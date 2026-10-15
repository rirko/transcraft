/*
 * nodespec-app.js —— NodeSpec 浏览器集成层
 * 职责：
 *   1. 注册表实例 + 规范 → NODE_DEFS 转换安装 + 组件栏「自定义」分组；
 *   2. 编译接入：统一可编译谓词（App.isStmtType）、buildScope 后置校验、
 *      stmtOf 统一分支（自定义节点从规范生成，无逐类型特判、无专用 sqrt 编译）、
 *      收集实际使用的自定义节点声明式 imports（去重后并入生成代码头部）；
 *   3. 节点制作器：页面内横向宽面板（元数据/调用模板/自由目标 + 声明式 imports/
 *      输入与参数增删/绑定/接收者/输出类型提示/实时卡片与 Python 预览/验证/
 *      独立试运行/发布/JSON 高级选项）。
 * 全部动态文本经 textContent/value 写入，禁止 HTML 注入；调用目标只做 Python
 * 语法校验（无隐藏白名单），并在试运行区明示信任边界与副作用提醒。
 */
(() => {
  'use strict';
  const NS = window.NS;
  if (!NS) throw Error('nodespec.js 未加载');

  const literalHook = { literal: source => App.literalCache.get(String(source ?? '')) || null };
  App.customRegistry = NS.createRegistry();
  App.specProblem = '';

  /* ---- 规范 → 现有 NODE_DEFS 结构（复用参数表单/端口/摘要机制） ---- */
  function installDef(spec) {
    NODE_DEFS[NS.encode(spec.id, spec.version)] = NS.specToDef(spec, literalHook);
  }
  App.installNodeSpecs = function () {
    for (const spec of App.customRegistry.list()) installDef(spec);
  };
  App.nodeSpecsSnapshot = () => (App.customRegistry ? App.customRegistry.serialize() : []);

  /* 加载项目：先验证并注册（切换项目即清空重建 custom registry），再导入图。
   * 旧 v2/v3 快照没有 nodeSpecs 字段 → 视为空数组，行为不变；
   * 图里引用了缺失规范 → 保留原图、给出明确报错，编译必然失败从而禁止运行。 */
  App.prepareProjectSpecs = function (graph, projectId) {
    App.customRegistry = NS.createRegistry();
    for (const key of Object.keys(NODE_DEFS)) if (NS.isCustomType(key)) delete NODE_DEFS[key];   // 清理上一项目的安装
    App.specProblem = '';
    /* 制作器草稿归属：载入不同项目时旧草稿作废（同项目重载保留进行中的草稿）；
     * 不带 projectId 的旧调用保持原行为。 */
    if (projectId !== undefined && App.maker) {
      if (App.maker.projectId !== projectId) { App.maker.draft = null; App.maker.samples = {}; }
      App.maker.projectId = projectId;
    }
    const problems = [];
    for (const raw of (Array.isArray(graph.nodeSpecs) ? graph.nodeSpecs : [])) {
      try { App.customRegistry.publish(raw, literalHook); }
      catch (e) { problems.push(`节点规范无效（${e.message}）`); }
    }
    App.installNodeSpecs();
    const missing = NS.scanMissing(graph, App.customRegistry);
    if (missing.length) problems.push(`缺少节点定义：${missing.join('、')}（原图已保留，禁止运行；可打开制作器重新发布同名同版本定义）`);
    if (problems.length) App.specProblem = problems.join('；');
    return App.specProblem;
  };
  /* 实时重算：当前画布上是否有解析不了的 custom 节点（删除坏节点后自动解除）。 */
  App.specProblemNow = function () {
    const missing = new Set();
    for (const key of Object.keys(ED.canvases)) {
      for (const n of Object.values(ED.exportData(key))) {
        if (NS.isCustomType(n.data.type) && !App.customRegistry.has(n.data.type)) missing.add(n.data.type);
      }
    }
    return missing.size ? `缺少节点定义：${[...missing].join('、')}（禁止运行）` : '';
  };

  /* ---- 统一可编译谓词：内置名单 + 自定义编码类型，避免 STMT_TYPES 漏新类型 ---- */
  const builtinIsStmt = App.isStmtType;
  App.isStmtType = t => (builtinIsStmt ? builtinIsStmt(t) : STMT_TYPES.includes(t)) || NS.isCustomType(t);

  /* ---- buildScope 接入：先给出缺失定义/未知类型的明确报错（保留原图、禁止运行），
   *      再做参数值渲染校验并生成 customExpr（一次解析，stmt 复用） ---- */
  const scopeWithSpecs = buildScope;
  buildScope = function (df, opts) {
    for (const [id, n] of Object.entries(df || {})) {
      if (NODE_DEFS[n.data?.type]) continue;
      if (NS.isCustomType(n.data?.type)) {
        throw Error(`缺少节点规范 ${n.data.type}（节点 ${id}）：本项目的节点规范中没有该定义，原图已保留，禁止运行`);
      }
      throw Error('未知节点类型: ' + id);
    }
    const s = scopeWithSpecs(df, opts);
    for (const rec of Object.values(s.recs)) {
      if (!NS.isCustomType(rec.type)) continue;
      const spec = App.customRegistry.get(rec.type);
      if (!spec) {
        rec.valid = false;
        rec.skipReason = `缺少节点规范 ${rec.type}`;
        s.errors.push(`节点「${dispName(rec)}」缺少节点规范 ${rec.type}，已保留原图但禁止运行`);
        continue;
      }
      try {
        rec.customExpr = NS.renderCall(spec, rec.argStrs, rec.n.data, literalHook.literal);
      } catch (e) {
        rec.valid = false;
        rec.skipReason = e.message;
        s.errors.push(`节点「${dispName(rec)}」${e.message}`);
      }
    }
    return s;
  };

  /* ---- stmtOf 统一调用：自定义节点走规范渲染分支，其余类型保持原逻辑 ---- */
  const stmtWithSpecs = stmtOf;
  stmtOf = function (rec) {
    if (NS.isCustomType(rec.type)) {
      const spec = App.customRegistry.get(rec.type);
      if (!spec) throw Error(`缺少节点规范 ${rec.type}`);
      return `${rec.outVar} = ${rec.customExpr || NS.renderCall(spec, rec.argStrs, rec.n.data, literalHook.literal)}`;
    }
    return stmtWithSpecs(rec);
  };

  /* ---- 生成代码头部的声明式 imports：只收集当前图（主画布 + 全部模型画布）
   * 实际用到的自定义节点所声明的 imports，按 module+alias 去重；与默认
   * torch / torch.nn as nn 重复的声明在规范校验时已被丢弃。 ---- */
  App.collectCustomImports = function () {
    const scopes = ['main', ...(App.state?.fnOrder || []).map(name => 'fn:' + name)];
    const seen = new Map();
    for (const key of scopes) {
      for (const n of Object.values(ED.exportData(key) || {})) {
        const spec = NS.isCustomType(n.data?.type) ? App.customRegistry.get(n.data.type) : null;
        if (!spec) continue;
        for (const entry of (spec.imports || [])) {
          const keyText = `${entry.module}|${entry.alias || ''}`;
          if (!seen.has(keyText)) seen.set(keyText, { module: entry.module, alias: entry.alias || null });
        }
      }
    }
    return [...seen.values()];
  };

  /* ---- Literal 预热：custom 节点的 literal 参数与内置节点共用后端校验缓存 ---- */
  const oldValidateLiterals = App.validateLiterals;
  App.validateLiterals = async function () {
    await oldValidateLiterals.call(App);
    for (const key of Object.keys(ED.canvases)) {
      const sources = [];
      for (const n of Object.values(ED.exportData(key))) {
        const spec = NS.isCustomType(n.data.type) ? App.customRegistry.get(n.data.type) : null;
        if (!spec) continue;
        for (const p of spec.params) {
          if (p.type !== 'literal') continue;
          const source = String(n.data[p.key] ?? p.def ?? '').trim();
          if (source) sources.push(source);
        }
      }
      for (const source of sources) {
        if (App.literalCache.has(source)) continue;
        const response = await fetch('/api/literal', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source }),
        });
        const result = await response.json();
        if (!response.ok && !result.error) throw Error('基础值校验服务不可用');
        App.literalCache.set(source, result);
      }
    }
  };

  /* ---- 组件栏：按分类追加「自定义」分组 ---- */
  const oldBuildPalette = App.buildPalette;
  App.buildPalette = function () {
    oldBuildPalette.call(App);
    App.renderCustomPalette();
  };
  App.renderCustomPalette = function () {
    const pal = $('#palette-basic');
    if (!pal || !pal.querySelectorAll) return;
    pal.querySelectorAll('[data-custom-group]').forEach(node => node.remove());
    const byCategory = new Map();
    for (const spec of App.customRegistry.list()) {
      if (!byCategory.has(spec.category)) byCategory.set(spec.category, []);
      byCategory.get(spec.category).push(spec);
    }
    for (const [category, specs] of byCategory) {
      const group = document.createElement('details');
      group.className = 'palette-group'; group.open = true; group.dataset.customGroup = '1';
      const heading = document.createElement('summary');
      heading.textContent = `${category}（自定义节点）`;
      group.appendChild(heading);
      for (const spec of specs) group.appendChild(App.paletteItem(NS.encode(spec.id, spec.version)));
      pal.appendChild(group);
    }
  };

  /* ================= 节点制作器 ================= */
  /* projectId：草稿归属的项目身份。草稿只属于绑定时打开的项目；
   * prepareProjectSpecs 载入不同项目时作废旧草稿，防止把前项目的草稿无意发布进另一个项目。 */
  App.maker = { open: false, draft: null, built: false, busy: false, samples: {}, projectId: null, runSeq: 0 };

  /* 绑定草稿归属：所有替换草稿的入口都必须经过这里。 */
  App.makerBindDraft = function (draft) {
    App.maker.draft = draft;
    App.maker.projectId = App.project?.id ?? null;
    return draft;
  };

  const makerEl = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  /* 一行字段：左侧中文名 + 可选 ? 帮助，右侧控件；全部 textContent，无 HTML。 */
  const makerField = (labelText, control, helpText) => {
    const row = makerEl('label', 'maker-row');
    const span = makerEl('span', null, labelText);
    if (helpText) {
      const help = makerEl('details', 'param-help');
      const summary = makerEl('summary', null, '?');
      summary.setAttribute('aria-label', labelText + ' 说明');
      const body = makerEl('div', null, helpText);
      help.appendChild(summary); help.appendChild(body);
      span.appendChild(help);
    }
    row.appendChild(span);
    row.appendChild(control);
    return row;
  };
  const makerInput = (value, placeholder, onInput) => {
    const input = document.createElement('input');
    input.type = 'text'; input.value = value ?? ''; input.placeholder = placeholder || '';
    input.addEventListener('input', () => onInput(input.value));
    return input;
  };
  const makerSelect = (options, value, onChange) => {
    const select = document.createElement('select');
    for (const option of options) {
      const opt = document.createElement('option');
      opt.value = option.value; opt.textContent = option.label;
      select.appendChild(opt);
    }
    select.value = value;
    select.addEventListener('change', () => onChange(select.value));
    return select;
  };
  const makerButton = (text, onClick, className) => {
    const button = makerEl('button', className, text);
    button.type = 'button';
    button.addEventListener('click', onClick);
    return button;
  };

  App.makerBlankDraft = function () {
    return { schema: 1, id: 'my_op', version: 1, label: '', description: '', category: '自定义', imports: [],
      output: { name: 'out', type: 'Tensor' }, inputs: [{ name: 'x', type: 'Tensor', optional: false, fallbackParam: null, help: '' }],
      params: [], call: { kind: 'function', target: NS.FUNCTION_TARGETS[0], args: [{ source: 'input', index: 1, keyword: null, fallbackParam: null }] } };
  };
  App.makerLoadTemplate = function (name) {
    const template = NS.TEMPLATES[name];
    if (!template) return;
    App.makerBindDraft(NS.deepCopy(template));
    App.maker.samples = {};
    App.makerRefresh();
    App.makerStatus(`已载入模板「${template.label}」，可修改后发布为自己的版本。`, 'ok');
  };

  App.makerStatus = function (text, kind) {
    const box = $('#maker-status');
    if (!box) return;
    box.textContent = text || '';
    box.className = 'maker-status ' + (kind || '');
  };

  App.normalizeDraft = function () {
    return NS.normalize(App.maker.draft, literalHook);
  };

  /* ---- 面板构建（一次性），列表区域由 refresh 重建 ---- */
  App.toggleMaker = function () {
    App.maker.open ? App.closeMaker() : App.openMaker();
  };
  App.openMaker = function () {
    if (!App.maker.draft) App.makerBindDraft(App.makerBlankDraft());
    if (!App.maker.built) { App.buildMakerPanel(); App.maker.built = true; }
    App.maker.open = true;
    $('#nodespec-maker').hidden = false;
    App.makerRefresh();
  };
  App.closeMaker = function () {
    App.maker.open = false;
    const panel = $('#nodespec-maker');
    if (panel) panel.hidden = true;
  };

  App.buildMakerPanel = function () {
    const panel = makerEl('section', 'maker-panel');
    panel.id = 'nodespec-maker';
    panel.hidden = true;
    panel.setAttribute('aria-label', '节点制作器');

    const header = makerEl('header');
    const title = makerEl('h3', null, '节点制作器');
    title.id = 'maker-title';
    header.appendChild(title);
    header.appendChild(makerEl('span', 'dim', '配置自定义节点规范：元数据 → 函数 / 方法（自由目标 + 声明式 imports）→ 输入与参数 → 绑定 → 验证 / 试运行 / 发布'));
    const spacer = makerEl('span', 'maker-spacer');
    header.appendChild(spacer);
    header.appendChild(makerButton('关闭', App.closeMaker));
    panel.appendChild(header);

    const columns = makerEl('div', 'maker-columns');

    /* ---- 第 1 列：模板 / 已发布 / 元数据 / JSON ---- */
    const colMeta = makerEl('div', 'maker-col');
    colMeta.appendChild(makerEl('h4', null, '模板（可载入修改）'));
    const templateRow = makerEl('div', 'maker-row');
    for (const name of Object.keys(NS.TEMPLATES)) {
      templateRow.appendChild(makerButton(name, () => App.makerLoadTemplate(name), 'mini'));
    }
    colMeta.appendChild(templateRow);

    colMeta.appendChild(makerEl('h4', null, '已发布定义'));
    const publishedBox = makerEl('div');
    publishedBox.id = 'maker-published';
    colMeta.appendChild(publishedBox);

    colMeta.appendChild(makerEl('h4', null, '元数据'));
    const metaBox = makerEl('div');
    metaBox.id = 'maker-meta';
    colMeta.appendChild(metaBox);

    const jsonDetails = makerEl('details', 'maker-json');
    const jsonSummary = makerEl('summary', null, 'JSON 导入 / 导出（高级）');
    jsonDetails.appendChild(jsonSummary);
    const jsonArea = document.createElement('textarea');
    jsonArea.id = 'maker-json-area';
    jsonArea.rows = 6;
    jsonArea.setAttribute('aria-label', '节点规范 JSON');
    jsonDetails.appendChild(jsonArea);
    const jsonRow = makerEl('div', 'maker-row');
    jsonRow.appendChild(makerButton('导出当前草稿', () => {
      jsonArea.value = JSON.stringify(App.maker.draft, null, 2);
      App.makerStatus('已导出草稿 JSON。', 'ok');
    }, 'mini'));
    jsonRow.appendChild(makerButton('导入 JSON', () => {
      try {
        const parsed = JSON.parse(jsonArea.value);
        NS.normalize(parsed, literalHook);   // 先验证再替换
        App.makerBindDraft(parsed);
        App.makerRefresh();
        App.makerStatus('JSON 已导入并通过校验。', 'ok');
      } catch (e) {
        App.makerStatus('导入失败：' + e.message, 'err');
      }
    }, 'mini'));
    jsonDetails.appendChild(jsonRow);
    colMeta.appendChild(jsonDetails);
    columns.appendChild(colMeta);

    /* ---- 第 2 列：调用模板 + imports + 输入 ---- */
    const colCall = makerEl('div', 'maker-col');
    colCall.appendChild(makerEl('h4', null, '函数 / 方法模板'));
    const callBox = makerEl('div');
    callBox.id = 'maker-call';
    colCall.appendChild(callBox);
    colCall.appendChild(makerEl('h4', null, '模块导入 imports（声明式，供自由目标使用）'));
    const importsBox = makerEl('div');
    importsBox.id = 'maker-imports';
    colCall.appendChild(importsBox);
    colCall.appendChild(makerButton('＋ 添加导入', () => {
      const draft = App.maker.draft;
      if (!Array.isArray(draft.imports)) draft.imports = [];
      if (draft.imports.length >= NS.MAX_IMPORTS) { App.makerStatus(`导入最多 ${NS.MAX_IMPORTS} 条。`, 'err'); return; }
      draft.imports.push({ module: '', alias: null });
      App.makerRefresh();
    }, 'mini'));
    colCall.appendChild(makerEl('h4', null, '输入端口（固定，单个输出）'));
    const inputsBox = makerEl('div');
    inputsBox.id = 'maker-inputs';
    colCall.appendChild(inputsBox);
    const addInput = makerButton('＋ 添加输入', () => {
      const draft = App.maker.draft;
      if (draft.inputs.length >= NS.MAX_INPUTS) { App.makerStatus(`输入最多 ${NS.MAX_INPUTS} 个。`, 'err'); return; }
      draft.inputs.push({ name: 'in' + (draft.inputs.length + 1), type: 'Tensor', optional: false, fallbackParam: null, help: '' });
      App.makerRefresh();
    }, 'mini');
    colCall.appendChild(addInput);
    columns.appendChild(colCall);

    /* ---- 第 3 列：参数 + 绑定 ---- */
    const colParams = makerEl('div', 'maker-col');
    colParams.appendChild(makerEl('h4', null, '参数（int / number / bool / string / literal / select）'));
    const paramsBox = makerEl('div');
    paramsBox.id = 'maker-params';
    colParams.appendChild(paramsBox);
    colParams.appendChild(makerButton('＋ 添加参数', () => {
      const draft = App.maker.draft;
      if (draft.params.length >= NS.MAX_PARAMS) { App.makerStatus(`参数最多 ${NS.MAX_PARAMS} 个。`, 'err'); return; }
      draft.params.push({ key: 'param' + (draft.params.length + 1), label: '参数' + (draft.params.length + 1), type: 'int', def: '0', help: '', options: [] });
      App.makerRefresh();
    }, 'mini'));

    colParams.appendChild(makerEl('h4', null, '实参绑定（位置 / 关键词；可选输入可回退到参数，连线优先）'));
    const argsBox = makerEl('div');
    argsBox.id = 'maker-args';
    colParams.appendChild(argsBox);
    colParams.appendChild(makerButton('＋ 添加绑定', () => {
      const draft = App.maker.draft;
      if (draft.call.args.length >= NS.MAX_ARGS) { App.makerStatus(`绑定最多 ${NS.MAX_ARGS} 个。`, 'err'); return; }
      draft.call.args.push({ source: 'input', index: 1, keyword: null, fallbackParam: null });
      App.makerRefresh();
    }, 'mini'));
    columns.appendChild(colParams);

    /* ---- 第 4 列：预览 + 试运行 + 动作 ---- */
    const colPreview = makerEl('div', 'maker-col');
    colPreview.appendChild(makerEl('h4', null, '实时卡片预览'));
    const cardBox = makerEl('div');
    cardBox.id = 'maker-card-preview';
    colPreview.appendChild(cardBox);
    colPreview.appendChild(makerEl('h4', null, 'Python 预览（in1/in2… 代表输入）'));
    const codeView = makerEl('pre');
    codeView.id = 'maker-code-preview';
    colPreview.appendChild(codeView);
    colPreview.appendChild(makerEl('h4', null, '试运行样本（仅字面量 + 明确张量选项）'));
    /* 信任边界提醒：自由目标与 imports 是真开放（仅语法校验），试运行会真实执行
     * 本机 Python；超时子进程只是兜底，不是安全沙箱；只按按钮执行，不自动运行。 */
    const warning = makerEl('div', 'maker-warning');
    warning.textContent = '⚠ 信任提醒：调用目标与模块导入会真实执行你本机 Python 里的代码——只填写并执行你信任的本地 Python 函数；'
      + '函数和 import 可能产生文件/网络/系统副作用；试运行是带超时的本地子进程，不是安全沙箱。'
      + '试运行只在你点击「▶ 独立试运行」按钮时执行；项目正式运行沿用同一信任边界。';
    colPreview.appendChild(warning);
    const samplesBox = makerEl('div');
    samplesBox.id = 'maker-samples';
    colPreview.appendChild(samplesBox);
    const actions = makerEl('div', 'maker-row');
    actions.appendChild(makerButton('验证', () => App.makerValidate(), 'primary'));
    actions.appendChild(makerButton('▶ 独立试运行', () => App.makerTryRun(), 'primary'));
    actions.appendChild(makerButton('发布到本项目', () => App.makerPublish(), 'primary'));
    colPreview.appendChild(actions);
    const status = makerEl('div', 'maker-status');
    status.id = 'maker-status';
    colPreview.appendChild(status);
    columns.appendChild(colPreview);

    panel.appendChild(columns);
    $('#canvas-stack').appendChild(panel);
  };

  /* ---- 列表重建 + 预览刷新 ---- */
  App.makerRefresh = function () {
    if (!App.maker.built) return;
    App.makerRefreshMeta();
    App.makerRefreshCall();
    App.makerRefreshImports();
    App.makerRefreshInputs();
    App.makerRefreshParams();
    App.makerRefreshArgs();
    App.makerRefreshPublished();
    App.makerRefreshPreview();
    App.makerRefreshSamples();
  };

  App.makerRefreshMeta = function () {
    const box = $('#maker-meta');
    const draft = App.maker.draft;
    box.replaceChildren();
    const idInput = makerInput(draft.id, '小写字母开头，如 my_relu', v => { draft.id = v.trim(); App.makerRefreshPreview(); });
    box.appendChild(makerField('规范 id', idInput, '节点类型身份的一部分；自定义类型编码为 custom.<id>@<版本>，例如 custom.sqrt@1。'));
    box.appendChild(makerField('版本 version', makerInput(String(draft.version ?? 1), '', v => { draft.version = parseInt(v, 10) || 1; App.makerRefreshPreview(); }),
      '与 id 共同构成不可变身份。同 id+version 不能覆盖不同内容；再次发布会自动产生新版本，旧节点仍绑旧版本。'));
    box.appendChild(makerField('显示名称 label', makerInput(draft.label, '如 自定义 ReLU', v => { draft.label = v; App.makerRefreshPreview(); }), '显示在节点卡片的黄色标题行。'));
    const desc = document.createElement('textarea');
    desc.rows = 2; desc.value = draft.description || '';
    desc.addEventListener('input', () => { draft.description = desc.value; });
    box.appendChild(makerField('说明 description', desc, '节点的用途说明。'));
    box.appendChild(makerField('分类 category', makerInput(draft.category, '自定义', v => { draft.category = v; }), '组件栏分组名，默认「自定义」。'));
    box.appendChild(makerField('输出名称', makerInput(draft.output.name, 'out', v => { draft.output.name = v; App.makerRefreshPreview(); }), '自定义节点固定只有一个输出。'));
    box.appendChild(makerField('输出类型提示', makerSelect(NS.INPUT_TYPE_HINTS.map(t => ({ value: t, label: t })),
      draft.output.type || 'Tensor', v => { draft.output.type = v; App.makerRefreshPreview(); }),
      '仅展示提示，不做类型转换；自由目标返回 Python 标量/容器时可选 string/bool/list/tuple/dict 等。单输出不变，tuple 返回值作为一个整体对象。'));
  };

  App.makerRefreshCall = function () {
    const box = $('#maker-call');
    const draft = App.maker.draft;
    box.replaceChildren();
    box.appendChild(makerField('调用类型', makerSelect(
      [{ value: 'function', label: '函数（自由目标）' }, { value: 'method', label: '方法（对象方法名）' }],
      draft.call.kind,
      v => {
        draft.call.kind = v;
        if (v === 'method' && String(draft.call.target ?? '').includes('.')) {
          draft.call.target = NS.METHOD_TARGETS[0];   // 方法名不能含点号，退回建议值
        }
        if (!String(draft.call.target ?? '').trim()) {
          draft.call.target = v === 'function' ? NS.FUNCTION_TARGETS[0] : NS.METHOD_TARGETS[0];
        }
        App.makerRefresh();
      }),
      'function 生成 目标(...)，目标可自由填写合法点分标识符；method 生成 接收者.方法名(...)，接收者固定为输入 1。'));
    const isFunction = draft.call.kind === 'function';
    const suggestions = isFunction ? NS.FUNCTION_TARGETS : NS.METHOD_TARGETS;
    /* 目标可自由填写：可编辑文本框 + datalist 候选建议（不再受白名单限制，
     * 只在规范校验时做 Python 语法检查）。 */
    const targetInput = makerInput(draft.call.target ?? '', isFunction ? '如 torch.sqrt、math.sqrt、builtins.len、包.子模块.函数' : '如 item、tolist、masked_fill',
      v => { draft.call.target = v.trim(); App.makerRefreshPreview(); });
    targetInput.setAttribute('list', 'maker-target-options');
    box.appendChild(makerField(isFunction ? '调用目标（自由填写）' : '方法名（单个标识符）', targetInput,
      isFunction
        ? '任意合法点分 Python 标识符：math.sqrt / torch.sqrt / 包.子模块.函数 / builtins.len 等；每段不能是保留字，拒绝空格、分号、换行、括号、通配符（不可注入额外语句）。没有隐藏白名单，但只执行你信任的函数；所需模块请在下方 imports 声明。'
        : '合法单个 Python 方法名（不能含点号，不能是保留字）：接收者仍是输入 1；支持无参数方法（item/tolist），此时不需要任何绑定；输出可为 Python 标量/容器（用输出类型提示声明）。'));
    const datalist = makerEl('datalist');
    datalist.id = 'maker-target-options';
    for (const t of suggestions) {
      const opt = document.createElement('option');
      opt.value = t;
      datalist.appendChild(opt);
    }
    box.appendChild(datalist);
    if (draft.call.kind === 'method') {
      const receiverRow = makerEl('div', 'maker-row');
      receiverRow.appendChild(makerEl('span', 'maker-note', '接收者 receiver：输入 1（隐式，生成 x.method(...)）'));
      box.appendChild(receiverRow);
    }
  };

  /* ---- 声明式 imports 编辑：{module, alias?} → import <module> [as <alias>] ----
   * 默认 torch / torch.nn as nn 已存在：重复声明会被规范校验静默去重；
   * 别名与 torch/nn 冲突或两条导入绑定同名会被明确拒绝。 */
  App.makerRefreshImports = function () {
    const box = $('#maker-imports');
    const draft = App.maker.draft;
    if (!Array.isArray(draft.imports)) draft.imports = [];
    box.replaceChildren();
    draft.imports.forEach((entry, i) => {
      const item = makerEl('div', 'maker-item');
      const head = makerEl('div', 'maker-row');
      head.appendChild(makerEl('strong', null, `导入 ${i + 1}`));
      head.appendChild(makerButton('删除', () => { draft.imports.splice(i, 1); App.makerRefresh(); }, 'mini danger'));
      item.appendChild(head);
      item.appendChild(makerField('模块名 module', makerInput(entry.module ?? '', '如 math、numpy、collections.abc',
        v => { entry.module = v.trim(); App.makerRefreshPreview(); }),
        '合法点分模块名，生成 import <module> [as <alias>]；模块必须已安装（后端不会自动安装依赖，也不支持动态加载任意文件路径）。'));
      item.appendChild(makerField('别名 alias（可留空）', makerInput(entry.alias ?? '', '如 np（留空用模块首段）',
        v => { entry.alias = v.trim() || null; App.makerRefreshPreview(); }),
        '省略时按 import <module> 生成（点分模块绑定首段名字）；别名 torch / nn 与默认导入冲突会被拒绝；与默认 torch、torch.nn as nn 重复的声明自动去重。'));
      box.appendChild(item);
    });
    if (!draft.imports.length) {
      box.appendChild(makerEl('div', 'maker-note', '（暂无自定义导入：只使用 torch/nn 或 builtins 内置名字时可留空）'));
    }
  };

  App.makerRefreshInputs = function () {
    const box = $('#maker-inputs');
    const draft = App.maker.draft;
    box.replaceChildren();
    draft.inputs.forEach((input, i) => {
      const item = makerEl('div', 'maker-item');
      const head = makerEl('div', 'maker-row');
      head.appendChild(makerEl('strong', null, (draft.call.kind === 'method' && i === 0) ? `输入 ${i + 1}（接收者）` : `输入 ${i + 1}`));
      const move = (delta) => {
        if (draft.call.kind === 'method' && (i === 0 || i + delta === 0)) { App.makerStatus('方法调用的接收者（输入 1）必须保持在第一位。', 'err'); return; }
        const target = i + delta;
        if (target < 0 || target >= draft.inputs.length) return;
        [draft.inputs[i], draft.inputs[target]] = [draft.inputs[target], draft.inputs[i]];
        App.makerSwapBindings(i + 1, target + 1);
        App.makerRefresh();
      };
      if (i > 0) head.appendChild(makerButton('↑', () => move(-1), 'mini'));
      if (i < draft.inputs.length - 1) head.appendChild(makerButton('↓', () => move(1), 'mini'));
      head.appendChild(makerButton('删除', () => {
        draft.inputs.splice(i, 1);
        draft.call.args = draft.call.args.filter(a => !(a.source === 'input' && a.index === i + 1));
        for (const binding of draft.call.args) if (binding.source === 'input' && binding.index > i + 1) binding.index--;
        App.makerRefresh();
      }, 'mini danger'));
      item.appendChild(head);
      item.appendChild(makerField('名称', makerInput(input.name, '', v => { input.name = v; App.makerRefreshPreview(); })));
      item.appendChild(makerField('类型提示', makerSelect(NS.INPUT_TYPE_HINTS.map(t => ({ value: t, label: t })),
        input.type || 'Tensor', v => { input.type = v; App.makerRefreshPreview(); })));
      const optional = document.createElement('input');
      optional.type = 'checkbox'; optional.checked = !!input.optional;
      optional.addEventListener('change', () => {
        input.optional = optional.checked;
        if (!optional.checked) input.fallbackParam = null;
        App.makerRefresh();
      });
      item.appendChild(makerField('可选输入', optional, '可选输入可以不连线；配合「回退参数」，连线时优先用连线值（连线优先），未连线时用参数值。'));
      const paramOptions = [{ value: '', label: '（无）' },
        ...draft.params.map(p => ({ value: p.key, label: p.key }))];
      item.appendChild(makerField('回退参数 fallbackParam', makerSelect(paramOptions, input.fallbackParam || '', v => { input.fallbackParam = v || null; }),
        '仅可选输入可用：未连线时该输入对应的实参取此参数的值。'));
      item.appendChild(makerField('输入说明', makerInput(input.help, '', v => { input.help = v; })));
      box.appendChild(item);
    });
  };

  /* 输入位置交换后，同步绑定里指向这两个位置的序号引用 */
  App.makerSwapBindings = function (p, q) {
    for (const binding of App.maker.draft.call.args) {
      if (binding.source !== 'input') continue;
      if (binding.index === p) binding.index = q;
      else if (binding.index === q) binding.index = p;
    }
  };

  App.makerRefreshParams = function () {
    const box = $('#maker-params');
    const draft = App.maker.draft;
    box.replaceChildren();
    draft.params.forEach((param, i) => {
      const item = makerEl('div', 'maker-item');
      const head = makerEl('div', 'maker-row');
      head.appendChild(makerEl('strong', null, `参数 ${i + 1}`));
      if (i > 0) head.appendChild(makerButton('↑', () => { [draft.params[i], draft.params[i - 1]] = [draft.params[i - 1], draft.params[i]]; App.makerRefresh(); }, 'mini'));
      head.appendChild(makerButton('删除', () => {
        draft.params.splice(i, 1);
        draft.call.args = draft.call.args.filter(a => a.key !== param.key);
        for (const input of draft.inputs) if (input.fallbackParam === param.key) input.fallbackParam = null;
        for (const binding of draft.call.args) if (binding.fallbackParam === param.key) binding.fallbackParam = null;
        App.makerRefresh();
      }, 'mini danger'));
      item.appendChild(head);
      item.appendChild(makerField('key', makerInput(param.key, '小写标识符', v => { param.key = v.trim(); })));
      item.appendChild(makerField('显示名', makerInput(param.label, '', v => { param.label = v; })));
      item.appendChild(makerField('类型', makerSelect(NS.PARAM_TYPES.map(t => ({ value: t, label: t })), param.type, v => {
        param.type = v;
        param.options = v === 'select' ? (param.options?.length ? param.options : ['选项1']) : [];
        if (v === 'bool') param.def = param.def === true || param.def === 'true';
        else if (param.def === true || param.def === 'true') param.def = '0';
        App.makerRefresh();
      })));
      const defControl = param.type === 'bool'
        ? (() => { const c = document.createElement('input'); c.type = 'checkbox'; c.checked = param.def === true || param.def === 'true'; c.addEventListener('change', () => { param.def = c.checked; App.makerRefreshPreview(); }); return c; })()
        : makerInput(param.def ?? '', param.type === 'select' ? '必须是选项之一' : '默认值', v => { param.def = v; App.makerRefreshPreview(); });
      item.appendChild(makerField('默认值 def', defControl,
        param.type === 'literal' ? 'Python 字面量源码（发布与编译时经后端 ast.literal_eval 白名单校验，不接受表达式）'
          : param.type === 'number' ? '数字，支持 inf / -inf / nan'
            : param.type === 'select' ? '必须等于某个选项' : ''));
      if (param.type === 'select') {
        item.appendChild(makerField('选项（逗号分隔）', makerInput((param.options || []).join(', '), '选项1, 选项2', v => {
          param.options = v.split(',').map(s => s.trim()).filter(Boolean);
        })));
      }
      item.appendChild(makerField('参数说明', makerInput(param.help, '', v => { param.help = v; })));
      box.appendChild(item);
    });
  };

  App.makerRefreshArgs = function () {
    const box = $('#maker-args');
    const draft = App.maker.draft;
    box.replaceChildren();
    const sourceOptions = [
      ...draft.inputs.map((input, i) => ({ value: 'input:' + (i + 1), label: `输入 ${i + 1}（${input.name || '?'}）` })),
      ...draft.params.map(p => ({ value: 'param:' + p.key, label: `参数 ${p.key}` })),
    ];
    draft.call.args.forEach((binding, i) => {
      const item = makerEl('div', 'maker-item');
      const head = makerEl('div', 'maker-row');
      head.appendChild(makerEl('strong', null, `实参 ${i + 1}`));
      if (i > 0) head.appendChild(makerButton('↑', () => { [draft.call.args[i], draft.call.args[i - 1]] = [draft.call.args[i - 1], draft.call.args[i]]; App.makerRefresh(); }, 'mini'));
      head.appendChild(makerButton('删除', () => { draft.call.args.splice(i, 1); App.makerRefresh(); }, 'mini danger'));
      item.appendChild(head);
      const current = binding.source === 'input' ? 'input:' + binding.index : 'param:' + binding.key;
      item.appendChild(makerField('绑定源', makerSelect(sourceOptions, sourceOptions.some(o => o.value === current) ? current : sourceOptions[0].value, v => {
        const [source, key] = v.split(':');
        if (source === 'input') { binding.source = 'input'; binding.index = parseInt(key, 10); binding.key = undefined; }
        else { binding.source = 'param'; binding.key = key; binding.index = undefined; }
        App.makerRefresh();
      })));
      item.appendChild(makerField('关键字名（可选）', makerInput(binding.keyword || '', '留空按位置传参', v => { binding.keyword = v.trim() || null; })));
      if (binding.source === 'input') {
        const input = draft.inputs[binding.index - 1];
        if (input && !input.optional) {
          const note = makerEl('div', 'maker-row');
          note.appendChild(makerEl('span', 'maker-note', '该输入是必选输入：画布上必须连线。'));
          item.appendChild(note);
        } else {
          item.appendChild(makerField('回退参数 fallbackParam', makerSelect(
            [{ value: '', label: '（无）' }, ...draft.params.map(p => ({ value: p.key, label: p.key }))],
            binding.fallbackParam || '', v => { binding.fallbackParam = v || null; }),
            '未连线时此实参取该参数的值；连线时优先用连线（连线优先）。'));
        }
      }
      box.appendChild(item);
    });
    if (draft.call.kind === 'method') {
      const note = makerEl('div', 'maker-row');
      note.appendChild(makerEl('span', 'maker-note', '方法调用：输入 1 自动作为接收者，不能出现在绑定里。'));
      box.appendChild(note);
    }
  };

  App.makerRefreshPublished = function () {
    const box = $('#maker-published');
    box.replaceChildren();
    if (!App.customRegistry.size()) {
      box.appendChild(makerEl('div', 'maker-note', '（本项目还没有已发布的自定义节点）'));
      return;
    }
    for (const spec of App.customRegistry.list()) {
      const row = makerEl('div', 'maker-row');
      row.appendChild(makerEl('span', null, NS.encode(spec.id, spec.version) + ' · ' + spec.label));
      row.appendChild(makerButton('编辑副本', () => {
        App.makerBindDraft(NS.deepCopy(spec));
        App.makerRefresh();
        App.makerStatus(`已载入 ${NS.encode(spec.id, spec.version)} 的副本；发布时会自动生成新版本 ${NS.encode(spec.id, App.customRegistry.nextVersion(spec.id))}。`, 'ok');
      }, 'mini'));
      box.appendChild(row);
    }
  };

  /* 实时卡片（固定 260px、黄色功能标题）+ Python 预览 + 校验状态 */
  App.makerRefreshPreview = function () {
    const cardBox = $('#maker-card-preview'), codeView = $('#maker-code-preview');
    if (!cardBox) return;
    const draft = App.maker.draft;
    cardBox.replaceChildren();
    let spec = null, problem = '';
    try { spec = NS.normalize(draft, literalHook); } catch (e) { problem = e.message; }
    const card = makerEl('div', 'mk-node');
    const head = makerEl('div', 'mk-head');
    head.appendChild(makerEl('span', 'mk-title', draft.id || 'my_op'));
    card.appendChild(head);
    card.appendChild(makerEl('div', 'mk-callable', draft.label || '（显示名称 label）'));
    if (spec) {
      card.appendChild(makerEl('div', 'mk-summary', NS.describeCall(spec, {}, literalHook)));
    }
    const pins = makerEl('div', 'mk-pins');
    (draft.inputs || []).forEach((input, i) => {
      const row = makerEl('div', 'mk-pin-row');
      row.appendChild(makerEl('span', 'mk-pin-label', `${input.name || '?'}: ${input.type || 'Tensor'}${input.optional ? ' · 可选' : ''}`));
      row.appendChild(makerEl('span', 'mk-pin-label mk-out', ''));
      pins.appendChild(row);
    });
    const outRow = makerEl('div', 'mk-pin-row');
    outRow.appendChild(makerEl('span', 'mk-pin-label', ''));
    outRow.appendChild(makerEl('span', 'mk-pin-label mk-out', `${draft.output?.name || 'out'}: ${draft.output?.type || 'Tensor'}`));
    pins.appendChild(outRow);
    card.appendChild(pins);
    cardBox.appendChild(card);
    if (problem) {
      codeView.textContent = `# 规范暂无效：${problem}`;
      return;
    }
    codeView.textContent = NS.previewStatement(spec);
  };

  /* ---- 试运行样本编辑 ---- */
  App.makerDefaultSample = function (input) {
    const hint = input.type || 'Tensor';
    if (hint === 'Tensor') return { mode: 'tensor', shape: '2, 2', dtype: 'float32', fill: 'constant', value: '1' };
    if (hint === 'BoolTensor') return { mode: 'tensor', shape: '2, 2', dtype: 'bool', fill: 'constant', value: 'True' };
    if (hint === 'string') return { mode: 'literal', source: "'知之'" };
    if (hint === 'dict') return { mode: 'literal', source: "{'a': 1}" };
    if (hint === 'list') return { mode: 'literal', source: '[1, 2]' };
    if (hint === 'tuple') return { mode: 'literal', source: '(1, 2)' };
    if (hint === 'bool') return { mode: 'literal', source: 'True' };
    return { mode: 'literal', source: '1' };
  };
  App.makerRefreshSamples = function () {
    const box = $('#maker-samples');
    const draft = App.maker.draft;
    box.replaceChildren();
    (draft.inputs || []).forEach((input, i) => {
      const sample = App.maker.samples[i] || (App.maker.samples[i] = App.makerDefaultSample(input));
      const item = makerEl('div', 'maker-item');
      item.appendChild(makerField(`输入 ${i + 1}（${input.name || '?'}）样本`, makerSelect(
        [{ value: 'literal', label: '字面量' }, { value: 'tensor', label: '张量' }, { value: 'omit', label: '省略（模拟未连线）' }],
        sample.mode, v => {
          sample.mode = v;
          if (v === 'tensor' && !sample.dtype) { sample.shape = '2, 2'; sample.dtype = 'float32'; sample.fill = 'constant'; sample.value = '1'; }
          if (v === 'literal' && sample.source === undefined) sample.source = '1';
          App.makerRefreshSamples();
        })));
      if (sample.mode === 'literal') {
        item.appendChild(makerField('字面量', makerInput(sample.source, '如 1、[1, 2]、True', v => { sample.source = v; })));
      } else if (sample.mode === 'tensor') {
        item.appendChild(makerField('shape', makerInput(sample.shape, '如 2, 3', v => { sample.shape = v; })));
        item.appendChild(makerField('dtype', makerSelect(['float32', 'float64', 'float16', 'bfloat16', 'int64', 'int32', 'bool'].map(t => ({ value: t, label: t })),
          sample.dtype || 'float32', v => { sample.dtype = v; })));
        item.appendChild(makerField('填充', makerSelect([{ value: 'constant', label: '常量值' }, { value: 'rand', label: '随机（种子 1234）' }],
          sample.fill || 'constant', v => { sample.fill = v; App.makerRefreshSamples(); })));
        if ((sample.fill || 'constant') === 'constant') {
          item.appendChild(makerField('填充值', makerInput(sample.value, '如 1、-inf', v => { sample.value = v; })));
        }
      } else {
        item.appendChild(makerEl('div', 'maker-note', '该输入按未连线处理（仅可选输入允许省略）。'));
      }
      box.appendChild(item);
    });
  };

  App.makerBuildSamplesPayload = function (spec) {
    const payload = {};
    spec.inputs.forEach((input, i) => {
      const sample = App.maker.samples[i];
      if (!sample || sample.mode === 'omit') {
        if (!input.optional) throw new NS.SpecError(`输入 ${i + 1}（${input.name}）是必选输入，必须提供试运行样本`);
        return;
      }
      if (sample.mode === 'literal') payload[String(i + 1)] = { kind: 'literal', source: sample.source };
      else payload[String(i + 1)] = { kind: 'tensor', shape: String(sample.shape || '').split(',').map(s => parseInt(s.trim(), 10)).filter(n => Number.isFinite(n) && n > 0),
        dtype: sample.dtype || 'float32', fill: sample.fill || 'constant', value: sample.value };
    });
    return payload;
  };

  /* ---- 动作：验证 / 试运行 / 发布 ---- */
  App.makerValidate = async function () {
    try {
      await App.makerWarmLiterals();
      const spec = App.normalizeDraft();
      App.makerStatus(`✓ 规范合法：${NS.encode(spec.id, spec.version)}（${spec.call.kind === 'method' ? '方法' : '函数'} ${spec.call.target}`
        + `${(spec.imports || []).length ? '，imports ' + spec.imports.map(i => i.alias ? `${i.module} as ${i.alias}` : i.module).join('；') : ''}）。发布前建议先独立试运行。`, 'ok');
      return spec;
    } catch (e) {
      App.makerStatus('✗ ' + e.message, 'err');
      return null;
    }
  };
  App.makerWarmLiterals = async function () {
    const draft = App.maker.draft;
    for (const p of (draft.params || [])) {
      if (p.type !== 'literal') continue;
      const source = String(p.def ?? '').trim();
      if (!source || App.literalCache.has(source)) continue;
      const response = await fetch('/api/literal', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source }),
      });
      const result = await response.json();
      if (!response.ok && !result.error) throw Error('基础值校验服务不可用');
      App.literalCache.set(source, result);
    }
  };

  /* 预览条目 → 一行短文本（试运行结果里的嵌套序列/映射用）。 */
  App.makerPreviewLine = function (p) {
    if (p === null || p === undefined || typeof p !== 'object') return String(p);
    if (p.kind === 'tensor') return `Tensor(shape=[${(p.shape || []).join(', ')}], dtype=${p.dtype})`;
    if (p.kind === 'sequence') return `(${(p.items || []).map(App.makerPreviewLine).join(', ')})`;
    if (p.kind === 'mapping') return `{${(p.entries || []).map(([k, v]) => `${k}: ${App.makerPreviewLine(v)}`).join(', ')}}`;
    return String(p.repr ?? '');
  };

  App.makerTryRun = async function () {
    if (App.maker.busy) return;
    /* 跨项目守卫：试运行是 async，等待期间可能切换项目（草稿/样本已被新项目清空）；
     * 旧项目的候选结果与生成代码一律丢弃，防止污染当前项目的制作器状态。 */
    const identity = App.project?.id ?? null;
    const runSeq = ++App.maker.runSeq;
    App.maker.busy = true;
    const button = [...$('#nodespec-maker').querySelectorAll('button')].find(b => b.textContent === '▶ 独立试运行');
    if (button) { button.disabled = true; }
    try {
      const spec = await App.makerValidate();
      if (!spec) return;
      const samples = App.makerBuildSamplesPayload(spec);
      const result = await App.request('/api/nodespec/try', { spec, samples });
      if ((App.project?.id ?? null) !== identity) {
        App.makerStatus('✗ 已丢弃本次试运行结果：等待期间项目已切换，旧项目的候选结果不会污染当前项目的制作器。', 'err');
        return;
      }
      if (runSeq !== App.maker.runSeq) return;   // 已有更新的试运行接管结果显示
      let text = '';
      if (result.ok && result.preview) {
        const p = result.preview;
        if (p.kind === 'tensor') {
          text = `✓ 试运行成功\nTensor · shape [${(p.shape || []).join(', ')}] · ${p.dtype} · ${p.numel} 个元素` +
            (p.requires_grad ? ' · grad' : '') + '\n值：' + JSON.stringify(p.values) + (p.truncated ? ` …（已截断，共 ${p.numel}）` : '');
        } else if (p.kind === 'value') {
          text = `✓ 试运行成功\n${p.type} · ${p.repr}` + (p.truncated ? ' …（已截断）' : '');
        } else if (p.kind === 'sequence') {
          text = `✓ 试运行成功\n${p.type}（${p.count} 个元素${p.truncated ? '，预览前 ' + (p.items || []).length : ''}）\n(${(p.items || []).map(App.makerPreviewLine).join(', ')})`
            + '\n（tuple 作为一个整体对象，仍是单输出，不自动拆成多输出）';
        } else if (p.kind === 'mapping') {
          text = `✓ 试运行成功\n${p.type}（${p.count} 个条目${p.truncated ? '，预览前 ' + (p.entries || []).length : ''}）\n{`
            + (p.entries || []).map(([k, v]) => `${k}: ${App.makerPreviewLine(v)}`).join(', ') + '}';
        } else {
          text = `✓ 试运行成功\n${p.type || '结果'} · ${p.repr ?? JSON.stringify(p)}`;
        }
        App.makerStatus(text, 'ok');
      } else {
        text = '✗ 试运行失败：' + (result.error || ('退出码 ' + result.returncode)) +
          (result.stderr ? '\n' + result.stderr.split('\n').slice(-4).join('\n') : '');
        App.makerStatus(text, 'err');
      }
      let codeBox = $('#maker-try-code');
      if (!codeBox) {
        codeBox = makerEl('details', 'maker-json');
        codeBox.id = 'maker-try-code';
        codeBox.appendChild(makerEl('summary', null, '本次试运行的后端生成代码'));
        const pre = makerEl('pre');
        pre.id = 'maker-try-code-view';
        codeBox.appendChild(pre);
        $('#maker-status').before(codeBox);
      }
      codeBox.querySelector('pre').textContent = result.code || '';
      codeBox.open = !result.ok;
    } catch (e) {
      App.makerStatus('✗ 试运行请求失败：' + e.message, 'err');
    } finally {
      App.maker.busy = false;
      if (button) button.disabled = false;
    }
  };

  /* ---- 动作：发布。审批要求「未被画布使用的定义也保存」：
   *      新定义 / 新版本发布成功后必须进入项目保存队列（App.scheduleSync →
   *      editRevision + keepDraft 浏览器草稿 + syncNow /api/save，快照携带 nodeSpecs），
   *      否则刷新后未使用定义会丢失；内容未变化的幂等发布不改状态、不排队、不增版本；
   *      任何失败路径都不改动注册表与项目存档；等待校验期间项目被切换则取消发布。 ---- */
  App.makerPublish = async function () {
    const identity = App.project?.id ?? null;   // 捕获项目身份：await 期间项目可能被切换
    try {
      await App.makerWarmLiterals();
      if ((App.project?.id ?? null) !== identity) {
        App.makerStatus('✗ 发布已取消：等待校验期间项目已切换，不能把定义发布进另一个项目。', 'err');
        return;
      }
      const spec = App.normalizeDraft();
      let result, updated = false;
      try {
        result = App.customRegistry.publish(spec, literalHook);   // 同 id+version 同内容 → 幂等，不产生新版本
      } catch (e) {
        if (!(e instanceof NS.VersionConflict)) throw e;   // 其余失败不动注册表、不动存档
        updated = true;
        result = App.customRegistry.publishUpdate(spec, literalHook);
      }
      installDef(result.spec);
      App.buildPalette();
      App.maker.draft = App.makerBindDraft(NS.deepCopy(result.spec));
      App.makerRefresh();
      if (result.created) {
        App.scheduleSync();
        App.makerStatus(updated
          ? `✓ 已发布新版本 ${result.encoded}（旧版本保持不变，旧节点仍绑旧版本）；组件栏「${result.spec.category}（自定义节点）」可拖入画布；已进入项目保存队列。`
          : `✓ 已发布 ${result.encoded}；组件栏「${result.spec.category}（自定义节点）」可拖入画布；已进入项目保存队列。`, 'ok');
      } else {
        App.makerStatus(`${result.encoded} 内容未变化，无需重复发布。`, 'ok');
      }
    } catch (e) {
      App.makerStatus('✗ 发布失败：' + e.message + '（项目存档未改动）', 'err');
    }
  };

  /* ---- 顶栏按钮 ---- */
  window.addEventListener('DOMContentLoaded', () => {
    const button = $('#btn-maker');
    if (button) button.addEventListener('click', App.toggleMaker);
  });
})();
