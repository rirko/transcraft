/* Project persistence overrides: serialize saves, snapshot identity, retain invalid drafts. */
App.project = null;
App.editRevision = 0;
App.savedEditRevision = -1;
App.loadingProject = false;
App.projectBusy = false;
App.saveQueue = Promise.resolve();
App.snapshot = () => JSON.parse(JSON.stringify({version:3, seed:App.state.seed,
  main:ED.exportData('main'), functions:App.state.fnOrder.map(name => ({name,
    ...App.state.definitions[name], data:ED.exportData('fn:'+name)}))}));
App.request = async function(path, data) {
  const response = await fetch(path, data === undefined ? {} : {method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) { const error=Error(result.error || 'HTTP '+response.status); error.status=response.status; throw error; }
  return result;
};
App.projectStatus = function(text, kind='ok') {
  $('#sync-status').textContent=text; $('#sync-status').className=kind;
  $('#sync-status').title=text;
};
App.keepDraft = function() {
  if (!App.project || App.loadingProject || App.restoreFailed) return;
  try { localStorage.setItem('canvas-draft:'+App.project.id, JSON.stringify({
    version:App.project.version, graph:App.snapshot()})); }
  catch(e) { App.projectStatus('浏览器草稿备份失败: '+e.message,'err'); }
};
App.scheduleSync = function(immediate=false) {
  if(App.loadingProject || !App.project) return;
  App.editRevision++;
  App.renderTensorViewer?.();
  App.keepDraft();
  clearTimeout(App.syncTimer);
  App.syncTimer=setTimeout(()=>App.syncNow(), immediate?0:400);
};
App.syncNow = function() {
  clearTimeout(App.syncTimer);
  const identity=App.project?.id;
  const work=App.saveQueue.then(async()=> {
    if(!identity || App.project?.id!==identity || App.restoreFailed) return false;
    // Refresh signatures before capturing the graph. Hooks are muted for derived updates.
    App.loadingProject=true;
    try { App.refreshSignatures(); } finally { App.loadingProject=false; }
    const revision=App.editRevision;
    const graph=App.snapshot();
    const version=App.project.version;
    let code=null, problem='';
    try {
      await App.validateLiterals();
      if(App.project.id!==identity || App.editRevision!==revision) return false;
      code=App.generatePython();
      App.renderCodePreview(code);
      if(App.lastErrors?.length) throw Error(App.lastErrors.join('; '));
    } catch(e) { problem=e.message; code=null; }
    if(App.project.id!==identity || App.editRevision!==revision) return false;
    try {
      let result;
      try { result=await App.request('/api/save',{project_id:identity,version,graph,code,trace_plan:code===null?null:App.tracePlan}); }
      catch(e) {
        // Compile rejection may still allow an incomplete draft; conflicts must remain conflicts.
        if(code===null || e.status!==422) throw e;
        problem=e.message; code=null;
        result=await App.request('/api/save',{project_id:identity,version,graph,code:null});
      }
      if(App.project.id!==identity) return false;
      App.project=result.project;
      App.revision=result.revision;
      App.savedEditRevision=revision;
      if(App.editRevision===revision) localStorage.removeItem('canvas-draft:'+identity);
      else App.keepDraft();
      App.projectStatus(problem?'草稿已保存，不能运行: '+problem:'已保存 · '+App.project.name,problem?'warn':'ok');
      return {project_id:identity,version:App.project.version,revision:result.revision,editRevision:revision};
    } catch(e) {
      App.keepDraft(); App.projectStatus('保存失败: '+e.message,'err'); return false;
    }
  });
  App.saveQueue=work.catch(()=>false);
  return work;
};
App.refreshProjects = async function() {
  const result=await App.request('/api/projects');
  const select=$('#project-select'); select.replaceChildren();
  for(const project of result.projects) {
    const option=document.createElement('option');option.value=project.id;option.textContent=project.name;
    select.appendChild(option);
  }
  if(App.project) select.value=App.project.id;
  return result;
};
App.loadProject = function(project) {
  App.loadingProject=true;
  clearTimeout(App.syncTimer);
  try {
    ED.clearClipboard();
    App.clearAllModels(); App.state.definitions={};
    App.state.seed=project.graph.seed??1234;
    for(const f of project.graph.functions) {
      App.state.definitions[f.name]={kind:f.kind||'model',params:f.params||''};
      App.createFunctionRaw(f.name); ED.importData('fn:'+f.name,f.data);
    }
    ED.importData('main',project.graph.main);
    App.project=project; App.revision=null; App.restoreFailed=false;
    App.runtimeTrace=null; App.renderTensorViewer?.();
    App.editRevision=0; App.savedEditRevision=0;
    App.renderTabs(); App.renderFnPalette(); App.switchTab('main');
    App.codeSourceMap={}; App.renderCodePreview(project.code||'# Draft');
    $('#output-view').textContent=''; $('#project-select').value=project.id;
    App.projectStatus('已打开 · '+project.name);
  } finally { App.loadingProject=false; }
};
App.projectAction = async function(action) {
  if(App.projectBusy) return;
  App.projectBusy=true;
  document.body.classList.add('project-busy');
  $('#main').inert=true; $('#topbar').inert=true;
  try { await action(); }
  catch(e) { App.projectStatus(e.message,'err'); alert(e.message); }
  finally {
    App.projectBusy=false; document.body.classList.remove('project-busy');
    $('#main').inert=false; $('#topbar').inert=false;
    if(App.project) $('#project-select').value=App.project.id;
  }
};
App.flushProject = async function() {
  const saved=await App.syncNow();
  if(!saved || saved.editRevision!==App.editRevision) throw Error('当前修改尚未保存，切换已取消。可另存为新项目。');
};
App.openProject = id => App.projectAction(async()=> {
  if(id===App.project?.id) return;
  await App.flushProject();
  const {project}=await App.request('/api/project?id='+encodeURIComponent(id));
  App.loadProject(project); await App.recoverDraft(); App.scheduleSync();
});
App.askProjectName=function(title,value='',submit='保存') {
  return new Promise(resolve=>{
    const dialog=document.createElement('dialog');dialog.id='project-name-dialog';dialog.setAttribute('aria-labelledby','project-name-title');
    const form=document.createElement('form');form.method='dialog';dialog.appendChild(form);
    const heading=document.createElement('h3');heading.id='project-name-title';heading.textContent=title;form.appendChild(heading);
    const label=document.createElement('label');label.textContent='项目名称';form.appendChild(label);
    const input=document.createElement('input');input.id='project-name-input';input.name='project-name';input.value=value;input.required=true;input.maxLength=100;input.setAttribute('aria-label','项目名称');label.appendChild(input);
    const actions=document.createElement('div');actions.className='dialog-actions';form.appendChild(actions);
    const cancel=document.createElement('button');cancel.type='button';cancel.textContent='取消';cancel.onclick=()=>dialog.close('cancel');actions.appendChild(cancel);
    const save=document.createElement('button');save.type='submit';save.textContent=submit;save.className='primary';actions.appendChild(save);
    form.onsubmit=e=>{e.preventDefault();if(input.value.trim())dialog.close('save');};
    dialog.onclose=()=>{const result=dialog.returnValue==='save'?input.value.trim():null;dialog.remove();resolve(result);};
    document.body.appendChild(dialog);dialog.showModal();input.focus();input.select();
  });
};
App.recoverDraft = async function() {
  const raw=localStorage.getItem('canvas-draft:'+App.project.id);
  if(!raw) return;
  const draft=JSON.parse(raw);
  localStorage.setItem('canvas-recovery:'+App.project.id+':'+Date.now(),raw);
  const name=await App.askProjectName('恢复未同步草稿（原项目保留）',App.project.name+' recovered','恢复为新项目');
  if(name) {
    const sourceId=App.project.id;
    const result=await App.request('/api/projects/save-as',{project_id:sourceId,name,graph:draft.graph});
    localStorage.removeItem('canvas-draft:'+sourceId);
    App.loadProject(result.project); await App.refreshProjects();
  } else {
    // Do not auto-save over a preserved browser draft after declining recovery.
    App.restoreFailed=true;
    throw Error('浏览器草稿仍保留；自动保存已暂停，可点恢复草稿');
  }
};
App.initProjects = async function() {
  $('#main').inert=true; $('#topbar').inert=true;
  $('#project-select').onchange=e=>App.openProject(e.target.value);
  $('#btn-save').onclick=()=>App.syncNow();
  $('#btn-project-new').onclick=()=>App.projectAction(async()=> {
    const name=await App.askProjectName('新建项目','','新建'); if(name===null) return;
    await App.flushProject();
    const {project}=await App.request('/api/projects/create',{name});
    App.loadProject(project); await App.refreshProjects(); App.scheduleSync();
  });
  $('#btn-project-save-as').onclick=()=>App.projectAction(async()=> {
    const name=await App.askProjectName('另存为独立项目',App.project.name+' copy','另存为'); if(name===null) return;
    clearTimeout(App.syncTimer); await App.saveQueue;
    const {project}=await App.request('/api/projects/save-as',{name,project_id:App.project.id,graph:App.snapshot()});
    App.loadProject(project); await App.refreshProjects(); App.scheduleSync();
  });
  $('#btn-project-rename').onclick=()=>App.projectAction(async()=> {
    const name=await App.askProjectName('重命名项目',App.project.name,'重命名'); if(name===null) return;
    await App.flushProject();
    const {project}=await App.request('/api/projects/rename',{name,project_id:App.project.id,version:App.project.version});
    App.project=project; await App.refreshProjects(); App.scheduleSync();
  });
  window.addEventListener('beforeunload',e=> {
    if(App.project && App.savedEditRevision!==App.editRevision) {App.keepDraft(); e.preventDefault(); e.returnValue='';}
  });
  try {
    const list=await App.refreshProjects();
    const {project}=await App.request('/api/project?id='+list.default_id);
    App.loadProject(project); await App.recoverDraft(); App.scheduleSync();
  } catch(e) {App.restoreFailed=true; App.projectStatus('恢复失败，自动保存已暂停: '+e.message,'err');}
  finally { $('#main').inert=false; $('#topbar').inert=false; }
};
App.run = async function() {
  if(App.projectBusy) return;
  await App.projectAction(async()=> {
    $('#btn-run').disabled=true;
    try {
      const saved=await App.syncNow();
      if(!saved?.revision || saved.editRevision!==App.editRevision) throw Error('草稿未通过校验，已取消运行');
      const result=await App.request('/api/run',saved);
      if(App.project.id!==saved.project_id) return;
      App.runtimeTrace={...result.trace,editRevision:saved.editRevision};
      App.renderTensorViewer?.();
      $('#output-view').textContent=`$ ${App.project.name} / generated_model.py\n${result.stdout||''}\n${result.stderr||''}\n[退出码 ${result.returncode}]`;
      $('#output-view').className=result.returncode===0?'ok':'bad';
      $('#output-panel').classList.add('open');
    } finally {$('#btn-run').disabled=false;}
  });
};
