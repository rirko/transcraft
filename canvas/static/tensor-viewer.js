/* Runtime previews are text-only and keyed by project/revision/scope/node. */
App.tensorChoice={};
App.tensorIsExpired=function(trace=App.runtimeTrace) {
  return !!trace && (trace.project_id!==App.project?.id || trace.revision!==App.project?.revision || trace.editRevision!==App.editRevision);
};
App.tensorText=function(tag,text,parent,className='') {
  const el=document.createElement(tag);el.textContent=text;el.className=className;parent.appendChild(el);return el;
};
App.renderTensorViewer=function(key,id) {
  if(key===undefined) {
    for(const drawer of App.drawers.values()) App.renderTensorViewer(drawer.key,drawer.id);
    return;
  }
  const selected=App.nodeDrawer(key,id);
  const node=selected&&App.nodeElement(selected.key,selected.id);
  const box=node?.querySelector('.nd-runtime-view'),status=node?.querySelector('.nd-runtime-status');
  if(!box || !status) return;
  const drawer=node.querySelector('.nd-drawer'),top=drawer.scrollTop,left=drawer.scrollLeft;
  const tables=Array.from(box.querySelectorAll('.tensor-table-wrap'),el=>[el.scrollTop,el.scrollLeft]);
  const refresh=()=>{App.selectNode(selected.key,selected.id);App.renderTensorViewer(selected.key,selected.id);};
  try {
  box.replaceChildren();
  const trace=App.runtimeTrace;
  const expired=App.tensorIsExpired(trace);
  status.textContent=!trace?'尚未运行':expired?'已过期':trace.error?'运行失败 · 部分结果':'本次运行';
  status.className='nd-runtime-status '+(expired||trace?.error?'warn':'ok');
  if(!trace) {App.tensorText('p','尚未运行',box,'dim');return;}
  if(trace.error) App.tensorText('pre',trace.error,box,'tensor-error');
  if(trace.limited) App.tensorText('p','已达到本次运行记录上限（512）',box,'warn');
  if(!selected) {App.tensorText('p','未选中节点',box,'dim');return;}
  const choiceKey=JSON.stringify([trace.project_id,selected.key,String(selected.id)]);
  const records=(trace.records||[]).filter(r=>r.scope===selected.key && r.node===String(selected.id));
  const plan=(trace.plan||[]).find(r=>r.scope===selected.key && r.node===String(selected.id));
  const choice=App.tensorChoice[choiceKey]||={sample:0,samples:{}};
  function select(id,label,options,value,change) {
    const row=document.createElement('label');row.className='tensor-control';box.appendChild(row);
    App.tensorText('span',label,row);const control=document.createElement('select');control.dataset.tensorControl=id;row.appendChild(control);control.setAttribute('aria-label',label);
    options.forEach(([v,text])=>{const o=document.createElement('option');o.value=v;o.textContent=text;control.appendChild(o);});control.value=value;control.onchange=()=>change(Number(control.value));
  }
  if(!records.length) App.tensorText('p',plan?.type==='print'||plan?.type==='print_shape'?'此节点仅打印，无输出':'本次运行未执行此节点或无捕获值',box,'dim');
  else {
    choice.sample=Math.min(choice.sample,records.length-1);
    select('tensor-invocation','调用样本',records.map((r,i)=>[i,`#${i+1} · ${r.context.map(c=>c.scope.replace('fn:','')+'.'+c.phase+' #'+c.invocation).join(' → ')||'main'}`]),choice.sample,v=>{choice.sample=v;refresh();});
    const record=records[choice.sample],outputs=record.outputs;
    const sampleChoice=choice.samples[choice.sample]||={output:0,outputs:{}};
    sampleChoice.output=Math.min(sampleChoice.output,Math.max(0,outputs.length-1));
    if(record.outputs_truncated) App.tensorText('p',`输出已截断：仅前 16 / ${record.output_count} 项`,box,'warn');
    if(outputs.length) {
      select('tensor-output','输出',outputs.map((p,i)=>[i,`${i+1}: ${p.kind==='tensor'?p.dtype:p.type}`]),sampleChoice.output,v=>{sampleChoice.output=v;refresh();});
      const outputChoice=sampleChoice.outputs[sampleChoice.output]||={slices:[],items:{}};
      App.renderTensorValue(outputs[sampleChoice.output],box,outputChoice,refresh);
    }
  }
  if(plan) {
    const sources=document.createElement('details');sources.dataset.tensorPart='sources';sources.open=true;box.appendChild(sources);
    App.tensorText('summary','来源与代码',sources);
    let links=(plan.sources||[]).map(s=>({...s,scope:selected.key}));
    const sample=records[choice.sample];
    const caller=sample?.context?.at(-1)?.caller;
    if(plan.type==='fn_input' && caller) {
      const upstream=caller.sources[plan.inputIndex??0];
      if(upstream) links=[{...upstream,scope:caller.scope,name:`${caller.scope} #${caller.node} · ${upstream.name}`}];
    }
    for(const source of links) {
      if(!source.node) {App.tensorText('div',source.name+' ← 静态参数',sources,'dim');continue;}
      const button=App.tensorText('button',source.name+' ← #'+source.node+' · output '+source.output,sources,'tensor-source');
      button.dataset.sourceNode=source.node;button.disabled=expired;button.title=expired?'旧版本来源；当前图已修改':'选中上游节点';
      button.onclick=e=>{
        e?.stopPropagation();
        if(!ED.canvases[source.scope]||!ED.getNode(source.scope,source.node)) return;
        App.switchTab(source.scope);
        const canvas=ED.canvases[source.scope];canvas.container.querySelectorAll('.drawflow-node.selected').forEach(n=>n.classList.remove('selected'));
        canvas.container.querySelector(`[id="node-${source.node}"]`)?.classList.add('selected');
        App.showInspector(source.scope,source.node);
      };
    }
    const codeLines=trace.code.split('\n');
    App.tensorText('pre',plan.lines.map(n=>`${n}  ${codeLines[n-1]}`).join('\n'),sources,'tensor-expression');
  }
  } finally {
    box.querySelectorAll('.tensor-table-wrap').forEach((el,i)=>{
      if(tables[i]) {el.scrollTop=tables[i][0];el.scrollLeft=tables[i][1];}
    });
    drawer.scrollTop=top;drawer.scrollLeft=left;
    ED.redraw(selected.key);
  }
};
App.renderTensorValue=function(value,box,choice,refresh) {
  if(value.kind==='sequence') {
    App.tensorText('p',`${value.type} · ${value.count} 项${value.truncated?' · 仅前 8 项':''}`,box);
    value.items.forEach((item,i)=>{App.tensorText('p','['+i+']',box,'dim');App.renderTensorValue(item,box,choice.items[i]||={slices:[],items:{}},refresh);});return;
  }
  if(value.kind!=='tensor') {App.tensorText('pre',value.type+'\n'+value.repr,box,'tensor-repr');return;}
  App.tensorText('p',`shape [${value.shape.join(', ')}] · ${value.dtype} · ${value.device}${value.requires_grad?' · grad':''}`,box,'tensor-meta');
  App.tensorText('p',`${value.captured||0} / ${value.numel} 个值${value.truncated?' · 已截断（仅起始区域）':''}`,box,value.truncated?'warn':'dim');
  if(value.unavailable) {App.tensorText('p',value.unavailable,box,'warn');return;}
  const dims=value.preview_shape,controls=document.createElement('div');controls.dataset.tensorPart='slice-controls';box.appendChild(controls);
  for(let axis=0;axis<dims.length-2;axis++) {
    const label=document.createElement('label');label.className='tensor-control';controls.appendChild(label);
    App.tensorText('span',`dim ${axis}${axis===0?' (batch)':axis===1?' (head)':''}`,label);
    const input=document.createElement('input');input.type='number';input.min=0;input.max=Math.max(0,dims[axis]-1);input.step=1;input.dataset.axis=axis;
    input.setAttribute('aria-label','切片 dim '+axis);input.title=`可查看 0–${dims[axis]-1}；原维度 ${value.shape[axis]}`;
    choice.slices[axis]=Math.min(Math.max(0,choice.slices[axis]||0),Math.max(0,dims[axis]-1));input.value=choice.slices[axis];label.appendChild(input);
    input.oninput=input.onchange=()=>{choice.slices[axis]=Math.min(Math.max(0,Math.floor(Number(input.value)||0)),Math.max(0,dims[axis]-1));refresh();};
  }
  if(!value.values.length) {App.tensorText('p','空张量',box,'dim');return;}
  if(!dims.length) {App.tensorText('pre',String(value.values[0]),box,'tensor-repr');return;}
  const rows=dims.length===1?1:dims[dims.length-2],cols=dims[dims.length-1];
  let offset=0;for(let axis=0;axis<dims.length-2;axis++) offset+=(choice.slices[axis]||0)*dims.slice(axis+1).reduce((a,b)=>a*b,1);
  const wrap=document.createElement('div');wrap.className='tensor-table-wrap';box.appendChild(wrap);
  const table=document.createElement('table');table.className='tensor-matrix';table.dataset.tensorPart='matrix';wrap.appendChild(table);table.setAttribute('aria-label','张量数值矩阵');
  const header=document.createElement('tr');table.appendChild(header);App.tensorText('th','r / c',header);
  for(let col=0;col<cols;col++) App.tensorText('th',String(col),header).scope='col';
  for(let row=0;row<rows;row++) {
    const tr=document.createElement('tr');table.appendChild(tr);App.tensorText('th',String(row),tr).scope='row';
    for(let col=0;col<cols;col++) {
      const original=value.values[offset+row*cols+col];
      const text=typeof original==='number'&&!Number.isInteger(original)?Number(original.toPrecision(6)).toString():String(original);
      const cell=App.tensorText('td',text,tr);cell.title=String(original);cell.dataset.row=row;cell.dataset.col=col;
    }
  }
};
const tensorProjectStatus=App.projectStatus;
App.projectStatus=function(...args) {tensorProjectStatus(...args);$('#btn-draft-recover').hidden=!App.restoreFailed;};
window.addEventListener('DOMContentLoaded',()=>{
  $('#btn-draft-export').onclick=()=>{
    const storage=Object.fromEntries(Object.entries(localStorage).filter(([key])=>key.startsWith('canvas-draft:')||key.startsWith('canvas-recovery:')));
    const blob=new Blob([JSON.stringify({project:App.project,graph:App.snapshot(),storage},null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='canvas-browser-backup-'+Date.now()+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  $('#btn-draft-recover').onclick=()=>App.projectAction(async()=>{
    if(!App.project) throw Error('项目未载入，先导出备份');
    const raw=localStorage.getItem('canvas-draft:'+App.project.id);
    if(!raw) throw Error('未找到该项目的草稿，先导出当前画布备份');
    const draft=JSON.parse(raw);localStorage.setItem('canvas-recovery:'+App.project.id+':'+Date.now(),raw);
    const name=await App.askProjectName('恢复未同步草稿（原项目保留）',App.project.name+' recovered','恢复为新项目');if(!name) return;
    const result=await App.request('/api/projects/save-as',{project_id:App.project.id,name,graph:draft.graph});
    App.loadProject(result.project);await App.refreshProjects();App.scheduleSync();$('#btn-draft-recover').hidden=true;
  });
});
