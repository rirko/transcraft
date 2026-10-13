/* V3: scoped lifecycle, explicit instances, ordinary functions; legacy calls remain supported. */
App.state.definitions = {};
const oldCreate = App.createFunctionRaw;
App.createFunctionRaw = function(name) {
  App.state.definitions[name] ||= {kind:'model', params:''};
  return oldCreate(name);
};
Object.assign(NODE_DEFS, {
  create: {label:'创建模型实例', defaults:{name:'model'}, inputs:0, outputs:1, inNames:[], outNames:['instance: Module'], params:[{key:'init_args',label:'构造实参（逗号分隔）',type:'text',def:''}], summary:d=>`new ${d.fname}(${d.init_args || ''})`},
  invoke: {label:'调用已有实例', defaults:{name:'result'}, inputs:1, outputs:1, inNames:[], outNames:[], params:[], dynamic:true, summary:d=>`${d.fname}.forward`},
  function_call: {label:'普通函数调用', defaults:{name:'result'}, inputs:0, outputs:1, inNames:[], outNames:[], params:[], dynamic:true, summary:d=>`${d.fname}(...)`}
});
NODE_DEFS.call.params=NODE_DEFS.create.params;
for(const t of ['embedding','linear']) for(const p of NODE_DEFS[t].params) if(p.type==='int') p.type='text';
STMT_TYPES.push('create','invoke','function_call','variable_ref');
const pythonIdent = value => {
  const text=String(value).normalize('NFKC');
  return /^[_\p{ID_Start}][_\p{ID_Continue}]*$/u.test(text) && !PY_KEYWORDS.has(text);
};
function resolveVariables(recs, errors, opts) {
  const symbols=new Map(),members=new Map();
  const add=(map,name,value)=>{if(!map.has(name))map.set(name,[]);map.get(name).push(value);};
  for(const r of Object.values(recs)) {
    if(r.isParam || STMT_TYPES.includes(r.type)) {
      r.outVars.slice(0,Object.keys(r.n.outputs||{}).length).forEach((name,port)=>add(symbols,name,{up:r,port,expression:opts.isClass&&r.type==='create'?`self.${name}`:name}));
    }
    if(opts.isClass && ['embedding','linear','call','create'].includes(r.type)) add(members,r.outVar,{up:r,port:0,expression:`self.${r.outVar}`,member:true});
  }
  if(opts.isClass) for(const name of opts.constructorParams||[]) {
    const value={up:null,port:0,expression:`self.${name}`,constructorParam:name};
    add(symbols,name,value);add(members,name,value);
  }
  for(const r of Object.values(recs)) if(r.type==='variable_ref') {
    const text=String(r.n.data.variable??'').trim(),attr=text.startsWith('self.'),name=(attr?text.slice(5):text).normalize('NFKC');
    let problem='';
    if(!pythonIdent(name) || (!attr && text.includes('.'))) problem='只允许变量名或 self.已声明成员，不接受运算、索引、调用等表达式';
    else if(attr&&!opts.isClass) problem='self. 引用只能用于模型 forward';
    const choices=(attr?members:symbols).get(name)||[];
    if(!problem && choices.length===1 && choices[0].constructorParam && ['training','forward','__init__'].includes(name)) problem='此名称为模型生命周期保留成员，不能提升构造参数';
    if(!problem && choices.length!==1) problem=choices.length?'名称有歧义，请使用代码预览中的唯一实际变量名':'当前作用域没有此变量；请声明参数或检查代码预览中的输出名（如 result_out2）';
    if(problem) {
      r.referenceError=true;r.skipReason=problem;
      errors.push(`变量引用「${text}」无效：${problem}`);continue;
    }
    r.variableSource=choices[0];
    if(r.variableSource.up) r.inRefs.push({...r.variableSource,implicit:true});
  }
}
const oldScope=buildScope;
buildScope=function(df,opts) {
  const s=oldScope(df,{...opts,isClass:false,resolveReferences:(recs,errors)=>resolveVariables(recs,errors,opts)});
  for(const [id,n] of Object.entries(df)) {
    if(!NODE_DEFS[n.data?.type]) throw Error('未知节点类型: '+id);
    for(const p of Object.values(n.inputs||{})) if(p.connections.length>1) throw Error('每个输入只允许一条连线: '+id);
  }
  for(const r of Object.values(s.recs)) {
    if(r.type==='create' && (!App.state.fnOrder.includes(r.n.data.fname)||App.state.definitions[r.n.data.fname]?.kind==='function')) s.errors.push('无效模型创建目标');
    if(['call','invoke','function_call'].includes(r.type)) {
      const sig=App.fnSignature(r.n.data.fname);
    if(r.type==='function_call' && App.state.definitions[r.n.data.fname]?.kind!=='function') s.errors.push('目标不是普通函数');
    if(['invoke','call'].includes(r.type) && App.state.definitions[r.n.data.fname]?.kind==='function') s.errors.push('目标不是模型类');
      if(!App.state.fnOrder.includes(r.n.data.fname)) s.errors.push('未定义: '+r.n.data.fname);
      if(Object.keys(r.n.inputs).length!==sig.inputs.length+(r.type==='invoke'?1:0)||Object.keys(r.n.outputs).length!==sig.outputs.length) s.errors.push('接口已改变，请更新调用连线: '+r.outVar);
    }
    r.isModule=!!opts.isClass && ['embedding','linear','call','create'].includes(r.type);
    if(!r.valid && !r.isParam && r.type!=='fn_output') s.errors.push('输入不完整: '+r.outVar);
    for(const ref of r.inRefs) if(ref.up && (ref.port<0||ref.port>=Object.keys(ref.up.n.outputs).length)) s.errors.push('无效输出端口: '+r.outVar);
  }
  return s;
};
const intExpr=v=>{
  const s=String(v).trim();
  if(!/^-?\d+$/.test(s)&&!(App.compileParams||[]).includes(s)) throw Error('整数或构造参数无效: '+s);
  return s;
};
moduleInitOf=function(r) {
  const d=r.n.data;
  if(r.type==='embedding') return `nn.Embedding(${intExpr(d.vocab_size)}, ${intExpr(d.embedding_dim)})`;
  if(r.type==='linear') return `nn.Linear(${intExpr(d.in_features)}, ${intExpr(d.out_features)}, bias=${asBool(d.bias)?'True':'False'})`;
  const params=String(App.state.definitions[d.fname]?.params||'').split(',').map(x=>x.trim()).filter(Boolean);
  const args=String(d.init_args||'').split(',').map(x=>x.trim()).filter(Boolean);
  if(args.length!==params.length) throw Error(d.fname+' 构造参数需要: '+params.join(', '));
  return `${d.fname}(${args.map(intExpr).join(', ')})`;
};
const oldStmt=stmtOf;
const dtypeExpr=value=>{
  if(!['bool','float32','float64','float16','bfloat16','int64','int32'].includes(value)) throw Error('不支持的 dtype: '+value);
  return 'torch.'+value;
};
const scalarExpr=value=>{
  const text=String(value).trim().toLowerCase();
  if(['inf','+inf','-inf','nan'].includes(text)) return `float(${JSON.stringify(text)})`;
  if(!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/.test(text)||!Number.isFinite(Number(text))) throw Error('value 必须是数字、inf、-inf 或 nan');
  return text;
};
stmtOf=function(r) {
  const d=r.n.data,args=r.argStrs.join(', '),lhs=r.outVars.join(', ');
  if(r.type==='variable_ref') return `${r.outVar} = ${r.variableSource.expression}`;
  if(r.type==='ones_like') {
    const options=[];
    if((d.dtype??'inherit')!=='inherit') options.push('dtype='+dtypeExpr(d.dtype));
    if((d.device??'inherit')!=='inherit') {
      if(!['cpu','cuda'].includes(d.device)) throw Error('不支持的 device');
      options.push('device='+JSON.stringify(d.device));
    }
    return `${r.outVar} = torch.ones_like(${r.argStrs[0]}${options.length?', '+options.join(', '):''})`;
  }
  if(r.type==='dtype') return `${r.outVar} = ${r.argStrs[0]}.to(dtype=${dtypeExpr(d.dtype??'bool')})`;
  if(r.type==='triu') {
    const diagonal=r.argStrs[1]??String(d.diagonal??0).trim();
    if(!r.argStrs[1]&&!/^-?\d+$/.test(diagonal)) throw Error('diagonal 必须是整数');
    return `${r.outVar} = torch.triu(${r.argStrs[0]}, diagonal=${diagonal})`;
  }
  if(r.type==='masked_fill') return `${r.outVar} = ${r.argStrs[0]}.masked_fill(${r.argStrs[1]}, ${r.argStrs[2]??scalarExpr(d.value??'-inf')})`;
  if(r.type==='create') return `${r.outVar} = ${moduleInitOf(r)}`;
  if(r.type==='invoke') return `${lhs} = ${r.argStrs[0]}(${r.argStrs.slice(1).join(', ')})`;
  if(r.type==='function_call') return `${lhs} = ${d.fname}(${args})`;
  if(r.type==='call') return [`${r.instanceVar} = ${moduleInitOf(r)}`,`${lhs} = ${r.instanceVar}(${args})`];
  return oldStmt(r);
};
App.generatePython=function() {
  const lines=['import torch','import torch.nn as nn','',`torch.manual_seed(${asInt(App.state.seed,1234)})`,''];
  const sourceMap=Object.create(null);
  const tracePlan=[];
  // Track physical lines at emission time, including multiline statements.
  function push(text,key,ids=[]) {
    const parts=text.split('\n'),first=lines.length+1;
    lines.push(...parts);
    for(const id of ids) {
      const mapped=sourceMap[key][String(id)];
      parts.forEach((_,i)=>mapped.push(first+i));
    }
  }
  const errors=[];
  const visiting=new Set(),done=new Set();
  for(const n of Object.values(ED.exportData('main'))) if(['fn_input','fn_output'].includes(n.data.type)) throw Error('主画布不能有函数接口节点');
  function visit(name) {
    if(['_canvas_trace','_canvas_return'].includes(name)) throw Error('定义名称为运行追踪保留名: '+name);
    if(visiting.has(name)) throw Error('递归定义依赖: '+name);
    if(done.has(name)) return;
    visiting.add(name);
    for(const n of Object.values(ED.exportData('fn:'+name))) if(['call','create','function_call'].includes(n.data.type)) {
      if(!App.state.fnOrder.includes(n.data.fname)) throw Error('未定义: '+n.data.fname);
      visit(n.data.fname);
    }
    visiting.delete(name);done.add(name);
  }
  App.state.fnOrder.forEach(visit);
  function emit(key,name) {
    const meta=App.state.definitions[name]||{},model=!!name&&meta.kind!=='function';
    App.compileParams=String(meta.params||'').split(',').map(x=>x.trim().normalize('NFKC')).filter(Boolean);
    if(new Set(App.compileParams).size!==App.compileParams.length||App.compileParams.some(x=>!pythonIdent(x)||['self','torch','nn','_canvas_trace','_canvas_return'].includes(x))) throw Error('构造参数名无效');
    const s=buildScope(ED.exportData(key),{isClass:model,constructorParams:App.compileParams,reserved:['self','torch','nn','_canvas_trace','_canvas_return','training','forward','__init__',...App.state.fnOrder,...App.compileParams]});
    errors.push(...s.errors.map(x=>`[${key}] ${x}`));
    sourceMap[key]=Object.create(null);
    for(const id of Object.keys(s.recs)) sourceMap[key][id]=[];
    const body=[],init=[],ret=[],constructorParamLines=new Map();
    const statement=(text,id)=>({text,ids:[id]});
    const promoted=new Set();
    for(const r of Object.values(s.recs)) if(r.valid && r.variableSource?.constructorParam) promoted.add(r.variableSource.constructorParam);
    for(const name of promoted) init.push({text:`self.${name} = ${name}`,ids:[],constructorParam:name});
    function instanceOf(r,seen=new Set()) {
      if(!r || seen.has(r.id)) return null;
      seen.add(r.id);
      if(r.type==='create') return r;
      if(r.type==='variable_ref') {
        if(r.variableSource?.member) return r.variableSource.up;
        return instanceOf(r.variableSource?.up,seen);
      }
      return null;
    }
    for(const id of s.order) {
      const r=s.recs[id]; if(!r.valid) continue;
      r.argStrs=r.inRefs.map(x=>model&&x.up?.type==='create'?`self.${x.up.outVar}`:x.up?.outVars[x.port]);
      if(r.type!=='variable_ref' && r.inRefs.some((x,i)=>instanceOf(x.up) && !(r.type==='invoke' && i===0))) errors.push('模型实例只能连接调用实例端口');
      if(r.type==='invoke') {
        const up=instanceOf(r.inRefs[0]?.up);
        if(!up||up.n.data.fname!==r.n.data.fname) errors.push('实例端口必须连接同类型模型创建节点');
      }
      if(r.isModule) {
        init.push(statement(`self.${r.outVar} = ${moduleInitOf(r)}`,id));
        if(r.type!=='create') body.push(statement(`${r.outVars.join(', ')} = self.${r.outVar}(${r.argStrs.join(', ')})`,id));
      } else {const st=stmtOf(r);body.push(...(Array.isArray(st)?st:[st]).map(text=>statement(text,id)));}
    }
    for(const r of Object.values(s.recs)) if(r.type==='fn_output') for(const x of r.inRefs) {
      if(!x.up?.valid||!x.up.outVars[x.port]) errors.push('返回值未连接或无效');
      else ret.push(x.up.outVars[x.port]);
    }
    const params=Object.values(s.recs).filter(r=>r.isParam),args=params.map(r=>r.outVar);
    if(name) {
      if(!ret.length) errors.push(name+' 缺少返回值');
      body.push({text:'return '+(ret.join(', ')||'None'),ids:Object.values(s.recs).filter(r=>r.type==='fn_output').map(r=>r.id)});
      if(model) {
        push(`class ${name}(nn.Module):`);
        push(`    def __init__(self${App.compileParams.map(x=>', '+x).join('')}):`);
        push('        super().__init__()');
        init.forEach(x=>{if(x.constructorParam)constructorParamLines.set(x.constructorParam,lines.length+1);push('        '+x.text,key,x.ids);});
        push('');
        push(`    def forward(self${args.map(x=>', '+x).join('')}):`,key,params.map(r=>r.id));
      } else push(`def ${name}(${args.join(', ')}):`,key,params.map(r=>r.id));
      body.forEach(x=>push((model?'        ':'    ')+x.text,key,x.ids));
      push('');
    } else {
      push('if __name__ == "__main__":');
      if(!body.length) push('    pass');
      body.forEach(x=>push('    '+x.text,key,x.ids));
    }
    for(const r of Object.values(s.recs)) {
      tracePlan.push({scope:key,node:r.id,type:r.type,name:r.n.data.name||r.outVar,
        lines:sourceMap[key][r.id],
        inputIndex:r.isParam?params.findIndex(p=>p.id===r.id):null,
        expressions:r.isParam?[r.outVar]:r.type==='fn_output'?r.inRefs.map(x=>x.up?.outVars[x.port]).filter(Boolean):[],
        sources:r.variableSource?[{input:1,node:r.variableSource.up?.id??null,output:r.variableSource.port+1,
          name:r.variableSource.constructorParam?`__init__ 构造参数 ${r.variableSource.constructorParam} → ${r.variableSource.expression}（代码行 ${constructorParamLines.get(r.variableSource.constructorParam)}）`:`变量 ${r.variableSource.expression}`}]:
          r.inRefs.map((x,i)=>({input:i+1,node:x.up?.id??null,output:x.port+1,
          name:x.staticExpr!=null?`输入 ${ED.binopName(i)}（手填 ${x.manualText}）`:
            r.type==='binop'?ED.binopName(i):
            NODE_DEFS[r.type]?.inNames?.[i]||r.n.data.inLabels?.[i]||`input ${i+1}`}))});
    }
  }
  for(const name of App.state.fnOrder) emit('fn:'+name,name);
  emit('main',null);
  App.codeSourceMap=sourceMap;
  App.tracePlan=tracePlan;
  App.lastErrors=errors;return lines.join('\n')+'\n';
};
const oldAdd=ED.addNodeTo;
ED.addNodeTo=function(key,type,x,y,fname='') {
  if(!['invoke','function_call','create'].includes(type)) return oldAdd(key,type,x,y,fname);
  const sig=App.fnSignature(fname);
  const data={type,fname,name:type==='create'?'model':'result',init_args:'',inLabels:type==='create'?[]:[...(type==='invoke'?['instance']:[]),...sig.inputs],outLabels:type==='create'?['instance']:sig.outputs};
  const ed=ED.canvases[key].editor;
  const id=ed.addNode(type,data.inLabels.length,data.outLabels.length,x,y,'node-call',data,ED.nodeHTML({data}));
  ED.refreshSummary(key,id);return id;
};
ED.api.add=(...args)=>ED.addNodeTo(...args);
const oldMeta=App.fnSignature;
App.fnSignature=function(name) {if(!ED.canvases['fn:'+name]) return {inputs:[],outputs:[]};return oldMeta(name);};
App.renderFnPalette=function() {
  const wrap=$('#palette-fns');wrap.innerHTML='';
  for(const name of App.state.fnOrder) {
    const types=App.state.definitions[name]?.kind==='function'?['function_call']:['create','invoke'];
    for(const type of types) {
      const el=App.paletteItem(type,name);
      el.querySelector('.pi-name').textContent=NODE_DEFS[type].label+' '+name;
      el.querySelector('.pi-desc').textContent=type==='create'?`init(${App.state.definitions[name]?.params||''})`:`(${App.fnSignature(name).inputs.join(', ')})`;
      wrap.appendChild(el);
    }
  }
};
const oldTabs=App.renderTabs;
App.renderTabs=function() {
  oldTabs();
  const b=document.createElement('button');b.textContent='+ 函数';
  b.onclick=()=>{const name=prompt('普通函数名称','my_function');if(name){const n=App.createFunction(name);App.state.definitions[n].kind='function';App.renderFnPalette();App.scheduleSync();}};
  $('#tabs').appendChild(b);
};
const oldSwitch=App.switchTab;
App.modelPhase='forward';
App.renderModelPhase=function(key,phase='forward') {
  if (phase === 'init') App.closeNodeDrawers(key);
  document.getElementById('model-phase-tabs')?.remove();
  document.getElementById('model-init-panel')?.remove();
  const stack=$('#canvas-stack');
  stack.classList.remove('has-model-phases');
  if(key==='main'||App.state.definitions[key.slice(3)]?.kind==='function') return;
  App.modelPhase=phase;
  const name=key.slice(3),meta=App.state.definitions[name];
  stack.classList.add('has-model-phases');
  const tabs=document.createElement('div');tabs.id='model-phase-tabs';tabs.setAttribute('role','tablist');tabs.setAttribute('aria-label','模型生命周期');
  for(const [value,label] of [['init','__init__'],['forward','forward']]) {
    const b=document.createElement('button');b.id='model-tab-'+value;b.textContent=label;b.setAttribute('role','tab');
    b.setAttribute('aria-selected',String(value===phase));b.setAttribute('aria-controls',value==='init'?'model-init-panel':App.containerOf(key).id);
    b.onclick=()=>{App.hideInspector();App.renderModelPhase(key,value);};tabs.appendChild(b);
  }
  stack.prepend(tabs);
  const canvas=App.containerOf(key);canvas.style.display=phase==='forward'?'block':'none';
  if(phase!=='init') return;
  const panel=document.createElement('section');panel.id='model-init-panel';panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby','model-tab-init');
  const heading=document.createElement('h3');heading.textContent=name+'.__init__';panel.appendChild(heading);
  const label=document.createElement('label');label.textContent='构造参数';
  const input=document.createElement('input');input.id='model-constructor-params';input.value=meta.params||'';
  input.setAttribute('aria-label','构造参数');
  input.onchange=()=>{meta.params=input.value;App.renderFnPalette();App.scheduleSync();};label.appendChild(input);panel.appendChild(label);
  const title=document.createElement('h4');title.textContent='注册成员';panel.appendChild(title);
  const list=document.createElement('div');list.id='model-members';
  for(const [id,n] of Object.entries(ED.exportData(key))) {
    if(!['embedding','linear','create','call'].includes(n.data.type)) continue;
    const row=document.createElement('div');row.className='model-member';row.dataset.nodeId=id;
    const button=document.createElement('button');button.dataset.memberId=id;
    button.textContent='self.'+n.data.name+' · '+(n.data.fname||NODE_DEFS[n.data.type].label);
    button.onclick=()=>{App.renderModelPhase(key,'forward');App.selectNode(key,id);};row.appendChild(button);
    for(const p of NODE_DEFS[n.data.type].params) {
      const field=document.createElement('label');field.textContent=p.label;
      const control=document.createElement('input');control.dataset.memberParam=p.key;
      control.type=p.type==='bool'?'checkbox':'text';control.value=n.data[p.key]??'';control.checked=asBool(n.data[p.key]);
      control.onchange=()=>{ED.updateNode(key,id,{[p.key]:control.type==='checkbox'?control.checked:control.value});App.scheduleSync();};
      field.appendChild(control);row.appendChild(field);
    }
    list.appendChild(row);
  }
  if(!list.children.length) list.textContent='（无注册成员）';
  panel.appendChild(list);stack.appendChild(panel);
};
App.switchTab=function(key) {
  oldSwitch(key);
  App.renderModelPhase(key,'forward');
  document.getElementById('definition-settings')?.remove();
  $('#canvas-stack').classList.remove('has-definition-toolbar');
  if(key==='main') return;
  const name=key.slice(3),meta=App.state.definitions[name];
  const box=document.createElement('div');box.id='definition-settings';
  box.textContent=meta.kind==='function'?'普通 def 函数 · '+name:'';
  if(meta.kind==='function') {
    $('#canvas-stack').prepend(box);
    $('#canvas-stack').classList.add('has-definition-toolbar');
  } else $('#canvas-stack').classList.remove('has-definition-toolbar');
};
App.refreshSignatures=function() {
  for(const [key,{editor}] of Object.entries(ED.canvases)) for(const [id,n] of Object.entries(ED.exportData(key))) {
    if(!['call','invoke','function_call'].includes(n.data.type)) continue;
    const sig=App.fnSignature(n.data.fname),inputs=[...(n.data.type==='invoke'?['instance']:[]),...sig.inputs],outputs=sig.outputs;
    while(Object.keys(ED.getNode(key,id).inputs).length<inputs.length) editor.addNodeInput(id);
    while(Object.keys(ED.getNode(key,id).outputs).length<outputs.length) editor.addNodeOutput(id);
    ED.updateNode(key,id,{inLabels:inputs,outLabels:outputs});
  }
  App.renderFnPalette();
};
const oldSync=App.syncNow;
App.syncNow=function() {
  clearTimeout(App.syncTimer);
  if(App.syncing) return App.syncing.then(()=>App.syncNow());
  App.refreshSignatures();
  App.syncing=oldSync().finally(()=>{App.syncing=null;});
  return App.syncing;
};
const oldInspector=App.showInspector;
App.showInspector=function(key,id) {
  oldInspector(key,id);
  const hint=App.nodeElement(key,id)?.querySelector('.ins-hint');
  if(hint) hint.textContent='兼容旧图：此节点创建并调用模型。新图使用独立创建和调用节点。';
};
// Built-in demonstrations use explicit main instances; restored v2 graphs stay untouched.
for(const example of ['example1','example2']) {
  const original=App[example];
  App[example]=function() {
    const before=JSON.stringify(ED.exportData('main'));
    original();
    if(before===JSON.stringify(ED.exportData('main'))) return;
    for(const [id,n] of Object.entries(ED.exportData('main'))) if(n.data.type==='call') {
      const instance=ED.api.add('main','create',n.pos_x,n.pos_y+180,n.data.fname);
      if(example==='example1') {
        App.state.definitions.ShankeModel.params='vocab_size, embedding_dim';
        for(const [eid,en] of Object.entries(ED.exportData('fn:ShankeModel'))) if(en.data.type==='embedding') ED.updateNode('fn:ShankeModel',eid,{vocab_size:'vocab_size',embedding_dim:'embedding_dim'});
        ED.updateNode('main',instance,{init_args:'50275, 768'});
      }
      const df=ED.exportData('main'),call=df[id];
      call.data.type='invoke';call.data.inLabels=['instance',...call.data.inLabels];
      const shifted={input_1:{connections:[{node:String(instance),input:'output_1'}]}};
      Object.entries(call.inputs).forEach(([p,v])=>{shifted['input_'+(Number(p.slice(6))+1)]=v;});
      call.inputs=shifted;
      for(const up of Object.values(df)) for(const output of Object.values(up.outputs)) for(const c of output.connections) if(String(c.node)===id) c.output='input_'+(Number(c.output.slice(6))+1);
      df[instance].outputs.output_1.connections=[{node:id,output:'input_1'}];
      ED.importData('main',df);
    }
    App.renderFnPalette();App.scheduleSync();
  };
}
