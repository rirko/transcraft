const fs=require('fs'),vm=require('vm'),assert=require('assert'),cp=require('child_process'),path=require('path');
const context=vm.createContext({console,window:{addEventListener(){}},setTimeout,clearTimeout});
for(const file of ['editor.js','app.js','architecture.js']) vm.runInContext(fs.readFileSync(path.join(__dirname,'static',file),'utf8'),context);
vm.runInContext(`ED.exportData=key=>fixtures[key]||{};App.fnSignature=name=>{const ns=Object.values(fixtures['fn:'+name]||{});return {inputs:ns.filter(n=>n.data.type==='fn_input').map(n=>n.data.argname),outputs:ns.filter(n=>n.data.type==='fn_output').flatMap(n=>Object.keys(n.inputs))};};`,context);
function node(id,type,data={},inputs=[],outputs=1){return {id,data:{type,name:type,...data},inputs:Object.fromEntries(inputs.map((c,i)=>['input_'+(i+1),{connections:c?[{node:String(c[0]),input:'output_'+(c[1]||1)}]:[]}])) ,outputs:Object.fromEntries(Array.from({length:outputs},(_,i)=>['output_'+(i+1),{connections:[]}]))};}
function compile(fixtures,defs){context.fixtures=fixtures;context.defs=defs;return vm.runInContext(`App.state.fnOrder=Object.keys(defs);App.state.definitions=defs;({code:App.generatePython(),errors:App.lastErrors})`,context);}
const model={1:node(1,'fn_input',{argname:'ids'}),2:node(2,'embedding',{name:'embedding_matrix',vocab_size:'vocab_size',embedding_dim:'embedding_dim'},[[1]]),3:node(3,'fn_output',{},[[2],[2]],0)};
const main={1:node(1,'tensor',{values:'[1,2]'}),2:node(2,'create',{fname:'ShankeModel',init_args:'8, 4',name:'model'}),3:node(3,'invoke',{fname:'ShankeModel',name:'a',inLabels:['instance','ids'],outLabels:['x','y']},[[2],[1]],2),4:node(4,'invoke',{fname:'ShankeModel',name:'b',inLabels:['instance','ids'],outLabels:['x','y']},[[2],[1]],2)};
const defs={ShankeModel:{kind:'model',params:'vocab_size, embedding_dim'}},fixture={'main':main,'fn:ShankeModel':model};
let result=compile(fixture,defs);assert.deepEqual(result.errors,[]);
assert.equal((result.code.match(/self.embedding_matrix\(ids\)/g)||[]).length,1);
assert.equal((result.code.match(/= ShankeModel\(8, 4\)/g)||[]).length,1);
const python=path.join(__dirname,'../.venv/Scripts/python.exe');
function run(code,checks){const r=cp.spawnSync(python,['-c',code+'\n'+checks],{encoding:'utf8',timeout:60000});assert.equal(r.status,0,r.stderr);}
run(result.code,'assert torch.equal(a_out, b_out)\nassert a_out is a_out2\nassert "embedding_matrix.weight" in model.state_dict()');
assert.equal(compile(JSON.parse(JSON.stringify(fixture)),defs).code,result.code);
const fn={1:node(1,'fn_input',{argname:'x'}),2:node(2,'binop',{op:'+'},[[1],[1]]),3:node(3,'fn_output',{},[[2]],0)};
result=compile({'main':{},'fn:twice':fn},{twice:{kind:'function',params:''}});
assert.deepEqual(result.errors,[]);run(result.code,'assert twice(torch.tensor(3)).item()==6');
const nested={1:node(1,'fn_input',{argname:'x'}),2:node(2,'call',{fname:'ShankeModel',init_args:'8, 4',inLabels:['ids'],outLabels:['a','b']},[[1]],2),3:node(3,'fn_output',{},[[2,1],[2,2]],0)};
result=compile({'main':{},'fn:Outer':nested,'fn:ShankeModel':model},{Outer:{kind:'model',params:''},...defs});
assert.deepEqual(result.errors,[]);run(result.code,'m=Outer()\na,b=m(torch.tensor([1]))\nassert a is b\nassert len(list(m.parameters()))==1\np=next(m.parameters())\nm(torch.tensor([2]))\nassert next(m.parameters()) is p');
const broken=JSON.parse(JSON.stringify(fixture));broken.main[3].inputs.input_2.connections=[];
assert(compile(broken,defs).errors.length>0);
const attention={1:node(1,'fn_input',{argname:'x'}),2:node(2,'linear',{name:'q',in_features:4,out_features:4,bias:false},[[1]]),3:node(3,'transpose',{dim0:-2,dim1:-1},[[2]]),4:node(4,'matmul',{},[[2],[3]]),5:node(5,'softmax',{dim:-1},[[4]]),6:node(6,'matmul',{name:'out'},[[5],[2]]),7:node(7,'fn_output',{},[[6],[5]],0)};
result=compile({'main':{},'fn:SelfAttention':attention},{SelfAttention:{kind:'model',params:''}});
assert.deepEqual(result.errors,[]);run(result.code,'m=SelfAttention()\na,w=m(torch.randn(3,4))\nassert a.shape==(3,4)\nassert w.shape==(3,3)\nassert torch.allclose(w.sum(-1),torch.ones(3))');
console.log('PASS: constructor, shared invocation, nested multi-output lifecycle, function, restore, invalid graph, attention');
const invalidCreate={main:{1:node(1,'create',{fname:'Missing',init_args:''})}};
assert(compile(invalidCreate,{}).errors.some(x=>x.includes('无效模型创建目标')));
const wrongInstance=JSON.parse(JSON.stringify(fixture));
wrongInstance.main[5]=node(5,'binop',{op:'+'},[[2],[1]]);
assert(compile(wrongInstance,defs).errors.some(x=>x.includes('模型实例只能连接调用实例端口')));
console.log('PASS: invalid create target and Module-to-binop rejection');

// Only the backend-validated expression is compiled, never graph-supplied cache data.
const values=['42','-1.25','"hello"','True','False','None','[1, "x", None]','(1, True)'];
for (const source of values) {
  const checked=cp.spawnSync(python,['-c',`import sys,json;sys.path.insert(0,sys.argv[1]);from server import validate_literal;print(json.dumps(validate_literal(sys.argv[2])))`,__dirname,source],{encoding:'utf8'});
  assert.equal(checked.status,0,checked.stderr);
  context.source=source;context.literal=JSON.parse(checked.stdout);
  vm.runInContext('App.literalCache.set(source,literal)',context);
  const graph={main:{1:node(1,'literal',{name:'value',value:source}),2:node(2,'print',{},[[1]],0)}};
  result=compile(graph,{});assert.deepEqual(result.errors,[]);
  run(result.code,`assert value == ${source}\nassert type(value) is type(${source})`);
  assert.equal(compile(JSON.parse(JSON.stringify(graph)),{}).code,result.code);
}
assert.throws(()=>compile({main:{1:node(1,'literal',{value:'__import__("os")',expression:'0'})}},{}),/尚未校验/);
const literalFn={1:node(1,'literal',{value:'42'}),2:node(2,'fn_output',{},[[1]],0)};
for (const kind of ['function','model']) {
  result=compile({main:{},'fn:Constant':literalFn},{Constant:{kind,params:''}});
  assert.deepEqual(result.errors,[]);run(result.code,kind==='function'?'assert Constant()==42':'assert Constant()()==42');
}
const stale=JSON.parse(JSON.stringify(fixture));
stale.main[3].inputs.input_3={connections:[{node:'1',input:'output_1'}]};
const before=JSON.stringify(stale);
assert(compile(stale,defs).errors.some(x=>x.includes('接口已改变')));
assert.equal(JSON.stringify(stale),before);
context.staleNode=stale.main[3];
vm.runInContext('ED.hooks.fnMeta=name=>App.fnSignature(name);ED.getNode=(key,id)=>fixtures[key]?.[id]',context);
assert.equal(vm.runInContext("ED.portInfo('main',staleNode,'input',1).name",context),'ids');
assert.equal(vm.runInContext("ED.portInfo('main',staleNode,'input',2).obsolete",context),true);
assert(!vm.runInContext("ED.nodeHTML({data:{name:'<img onerror=alert(1)>'}})",context).includes('<img'));
console.log('PASS: literal types, safe compilation, class/function scope, restore, obsolete named ports, escaped template');

function mapped(result,key,id) {
  const map=vm.runInContext('App.codeSourceMap',context);
  return Array.from(map[key][id],line=>result.code.split('\n')[line-1].trim());
}
result=compile(fixture,defs);
assert.deepEqual(mapped(result,'fn:ShankeModel',1),['def forward(self, ids):']);
assert.deepEqual(mapped(result,'fn:ShankeModel',2),['self.embedding_matrix = nn.Embedding(vocab_size, embedding_dim)','embedding_matrix = self.embedding_matrix(ids)']);
assert.deepEqual(mapped(result,'fn:ShankeModel',3),['return embedding_matrix, embedding_matrix']);
assert.deepEqual(mapped(result,'main',1),['tensor = torch.tensor([1,2])']);
assert.deepEqual(mapped(result,'main',2),['model = ShankeModel(8, 4)']);
assert.deepEqual(mapped(result,'main',3),['a_out, a_out2 = model(tensor)']);
const legacyMain={1:main[1],2:node(2,'call',{name:'legacy',fname:'ShankeModel',init_args:'8, 4',inLabels:['ids'],outLabels:['a','b']},[[1]],2)};
result=compile({main:legacyMain,'fn:ShankeModel':model},defs);
assert.deepEqual(mapped(result,'main',2),['legacy_instance = ShankeModel(8, 4)','legacy_out, legacy_out2 = legacy_instance(tensor)']);
result=compile({main:{},'fn:Outer':nested,'fn:ShankeModel':model},{Outer:{kind:'model',params:''},...defs});
assert.deepEqual(mapped(result,'fn:Outer',2),['self.call = ShankeModel(8, 4)','call_out, call_out2 = self.call(x)']);
const explicit={1:nested[1],2:node(2,'create',{name:'child',fname:'ShankeModel',init_args:'8, 4'}),3:node(3,'invoke',{name:'r',fname:'ShankeModel',inLabels:['instance','ids'],outLabels:['a','b']},[[2],[1]],2),4:node(4,'fn_output',{},[[3,1],[3,2]],0)};
result=compile({main:{},'fn:Outer':explicit,'fn:ShankeModel':model},{Outer:{kind:'model',params:''},...defs});
assert.deepEqual(mapped(result,'fn:Outer',2),['self.child = ShankeModel(8, 4)']);
assert.deepEqual(mapped(result,'fn:Outer',3),['r_out, r_out2 = self.child(x)']);
result=compile({main:{1:main[1],2:node(2,'function_call',{fname:'twice',inLabels:['x'],outLabels:['r']},[[1]])},'fn:twice':fn},{twice:{kind:'function',params:''}});
assert.deepEqual(mapped(result,'fn:twice',1),['def twice(x):']);
assert.deepEqual(mapped(result,'fn:twice',2),['binop = x + x']);
assert.deepEqual(mapped(result,'fn:twice',3),['return binop']);
assert.deepEqual(mapped(result,'main',2),['function_call_out = twice(tensor)']);
result=compile({main:{},'fn:SelfAttention':attention},{SelfAttention:{kind:'model',params:''}});
assert.deepEqual(mapped(result,'fn:SelfAttention',2),['self.q = nn.Linear(4, 4, bias=False)','q = self.q(x)']);
result=compile(broken,defs);assert.deepEqual(mapped(result,'main',3),[]);
const multiline={main:{1:node(1,'tensor',{values:'[1,\n2]'}),2:node(2,'print',{},[[1]],0)}};
result=compile(multiline,{});
assert.deepEqual(mapped(result,'main',1),['tensor = torch.tensor([1,','2])']);
assert.deepEqual(mapped(result,'main',2),['print(tensor)']);
assert.equal(result.code, 'import torch\nimport torch.nn as nn\n\ntorch.manual_seed(1234)\n\nif __name__ == "__main__":\n    tensor = torch.tensor([1,\n2])\n    print(tensor)\n');
console.log('PASS: scoped source maps, init/forward, signatures, returns, create/invoke, legacy calls, multi-output, physical lines');

// Minimal DOM stand-in: throw on HTML parsing and exercise the real Prism grammar.
let scrolls=0;
function element() {
  return {dataset:{},className:'',children:[],scrollTop:37,scrollLeft:11,
    get textContent(){return this.children.map(child=>child.textContent).join('');},
    set textContent(text){this.children=[{textContent:text}];},
    set innerHTML(value){throw Error('Preview must not parse HTML');},
    classList:{toggle(name,on){this[name]=on;}},
    appendChild(child){this.children.push(child);},
    replaceChildren(fragment){this.children=fragment.children;},
    querySelectorAll(){return this.children;},
    scrollIntoView(){scrolls++;}};
}
const view=element();
// Load in manual mode before installing the minimal document, just as index.html does.
context.window.Prism={manual:true};
for (const file of ['prism-core-1.30.0.min.js','prism-python-1.30.0.min.js']) {
  vm.runInContext(fs.readFileSync(path.join(__dirname,'static/vendor',file),'utf8'),context);
}
context.document={querySelector:()=>view,createElement:tag=>{assert.equal(tag,'span');return element();},
  createTextNode:text=>({textContent:text}),createDocumentFragment:()=>element()};
result=compile(fixture,defs);context.previewCode=result.code;
vm.runInContext("App.selected={key:'fn:ShankeModel',id:2};App.renderCodePreview(previewCode)",context);
assert.equal(view.children.map(x=>x.textContent).join(''),result.code);
assert.equal(view.children.filter(x=>x.classList['is-selected']).length,2);
assert.equal(scrolls,0);
vm.runInContext('App.highlightCodeSelection(true)',context);assert.equal(scrolls,1);
const renamed=JSON.parse(JSON.stringify(fixture));renamed['fn:ShankeModel'][2].data.name='renamed_layer';
result=compile(renamed,defs);context.previewCode=result.code;
vm.runInContext('App.renderCodePreview(previewCode)',context);
assert(view.children.filter(x=>x.classList['is-selected']).every(x=>x.textContent.includes('renamed_layer')));
assert.equal(scrolls,1);assert.equal(view.scrollTop,37);assert.equal(view.scrollLeft,11);
vm.runInContext("App.selected={key:'main',id:'2'};App.highlightCodeSelection(true)",context);
assert.equal(view.children.filter(x=>x.classList['is-selected']).length,1);
assert.equal(scrolls,2);
vm.runInContext('App.selected=null;App.highlightCodeSelection()',context);
assert.equal(view.children.filter(x=>x.classList['is-selected']).length,0);
vm.runInContext("App.renderCodePreview('<img src=x onerror=alert(1)>\\n')",context);
assert.equal(view.children[0].textContent,'<img src=x onerror=alert(1)>\n');
assert.equal(scrolls,2);
console.log('PASS: safe preview text, scoped selection, rename persistence, deselection, sync scroll preservation');

function tokenText(root,type) {
  return (root.children||[]).flatMap(child=>
    (child.className||'').split(' ').includes(type) ? [child.textContent] : tokenText(child,type)).join('');
}
const samples=[
  '', '\n', '\n\n', 'pass', 'pass\n\n', '\tvalue = 1\r\n# comment\r\n',
  'message = """first\n# not a comment\n<img onerror=alert(1)> & last"""\n# real comment\n',
  "message = '''first\nsecond'''\n",
  'message = f"""first {1 + 2}\nsecond {3!r}"""\n',
  'value = "</span><script>alert(1)</script>&\\\""\n# <img src=x onerror=alert(1)>\n',
  'value = "\u00a0\u4e2d\u6587"\n'
];
for (const source of samples) {
  context.previewCode=source;
  vm.runInContext('App.renderCodePreview(previewCode)',context);
  assert.equal(view.textContent,source);
  const lines=source.split('\n');
  if (lines[lines.length-1]==='') lines.pop();
  assert.equal(view.children.length,lines.length);
  view.children.forEach((line,i)=>{
    assert.equal(line.dataset.line,String(i+1));
    assert.equal(line.textContent,lines[i]+(i<lines.length-1||source.endsWith('\n')?'\n':''));
  });
  assert.equal(view.scrollTop,37);assert.equal(view.scrollLeft,11);
}
context.previewCode=samples[6];
vm.runInContext("App.codeSourceMap={main:{9:[2,3]}};App.selected={key:'main',id:9};App.renderCodePreview(previewCode)",context);
assert.equal(tokenText(view.children[1],'string'),'# not a comment\n');
assert.equal(tokenText(view.children[1],'comment'),'');
assert.equal(tokenText(view.children[3],'comment'),'# real comment');
assert.deepEqual(view.children.filter(x=>x.classList['is-selected']).map(x=>x.dataset.line),['2','3']);
context.previewCode=samples[8];
vm.runInContext('App.renderCodePreview(previewCode)',context);
assert.equal(tokenText(view,'number'),'123');
assert(tokenText(view.children[1],'string').includes('second '));
vm.runInContext('window.Prism=null;App.renderCodePreview(previewCode)',context);
assert.equal(view.textContent,context.previewCode);
assert.equal(scrolls,2);
console.log('PASS: Prism multiline strings, comments, nested f-strings, exact physical lines, unsafe text, Unicode, selection, plain-text fallback');

// Isolated DOM contract tests: no server, browser storage or live project writes.
class UIElement {
  constructor(tag='div') {
    this.tagName=tag.toUpperCase();this.children=[];this.dataset={};this.attributes={};this.events={};this.hidden=false;
    this.scrollTop=0;this.scrollLeft=0;this.style={};this.parts=null;
    const self=this;
    this.classList={add(n){self.className=(self.className?self.className+' ':'')+n;},
      remove(n){self.className=(self.className||'').split(' ').filter(c=>c&&c!==n).join(' ');},
      toggle(){}};
  }
  set textContent(value){this.text=String(value);this.children=[];}
  get textContent(){return (this.text||'')+this.children.map(x=>x.textContent).join('');}
  appendChild(child){this.children.push(child);return child;}
  prepend(child){this.children.unshift(child);}
  replaceChildren(...nodes){this.children=[...nodes];this.text='';}
  setAttribute(key,value){this.attributes[key]=String(value);}
  addEventListener(name,callback){this.events[name]=callback;}
  focus(){} select(){}
  querySelectorAll(selector){
    if(selector==='[data-pkey]')return this.controls||[];
    return this.children.flatMap(child=>[
      ...((selector.startsWith('.')&&(child.className||'').split(' ').includes(selector.slice(1)))?[child]:[]),
      ...child.querySelectorAll?.(selector)||[]]);
  }
  querySelector(selector){
    if(this.parts&&Object.prototype.hasOwnProperty.call(this.parts,selector))return this.parts[selector];
    if(selector.startsWith('.')&&!selector.includes(' '))return deepFind(this,selector);
    return null;
  }
  set innerHTML(html){
    this.html=html;this.controls=[];this.parts={};
    for(const match of html.matchAll(/<(input|select)\b([^>]*data-pkey="[^"]+"[^>]*)>/g)) {
      const control=new UIElement(match[1]),attrs=match[2];
      control.dataset.pkey=attrs.match(/data-pkey="([^"]+)"/)[1];
      control.type=attrs.match(/type="([^"]+)"/)?.[1]||'select';
      control.value=attrs.match(/value="([^"]*)"/)?.[1]||'';control.checked=/\bchecked\b/.test(attrs);
      this.controls.push(control);
    }
    for(const match of html.matchAll(/<button[^>]*data-ins-action="([^"]+)"/g)) this.parts['[data-ins-action="'+match[1]+'"]']=new UIElement('button');
  }
}
function deepFind(root,selector){
  for(const child of root.children||[]) {
    if((child.className||'').split(' ').includes(selector.slice(1)))return child;
    const found=deepFind(child,selector);if(found)return found;
  }
  return null;
}
const ui=vm.createContext({console,window:{addEventListener(){}},setTimeout,clearTimeout});
for(const file of ['editor.js','app.js','architecture.js','tensor-viewer.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'static',file),'utf8'),ui);
const basic=new UIElement(),fnNodes=new UIElement();
const docListeners={};
ui.document={createElement:tag=>new UIElement(tag),
  querySelector:selector=>({'#palette-basic':basic,'#palette-fnnodes':fnNodes}[selector]),
  addEventListener:(name,callback,capture)=>{(docListeners[name+(capture?':capture':'')]||=[]).push(callback);},
  removeEventListener:(name,callback,capture)=>{const key=name+(capture?':capture':'');docListeners[key]=(docListeners[key]||[]).filter(c=>c!==callback);}};
function uiDocKey(event,mock){for(const callback of [...(docListeners[event+':capture']||[])])callback(mock);}
vm.runInContext('App.buildPalette()',ui);
assert.deepEqual(basic.children.map(g=>g.children[0].textContent),['数据 / 输入','模型层','数学运算','张量变换','遮罩','输出']);
assert(basic.children.every(g=>g.tagName==='DETAILS'&&g.open));
const paletteTypes=basic.children.flatMap(g=>g.children.slice(1).map(n=>n.dataset.type));
assert.equal(new Set(paletteTypes).size,16);assert(!paletteTypes.includes('create'));
assert(paletteTypes.includes('variable_ref'));
const index=fs.readFileSync(path.join(__dirname,'static/index.html'),'utf8');
const sidebar=index.match(/<aside id="right">([\s\S]*?)<\/aside>/)[1];
assert(sidebar.includes('id="code-panel"'));assert(!/inspector|tensor-panel|tensor-view/.test(sidebar));
const template=vm.runInContext('ED.nodeHTML()',ui);
assert(template.includes('aria-expanded="false"'));assert(template.includes('aria-label="展开运行结果"'));
assert(template.includes('class="nd-drawer" hidden'));
assert(template.indexOf('class="nd-params"')<template.indexOf('class="nd-drawer"'));   // 参数常显在运行结果抽屉之外
assert(template.indexOf('class="nd-head"')<template.indexOf('class="nd-callable"'));   // 黄色功能标题位于节点名标题框正下方（head 之后）
assert(!template.includes('data-node-tab')&&!template.includes('nd-tabs'));            // 不再有参数/运行结果页签
const nodeElements={};
function drawerNode(id,key='main') {
  const n=new UIElement();n.parts={};
  for(const part of ['nd-title','nd-callable','nd-arguments','nd-pins','nd-warning',
    'nd-toggle','nd-drawer','nd-params','nd-runtime','nd-runtime-view','nd-runtime-status']) {
    const el=new UIElement();el.className=part;n.parts['.'+part]=el;
  }
  n.parts['.nd-drawer'].hidden=true;
  nodeElements[key+':'+id]=n;return n;
}
const first=drawerNode(1),second=drawerNode(2);
ui.fixtures={main:{1:node(1,'tensor',{values:'[1,2]'}),2:node(2,'print',{},[[1]],0)}};
ui.nodeElements=nodeElements;
vm.runInContext(`
ED.canvases={main:{container:{querySelector:s=>nodeElements['main:'+s.match(/node-(\\d+)/)[1]],querySelectorAll:()=>[]},editor:{}}};
ED.getNode=(key,id)=>fixtures[key]?.[id];ED.exportData=key=>fixtures[key]||{};
ED.redraw=()=>{};ED.updateNode=(key,id,patch)=>Object.assign(fixtures[key][id].data,patch);
ED.hooks.onFormMount=(key,id,box)=>App.renderInspectorForm(key,id,box);
ED.hooks.onFormSync=(key,id,box)=>App.syncInspectorForm(key,id,box);
ED.hooks.onNameEdit=(key,id)=>App.editNodeName(key,id);
App.state.fnOrder=[];App.state.definitions={};App.highlightCodeSelection=()=>{};
App.scheduleSync=()=>{App.edits=(App.edits||0)+1;App.editRevision++;};
App.project={id:'offline',revision:'r'};App.editRevision=0;`,ui);
// 常显参数：新增/导入路径（refreshSummary）即挂载表单，无需点开任何抽屉。
vm.runInContext("ED.refreshSummary('main',1)",ui);
const form=first.parts['.nd-params'];
assert.equal(form.dataset.mounted,'true');
assert(form.controls.some(x=>x.dataset.pkey==='values'));
assert(!form.html.includes('ins-title'));                     // 黄色功能标题只在 nd-callable 一份，参数体不重复
assert(!form.controls.some(x=>x.dataset.pkey==='__name'));    // 节点名由标题编辑，不在表单
assert(!form.controls.some(x=>x.dataset.pkey==='argname'));   // 参数名同理
assert.equal(first.parts['.nd-title'].textContent,'tensor');
assert.equal(first.parts['.nd-callable'].textContent,'Tensor 输入');   // 黄色功能标题 = NODE_DEFS.label
assert(!form.html.includes('Tensor 输入'));                            // 标题仅一份，不在参数体重复
assert(!form.html.includes('data-ins-action="delete"'));               // 底部“删除该节点”按钮已移除
assert(!form.querySelector('[data-ins-action="delete"]'));
const valuesField=form.controls.find(x=>x.dataset.pkey==='values');
vm.runInContext("ED.refreshSummary('main',1)",ui);            // 重复刷新不重复挂载、控件不换实例
assert.strictEqual(first.parts['.nd-params'],form);
assert.strictEqual(form.controls.find(x=>x.dataset.pkey==='values'),valuesField);
// 标题即节点名：Enter/失焦提交、Esc 取消、空值不提交、同步刷新不打断编辑。
vm.runInContext("App.editNodeName('main',1)",ui);
const nameInput=first.parts['.nd-title'].children[0];
assert.equal(nameInput.className,'nd-name-input');
assert.equal(nameInput.name,'name');
assert.equal(nameInput.value,'tensor');
vm.runInContext("ED.refreshSummary('main',1)",ui);
assert.strictEqual(first.parts['.nd-title'].children[0],nameInput);   // 同步刷新不丢编辑状态
vm.runInContext("App.editNodeName('main',1)",ui);
assert.strictEqual(first.parts['.nd-title'].children[0],nameInput);   // 编辑中重入被忽略
nameInput.value='renamed_tensor';
uiDocKey('keydown',{target:nameInput,key:'Enter',preventDefault(){}});
assert.equal(ui.fixtures.main[1].data.name,'renamed_tensor');
vm.runInContext("ED.refreshSummary('main',1)",ui);
assert.equal(first.parts['.nd-title'].textContent,'renamed_tensor');
assert.equal(first.parts['.nd-title'].children.length,0);
vm.runInContext("App.editNodeName('main',1)",ui);
const escInput=first.parts['.nd-title'].children[0];
escInput.value='discard_me';
uiDocKey('keydown',{target:escInput,key:'Escape',preventDefault(){}});
assert.equal(ui.fixtures.main[1].data.name,'renamed_tensor');          // Esc 取消
vm.runInContext("ED.refreshSummary('main',1)",ui);
assert.equal(first.parts['.nd-title'].textContent,'renamed_tensor');
vm.runInContext("App.editNodeName('main',1)",ui);
const blankInput=first.parts['.nd-title'].children[0];
blankInput.value='   ';
blankInput.events.blur();                                              // 失焦提交空值 → 非空校验拒绝
assert.equal(ui.fixtures.main[1].data.name,'renamed_tensor');
assert.equal(first.parts['.nd-title'].textContent,'renamed_tensor');
vm.runInContext("App.editNodeName('main',1)",ui);
const blurInput=first.parts['.nd-title'].children[0];
blurInput.value='renamed_by_blur';
blurInput.events.blur();                                               // 失焦提交
assert.equal(ui.fixtures.main[1].data.name,'renamed_by_blur');
// 模型输入节点的标题就是 forward 参数名：编辑写入 argname，表单不再有参数名输入。
ui.fixtures['fn:Args']={1:node(1,'fn_input',{argname:'x'})};
const argCard=drawerNode(1,'fn:Args');
vm.runInContext(`ED.canvases['fn:Args']={container:{querySelector:s=>nodeElements['fn:Args:'+s.match(/node-(\\d+)/)[1]],querySelectorAll:()=>[]},editor:{}};
ED.refreshSummary('fn:Args',1);`,ui);
assert.equal(argCard.parts['.nd-title'].textContent,'x');
assert(!argCard.parts['.nd-params'].controls.some(x=>x.dataset.pkey==='argname'));
vm.runInContext("App.editNodeName('fn:Args',1)",ui);
const argInput=argCard.parts['.nd-title'].children[0];
assert.equal(argInput.name,'argname');
argInput.value='ids';
uiDocKey('keydown',{target:argInput,key:'Enter',preventDefault(){}});
assert.equal(ui.fixtures['fn:Args'][1].data.argname,'ids');
valuesField.value='[3,4]';valuesField.events.input();
const edited=vm.runInContext('App.generatePython()',ui);
assert(edited.includes('renamed_by_blur = torch.tensor([3,4])'));assert(edited.includes('print(renamed_by_blur)'));
run(edited,'assert renamed_by_blur.tolist()==[3,4]');
vm.runInContext('App.hideInspector()',ui);
assert.equal(vm.runInContext('App.selected',ui),null);
assert.strictEqual(first.parts['.nd-params'],form);              // 取消选中保留常显表单
// 标题箭头只负责展开/收起运行结果；关闭结果不清参数；多个结果可同时展开。
vm.runInContext("App.showInspector('main',1)",ui);
assert.equal(first.parts['.nd-toggle'].attributes['aria-expanded'],'true');
assert.equal(first.parts['.nd-drawer'].hidden,false);
assert.equal(first.querySelector('.nd-runtime-status').textContent,'尚未运行');
assert(first.parts['.nd-runtime-status'].className.split(' ').includes('nd-runtime-status'));
vm.runInContext("App.toggleNodeDrawer('main',1)",ui);
assert.equal(first.parts['.nd-drawer'].hidden,true);
assert.equal(first.parts['.nd-toggle'].attributes['aria-expanded'],'false');
assert.strictEqual(first.parts['.nd-params'],form);              // 关闭结果不清参数
assert.strictEqual(form.controls.find(x=>x.dataset.pkey==='values'),valuesField);
vm.runInContext("App.showInspector('main',1);App.showInspector('main',2)",ui);
assert.equal(first.parts['.nd-drawer'].hidden,false);
assert.equal(second.parts['.nd-drawer'].hidden,false);
vm.runInContext("App.toggleNodeDrawer('main',2)",ui);
assert.equal(second.parts['.nd-toggle'].attributes['aria-expanded'],'false');
assert.equal(first.parts['.nd-drawer'].hidden,false);
// Preserve help and select/checkbox/int controls in the shared form renderer.
for(const type of ['ones_like','dtype','triu','masked_fill','linear','create','fn_input','fn_output']) {
  ui.fixtures.main[1]=node(1,type,{dtype:'bool',device:'inherit',value:'-inf',diagonal:1,bias:true,argname:'x',fname:'Model'});
  vm.runInContext("App.renderInspectorForm('main',1,nodeElements['main:1'].querySelector('.nd-params'))",ui);
  if(['ones_like','dtype','triu','masked_fill'].includes(type))assert(form.html.includes('param-help'));
  if(type==='linear')assert(form.controls.some(c=>c.type==='checkbox'&&c.checked));
  if(type==='fn_output')assert(form.querySelector('[data-ins-action="add-ret"]'));
}
// 参数为空的节点（matmul）不留空白表单容器；有参数的节点容器保持显示且无删除按钮。
ui.fixtures.main[1]=node(1,'matmul',{});
vm.runInContext("App.renderInspectorForm('main',1,nodeElements['main:1'].querySelector('.nd-params'))",ui);
assert.equal(form.hidden,true);
assert(!form.html.includes('删除该节点'));assert(!form.querySelector('[data-ins-action="delete"]'));
ui.fixtures.main[1]=node(1,'tensor',{values:'[1,2]'});
vm.runInContext("App.renderInspectorForm('main',1,nodeElements['main:1'].querySelector('.nd-params'))",ui);
assert.equal(form.hidden,false);
assert(form.controls.some(x=>x.dataset.pkey==='values'));
console.log('PASS: grouped accessible palette, code-only sidebar, always-on params mounted once, yellow def.label title under header, bottom delete button removed, empty params hide blank form, title edit commit/cancel/blank/argname, runtime toggles keep params, shared form help and edits compile');

ui.fixtures.main[1]=node(1,'tensor',{values:'[1]'});
ui.trace={project_id:'offline',revision:'r',editRevision:2,code:'a = tensor\nprint(a)',plan:[{scope:'main',node:'1',type:'tensor',sources:[{node:'2',output:1,name:'input'}],lines:[1]}],records:[{scope:'main',node:'1',context:[],outputs:[{kind:'tensor',shape:[2,2,2,3],preview_shape:[2,2,2,3],dtype:'float32',device:'cpu',numel:24,captured:24,values:Array.from({length:24},(_,i)=>i)}]}]};
vm.runInContext("App.runtimeTrace=trace;App.editRevision=trace.editRevision;App.showInspector('main',1)",ui);
function findUI(root,predicate){if(predicate(root))return root;for(const child of root.children||[]){const found=findUI(child,predicate);if(found)return found;}}
const runtime=first.parts['.nd-runtime-view'];
assert.equal(first.querySelector('.nd-runtime-status').textContent,'本次运行');
assert(first.querySelector('.nd-runtime-status'));
const matrix=findUI(runtime,n=>n.dataset.tensorPart==='matrix');assert(matrix);assert.equal(matrix.children[1].children[1].textContent,'0');
const axis=findUI(runtime,n=>n.dataset.axis===1);axis.value='1';axis.oninput();
assert.equal(findUI(runtime,n=>n.dataset.tensorPart==='matrix').children[1].children[1].textContent,'6');
vm.runInContext('App.editRevision++;App.renderTensorViewer()',ui);
assert.equal(first.parts['.nd-runtime-status'].textContent,'已过期');
assert.equal(findUI(runtime,n=>n.dataset.sourceNode==='2').disabled,true);
vm.runInContext("App.editRevision--;App.renderTensorViewer();App.switchTab=()=>{};",ui);
findUI(runtime,n=>n.dataset.sourceNode==='2').onclick();
assert.equal(vm.runInContext("App.nodeDrawer('main',2).id",ui),'2');
assert.equal(first.parts['.nd-drawer'].hidden,false);
console.log('PASS: drawer runtime tensor matrix, high-dimensional slices, expired provenance guard and upstream drawer navigation');

// Identical Drawflow node IDs in separate scopes must never share form or runtime state.
const scoped=drawerNode(1,'fn:Scoped');
ui.fixtures['fn:Scoped']={1:node(1,'tensor',{name:'scoped',values:'[9]'})};
ui.scoped=scoped;
vm.runInContext(`ED.canvases['fn:Scoped']={container:{querySelector:s=>nodeElements['fn:Scoped:'+s.match(/node-(\\d+)/)[1]],querySelectorAll:()=>[]},editor:{}};
App.closeNodeDrawer('main',1);ED.refreshSummary('fn:Scoped',1);
App.showInspector('main',1);App.showInspector('fn:Scoped',1);`,ui);
assert.equal(first.parts['.nd-drawer'].hidden,false);assert.equal(scoped.parts['.nd-drawer'].hidden,false);
assert.notEqual(first.parts['.nd-drawer'].id,scoped.parts['.nd-drawer'].id);
const scopedForm=scoped.parts['.nd-params'];
const scopedValues=scopedForm.controls.find(x=>x.dataset.pkey==='values');
vm.runInContext("App.selectNode('main',2)",ui);
scopedValues.value='[10,11]';scopedValues.events.input();
assert.equal(ui.fixtures['fn:Scoped'][1].data.values,'[10,11]');assert.equal(ui.fixtures.main[1].data.values,'[1]');
assert.equal(vm.runInContext('App.selected.key',ui),'fn:Scoped');
vm.runInContext("App.showInspector('fn:Scoped',1)",ui);
assert.strictEqual(scopedForm.controls.find(x=>x.dataset.pkey==='values'),scopedValues);
assert(!scopedForm.html.includes('id="ins-'));
const tensor=ui.trace.records[0].outputs[0];
ui.trace.records=[
  {scope:'main',node:'1',context:[],outputs:[tensor,tensor]},
  {scope:'main',node:'1',context:[],outputs:[tensor,tensor]},
  {scope:'fn:Scoped',node:'1',context:[],outputs:[{...tensor,values:tensor.values.map(v=>v+100)}]},
];
ui.trace.plan.push({scope:'fn:Scoped',node:'1',type:'tensor',sources:[],lines:[1]});
vm.runInContext("App.editRevision=trace.editRevision;App.showInspector('fn:Scoped',1);App.renderTensorViewer()",ui);
const scopedRuntime=scoped.parts['.nd-runtime-view'];
function control(box,name){return findUI(box,n=>n.dataset.tensorControl==='tensor-'+name);}
function matrixStart(box){return findUI(box,n=>n.dataset.tensorPart==='matrix').children[1].children[1].textContent;}
assert.equal(matrixStart(scopedRuntime),'100');assert.equal(matrixStart(runtime),'6');
control(runtime,'invocation').value='1';control(runtime,'invocation').onchange();
control(runtime,'output').value='1';control(runtime,'output').onchange();
findUI(runtime,n=>n.dataset.axis===0).value='1';findUI(runtime,n=>n.dataset.axis===0).oninput();
assert.equal(matrixStart(runtime),'12');assert.equal(matrixStart(scopedRuntime),'100');
const otherAxis=findUI(scopedRuntime,n=>n.dataset.axis===1);otherAxis.value='1';otherAxis.oninput();
assert.equal(matrixStart(scopedRuntime),'106');assert.equal(matrixStart(runtime),'12');
control(runtime,'output').value='0';control(runtime,'output').onchange();
control(runtime,'output').value='1';control(runtime,'output').onchange();assert.equal(matrixStart(runtime),'12');
control(runtime,'invocation').value='0';control(runtime,'invocation').onchange();assert.equal(matrixStart(runtime),'6');
control(runtime,'invocation').value='1';control(runtime,'invocation').onchange();assert.equal(matrixStart(runtime),'12');
first.parts['.nd-drawer'].scrollTop=80;
findUI(runtime,n=>n.className==='tensor-table-wrap').scrollLeft=45;
vm.runInContext("App.runtimeTrace={...trace,revision:'next'};App.project.revision='next';App.renderTensorViewer()",ui);
assert.equal(matrixStart(runtime),'12');assert.equal(matrixStart(scopedRuntime),'106');
assert.equal(first.parts['.nd-drawer'].scrollTop,80);
assert.equal(findUI(runtime,n=>n.className==='tensor-table-wrap').scrollLeft,45);
assert(!findUI(runtime,n=>n.id&&n.id.startsWith('tensor-')));
vm.runInContext('App.editRevision++;App.renderTensorViewer()',ui);
assert.equal(first.parts['.nd-runtime-status'].textContent,'已过期');assert.equal(scoped.parts['.nd-runtime-status'].textContent,'已过期');
vm.runInContext("App.toggleNodeDrawer('main',1)",ui);
assert.equal(first.parts['.nd-drawer'].hidden,true);assert.equal(scoped.parts['.nd-drawer'].hidden,false);
vm.runInContext("App.showInspector('main',1);App.closeNodeDrawers('fn:Scoped')",ui);
assert.equal(scoped.parts['.nd-drawer'].hidden,true);assert.equal(first.parts['.nd-drawer'].hidden,false);
const outputMain=drawerNode(3),outputScoped=drawerNode(3,'fn:Scoped');
ui.fixtures.main[3]=node(3,'fn_output',{},[null],0);
ui.fixtures['fn:Scoped'][3]=node(3,'fn_output',{},[null],0);
ui.confirm=()=>true;
vm.runInContext(`ED.addInputPort=(key,id)=>{const n=fixtures[key][id];n.inputs['input_'+(Object.keys(n.inputs).length+1)]={connections:[]};};
ED.removeNode=(key,id)=>{delete fixtures[key][id];App.closeNodeDrawer(key,id);};
ED.refreshSummary('main',3);ED.refreshSummary('fn:Scoped',3);
App.showInspector('main',3);App.showInspector('fn:Scoped',3);App.selectNode('main',1);`,ui);
outputScoped.parts['.nd-params'].querySelector('[data-ins-action="add-ret"]').events.click();
assert.equal(Object.keys(ui.fixtures['fn:Scoped'][3].inputs).length,2);assert.equal(Object.keys(ui.fixtures.main[3].inputs).length,1);
assert.equal(vm.runInContext('App.selected.key',ui),'fn:Scoped');
// 底部“删除该节点”按钮已移除；节点删除仍由右键菜单的“删除节点”触发（ED.removeNode → 关闭对应抽屉，不影响其他作用域）。
assert(!outputScoped.parts['.nd-params'].querySelector('[data-ins-action="delete"]'));
assert(!outputScoped.parts['.nd-params'].html.includes('删除该节点'));
vm.runInContext("ED.removeNode('fn:Scoped',3)",ui);
assert.equal(ui.fixtures['fn:Scoped'][3],undefined);assert(ui.fixtures.main[3]);
assert.equal(outputScoped.parts['.nd-drawer'].hidden,true);assert.equal(outputMain.parts['.nd-drawer'].hidden,false);
vm.runInContext("App.showInspector('main',1);App.showInspector('fn:Scoped',1)",ui);
const stack=new UIElement(),mainContainer=new UIElement(),fnContainer=new UIElement();
ui.stack=stack;ui.mainContainer=mainContainer;ui.fnContainer=fnContainer;
ui.document.querySelector=selector=>selector==='#canvas-stack'?stack:null;
ui.document.querySelectorAll=selector=>selector==='.dflow-container'?[mainContainer,fnContainer]:[];
ui.document.getElementById=()=>null;ui.document.body=new UIElement();
vm.runInContext(`App.containerOf=key=>key==='main'?mainContainer:fnContainer;
App.state.definitions.Scoped={kind:'model',params:''};oldSwitch('fn:Scoped');`,ui);
assert.equal(mainContainer.style.display,'none');assert.equal(fnContainer.style.display,'block');
assert(vm.runInContext("App.nodeDrawer('main',1)&&App.nodeDrawer('fn:Scoped',1)",ui));
assert.equal(vm.runInContext('App.selected',ui),null);
vm.runInContext("App.renderModelPhase('fn:Scoped','init')",ui);
assert.equal(scoped.parts['.nd-drawer'].hidden,true);assert.equal(first.parts['.nd-drawer'].hidden,false);
assert.equal(fnContainer.style.display,'none');
vm.runInContext("App.renderModelPhase('fn:Scoped','forward');App.showInspector('fn:Scoped',1)",ui);
assert.equal(fnContainer.style.display,'block');
vm.runInContext(`ED.hooks.onReset=key=>App.closeNodeDrawers(key);
ED.canvases['fn:Scoped'].editor.clear=()=>{};ED.clearCanvas('fn:Scoped');`,ui);
assert.equal(scoped.parts['.nd-drawer'].hidden,true);assert.equal(first.parts['.nd-drawer'].hidden,false);
vm.runInContext("App.showInspector('fn:Scoped',1);App.state.fnOrder=['Scoped'];App.containerOf=()=>null;App.removeFunctionRaw('Scoped')",ui);
assert.equal(vm.runInContext("App.nodeDrawer('fn:Scoped',1)",ui),undefined);
assert(vm.runInContext("App.nodeDrawer('main',1)",ui));
vm.runInContext('ED.clearCanvas=()=>{};App.clearAllModels()',ui);
assert.equal(vm.runInContext('App.drawers.size',ui),0);assert.equal(vm.runInContext('Object.keys(App.tensorChoice).length',ui),0);
const css=fs.readFileSync(path.join(__dirname,'static/style.css'),'utf8');
assert(/drawflow-node\.drawer-open\s*\{[^}]*width:260px;[^}]*min-width:260px;/.test(css));
assert(/nd-drawer\s*\{[^}]*max-height:320px;[^}]*overflow:auto;/.test(css));
assert(/nd-params\s*\{[^}]*max-height:240px;[^}]*overflow:auto;/.test(css));
assert(!css.includes('nd-tabs'));assert(css.includes('nd-name-input'));
// 黄色功能标题：.nd-callable 沿用原 ins-title 的橙色（--acc）并加粗醒目；参数为空不留空白容器。
assert(/\.dflow-container \.nd-callable\s*\{[^}]*color:\s*var\(--acc\)/.test(css));
assert(/\.dflow-container \.nd-callable\s*\{[^}]*font-weight:\s*700/.test(css));
assert(/\.nd-params\[hidden\]\s*\{[^}]*display:\s*none/.test(css));
console.log('PASS: duplicate IDs across scopes, independent forms/sample/output/slices, refresh/expiry, scroll retention, scoped collapse/removal via context-menu delete and project reset, yellow callable title with acc color, fixed 260px width with params/runtime scrolling separately');

const listeners={};
const fakeContainer={addEventListener:(event,handler,capture)=>{(listeners[event]||=[]).push({handler,capture:!!capture});}};
ui.document.getElementById=()=>fakeContainer;
ui.Drawflow=class {constructor(){this.precanvas={};}start(){}on(){}};
ui.ResizeObserver=class {observe(){}};
vm.runInContext("ED.makeEditor('isolated','fake-canvas')",ui);
let intercepted=0;
for(const event of ['mousedown','touchstart','pointerdown','keydown','dblclick','contextmenu','wheel']) {
  const captures=listeners[event].filter(x=>x.capture===true);
  const interceptor=captures[event==='mousedown'?1:0];   // mousedown 的首个捕获监听是拖拽阈值记录器
  assert(interceptor);
  interceptor.handler({target:{closest:()=>true},stopPropagation:()=>intercepted++});
  interceptor.handler({target:{closest:()=>null},stopPropagation:()=>{throw Error('Node header must remain draggable');}});
}
assert.equal(intercepted,7);
// 标题点击进入编辑、拖拽后松开不触发；箭头切换运行结果；参数区聚焦选中节点；键盘可达。
const nameEdits=[],toggles=[],selectCalls=[];
ui.nameEdits=nameEdits;ui.toggles=toggles;ui.selectCalls=selectCalls;
vm.runInContext('ED.hooks.onNameEdit=(...args)=>nameEdits.push(args);ED.hooks.onDrawerToggle=(...args)=>toggles.push(args);ED.hooks.onSelect=(...args)=>selectCalls.push(args)',ui);
const recorder=listeners.mousedown.find(x=>x.capture===true).handler;
const click=listeners.click[0].handler;
const titleStub={closest:()=>({id:'node-1'})};
recorder({target:{closest:s=>s==='.drawflow-node'?titleStub:null},clientX:10,clientY:10});
click({target:{closest:s=>s==='.nd-title'?titleStub:null},clientX:10,clientY:10});
assert.deepEqual(Array.from(nameEdits[0]),['isolated','1']);
recorder({target:{closest:s=>s==='.drawflow-node'?titleStub:null},clientX:10,clientY:10});
click({target:{closest:s=>s==='.nd-title'?titleStub:null},clientX:60,clientY:10});
assert.equal(nameEdits.length,1);   // 拖拽距离超过阈值，视为拖动而非点击
click({target:{closest:s=>s==='.nd-toggle'?{closest:()=>({id:'node-2'})}:null}});
assert.deepEqual(Array.from(toggles[0]),['isolated','2']);
listeners.focusin[0].handler({target:{closest:()=>({closest:()=>({id:'node-2'})})}});
assert.deepEqual(Array.from(selectCalls[0]),['isolated','2']);
const titleKey=listeners.keydown.find(x=>!x.capture).handler;
titleKey({target:{closest:s=>s==='.nd-title'?{closest:()=>({id:'node-3'})}:null},key:'Enter',preventDefault(){}});
assert.deepEqual(Array.from(nameEdits[1]),['isolated','3']);
console.log('PASS: inline controls isolate drag, delete, touch and wheel events without blocking node header dragging; title click/keyboard edit entry with drag threshold; toggle and params selection');

// Real editor/header/clipboard functions against isolated in-memory graphs and DOM.
const clipboardUI=vm.createContext({console,setTimeout:()=>0,clearTimeout(){},
  requestAnimationFrame:callback=>{clipboardUI.frame=callback;return 1;},
  window:{innerWidth:320,innerHeight:240,events:{},
    addEventListener(name,callback){this.events[name]=callback;},removeEventListener(name){delete this.events[name];}}});
for(const file of ['editor.js','app.js','architecture.js','projects.js'])
  vm.runInContext(fs.readFileSync(path.join(__dirname,'static',file),'utf8'),clipboardUI);
const graphNodes={},graphElements={},docEvents={},body=new UIElement('body');
let activeElement=null;
function docFire(name,capture,event){for(const callback of [...(docEvents[name+(capture?':capture':'')]||[])])callback(event);}
function editableElement(tag='div') {
  const el=new UIElement(tag);
  el.focus=()=>{activeElement=el;};
  el.select=()=>{};
  el.remove=()=>{body.children=body.children.filter(child=>child!==el);};
  el.contains=target=>target===el||el.children.includes(target);
  el.getBoundingClientRect=()=>({width:132,height:72});
  return el;
}
function cardElement(id,key) {
  const card=editableElement();card.id='node-'+id;card.parts={};
  card.offsetHeight=200;card.clientTop=1;card.isConnected=true;
  card.getBoundingClientRect=()=>({top:100,height:100,width:130});
  for(const part of ['nd-title','nd-callable','nd-arguments','nd-pins','nd-warning',
    'nd-toggle','nd-drawer','nd-params','nd-runtime','nd-runtime-view','nd-runtime-status']) {
    const el=editableElement();el.className=part;card.parts['.'+part]=el;
  }
  card.parts['.nd-drawer'].hidden=true;
  for(const side of ['input','output']) {
    const pins=editableElement();card.parts['.'+side+'s']=pins;
    for(const port of Object.keys(graphNodes[key][id][side+'s'])) {
      const pin=editableElement();pin.offsetHeight=12;pins.appendChild(pin);
      card.parts['.'+side+'s > .'+port]=pin;
    }
  }
  card.querySelectorAll=selector=>selector==='.nd-pin-row'?card.parts['.nd-pins'].children:[];
  return card;
}
clipboardUI.document={body,get activeElement(){return activeElement;},
  createElement:tag=>editableElement(tag),getElementById:()=>null,
  querySelector:()=>editableElement(),querySelectorAll:()=>[],
  addEventListener:(name,callback,capture)=>{(docEvents[name+(capture?':capture':'')]||=[]).push(callback);},
  removeEventListener:(name,callback,capture)=>{const key=name+(capture?':capture':'');docEvents[key]=(docEvents[key]||[]).filter(c=>c!==callback);}};
clipboardUI.graphNodes=graphNodes;clipboardUI.graphElements=graphElements;
clipboardUI.makeScope=key=>{
  graphNodes[key]={};graphElements[key]={};
  let nextId=1;
  const container=editableElement();container.isConnected=true;
  container.querySelector=s=>graphElements[key][s.match(/node-(\d+)/)?.[1]]||null;
  container.querySelectorAll=()=>[];
  const editor={module:'Home',drawflow:{drawflow:{Home:{data:graphNodes[key]}}},
    precanvas:{clientWidth:1000,clientHeight:800,getBoundingClientRect:()=>({left:20,top:10,width:500,height:400})},
    addNode(type,numIn,numOut,x,y,css,data,html){
      const id=nextId++;
      graphNodes[key][id]={id,name:type,data,pos_x:x,pos_y:y,class:css,html,
        inputs:Object.fromEntries(Array.from({length:numIn},(_,i)=>['input_'+(i+1),{connections:[]}])),
        outputs:Object.fromEntries(Array.from({length:numOut},(_,i)=>['output_'+(i+1),{connections:[]}]))};
      graphElements[key][id]=cardElement(id,key);
      vm.runInContext('ED.hooks.onCanvasChanged?.()',clipboardUI);
      return id;
    },
    import(obj){for(const [id,n] of Object.entries(obj.drawflow.Home.data)){
      graphNodes[key][id]=n;graphElements[key][id]=cardElement(id,key);}},
    addNodeInput(id){const n=graphNodes[key][id];n.inputs['input_'+(Object.keys(n.inputs).length+1)]={connections:[]};},
    // Mirrors Drawflow: drop the port, its connections and the upstream records; last port → no renumbering.
    removeNodeInput(id,cls){const n=graphNodes[key][id];
      for(const c of n.inputs[cls].connections){const up=graphNodes[key][c.node];
        for(const o of Object.values(up.outputs)) o.connections=o.connections.filter(x=>!(x.node===String(id)&&x.output===cls));}
      delete n.inputs[cls];},
    updateNodeDataFromId(id,data){graphNodes[key][id].data=data;},updateConnectionNodes(){},export:()=>({drawflow:{Home:{data:graphNodes[key]}}})};
  return {editor,container};
};
vm.runInContext(`
const realMakeEditor=ED.makeEditor;
ED.makeEditor=key=>{ED.canvases[key]=makeScope(key);return ED.canvases[key].editor;};
App.buildPalette=App.bindHeader=App.renderTabs=App.renderFnPalette=()=>{};
App.initProjects=async()=>{};App.highlightCodeSelection=()=>{};
App.scheduleSync=()=>{App.edits=(App.edits||0)+1;App.editRevision++;};
App.renderTensorViewer=()=>{};
App.init();App.project={id:'offline-a'};
ED.canvases['fn:Model']=makeScope('fn:Model');
App.state.fnOrder=['Model'];App.state.definitions={Model:{kind:'model',params:''}};
`,clipboardUI);
function clipEval(code){return vm.runInContext(code,clipboardUI);}
const layerId=clipEval("ED.addNodeTo('fn:Model','linear',40,50)");
Object.assign(graphNodes['fn:Model'][layerId].data,{name:'WQ',in_features:'embedding_dim',out_features:768,bias:false,runtime:{values:[99]},id:'unsafe'});
clipEval(`ED.refreshSummary('fn:Model',${layerId})`);
const layerCard=graphElements['fn:Model'][layerId];
assert.equal(layerCard.parts['.nd-title'].textContent,'WQ');
assert.equal(layerCard.parts['.nd-callable'].textContent,'nn.Linear');   // 黄色功能标题 = NODE_DEFS.label
assert.equal(layerCard.parts['.nd-arguments'].textContent,'(embedding_dim, 768, bias=False)');
assert(!layerCard.parts['.nd-arguments'].textContent.includes('WQ'));
// 常显参数：不打开任何抽屉，表单已随节点挂载，外部改值后就地同步且不重复绑定。
const layerForm=layerCard.parts['.nd-params'];
assert.equal(layerForm.dataset.mounted,'true');
assert(!layerForm.html.includes('ins-title'));
assert(!layerForm.controls.some(c=>c.dataset.pkey==='__name'));
const featureField=layerForm.controls.find(c=>c.dataset.pkey==='in_features');
assert.equal(featureField.value,'embedding_dim');
graphNodes['fn:Model'][layerId].data.in_features='768';
clipEval(`ED.refreshSummary('fn:Model',${layerId})`);
assert.equal(featureField.value,'768');
assert.strictEqual(layerForm.controls.find(c=>c.dataset.pkey==='in_features'),featureField);
featureField.value='embedding_dim';featureField.events.input();
assert.equal(graphNodes['fn:Model'][layerId].data.in_features,'embedding_dim');
Object.assign(graphNodes['fn:Model'][layerId].data,{name:'<img onerror=alert(1)>'.repeat(5)});
clipEval(`ED.refreshSummary('fn:Model',${layerId})`);
assert.equal(layerCard.parts['.nd-title'].textContent,graphNodes['fn:Model'][layerId].data.name);
assert.equal(layerCard.parts['.nd-title'].children.length,0);
graphNodes['fn:Model'][layerId].data.name='WQ';
// 标题编辑：Enter 提交、Esc 取消、空值失焦不提交；编辑中的同步刷新不打断。
clipEval(`App.editNodeName('fn:Model',${layerId})`);
const layerNameInput=layerCard.parts['.nd-title'].children[0];
assert.equal(layerNameInput.className,'nd-name-input');assert.equal(layerNameInput.name,'name');assert.equal(layerNameInput.value,'WQ');
clipEval(`ED.refreshSummary('fn:Model',${layerId})`);
assert.strictEqual(layerCard.parts['.nd-title'].children[0],layerNameInput);
layerNameInput.value='WQ_renamed';
docFire('keydown',true,{target:layerNameInput,key:'Enter',preventDefault(){}});
assert.equal(graphNodes['fn:Model'][layerId].data.name,'WQ_renamed');
assert.equal(layerCard.parts['.nd-title'].textContent,'WQ_renamed');
assert.equal(layerCard.parts['.nd-title'].children.length,0);
clipEval(`App.editNodeName('fn:Model',${layerId})`);
const layerEscInput=layerCard.parts['.nd-title'].children[0];
layerEscInput.value='discard';
docFire('keydown',true,{target:layerEscInput,key:'Escape',preventDefault(){}});
assert.equal(graphNodes['fn:Model'][layerId].data.name,'WQ_renamed');
assert.equal(layerCard.parts['.nd-title'].textContent,'WQ_renamed');
clipEval(`App.editNodeName('fn:Model',${layerId})`);
const layerBlankInput=layerCard.parts['.nd-title'].children[0];
layerBlankInput.value='  ';layerBlankInput.events.blur();
assert.equal(graphNodes['fn:Model'][layerId].data.name,'WQ_renamed');
graphNodes['fn:Model'][layerId].data.name='WQ';
clipEval(`ED.refreshSummary('fn:Model',${layerId})`);
clipEval(`App.showInspector('fn:Model',${layerId})`);
graphNodes['fn:Model'][layerId].inputs.input_1.connections=[{node:'777',input:'output_1'}];
graphNodes['fn:Model'][layerId].outputs.output_1.connections=[{node:'888',output:'input_1'}];
const originalLayer=JSON.stringify(graphNodes['fn:Model'][layerId]);
assert.equal(clipEval(`ED.copyNode('fn:Model',${layerId})`),true);
assert.equal(clipEval('ED.clipboard.data.runtime'),undefined);assert.equal(clipEval('ED.clipboard.data.id'),undefined);
const beforePasteEdits=clipEval('App.edits');
const cloneId=clipEval("ED.pasteNode('fn:Model',123.4,456.6)");
assert.notEqual(cloneId,layerId);assert.equal(clipEval('App.edits'),beforePasteEdits+1);
const clonedLayer=graphNodes['fn:Model'][cloneId];
assert.equal(clonedLayer.data.name,'WQ_copy');assert.equal(clonedLayer.pos_x,123);assert.equal(clonedLayer.pos_y,457);
for(const parameter of ['in_features','out_features','bias'])assert.equal(clonedLayer.data[parameter],graphNodes['fn:Model'][layerId].data[parameter]);
assert.equal(JSON.stringify(graphNodes['fn:Model'][layerId]),originalLayer);
assert(Object.values({...clonedLayer.inputs,...clonedLayer.outputs}).every(p=>p.connections.length===0));
assert(clipEval(`App.nodeDrawer('fn:Model',${cloneId})`));                       // 展开状态随粘贴恢复
assert.equal(graphElements['fn:Model'][cloneId].parts['.nd-runtime-view'].children.length,0);
assert.equal(graphElements['fn:Model'][cloneId].parts['.nd-params'].dataset.mounted,'true');   // 粘贴即挂载常显参数
assert(graphElements['fn:Model'][cloneId].parts['.nd-params'].controls.some(c=>c.dataset.pkey==='in_features'));
clonedLayer.data.in_features=17;assert.equal(graphNodes['fn:Model'][layerId].data.in_features,'embedding_dim');
const againId=clipEval("ED.pasteNode('fn:Model',0,0)");assert.equal(graphNodes['fn:Model'][againId].data.name,'WQ_copy_2');
assert.equal(graphNodes['fn:Model'][againId].data.in_features,'embedding_dim');

const paramId=clipEval("ED.addNodeTo('fn:Model','fn_input',0,0)");
clipEval(`ED.refreshSummary('fn:Model',${paramId});ED.copyNode('fn:Model',${paramId})`);
assert.equal(graphElements['fn:Model'][paramId].parts['.nd-title'].textContent,'x');
assert.equal(graphElements['fn:Model'][paramId].parts['.nd-arguments'].hidden,true);
assert(!graphElements['fn:Model'][paramId].parts['.nd-params'].controls.some(c=>c.dataset.pkey==='argname'));
assert.equal(clipEval("ED.canPaste('main')"),false);assert.equal(clipEval("ED.pasteNode('main',0,0)"),null);
const paramClone=clipEval("ED.pasteNode('fn:Model',0,0)");
assert.equal(graphNodes['fn:Model'][paramClone].data.argname,'x_copy');
// 输入节点标题编辑写入 argname 字段
clipEval(`App.editNodeName('fn:Model',${paramId})`);
const layerArgInput=graphElements['fn:Model'][paramId].parts['.nd-title'].children[0];
assert.equal(layerArgInput.name,'argname');assert.equal(layerArgInput.value,'x');
layerArgInput.value='ids';
docFire('keydown',true,{target:layerArgInput,key:'Enter',preventDefault(){}});
assert.equal(graphNodes['fn:Model'][paramId].data.argname,'ids');
assert.equal(graphElements['fn:Model'][paramId].parts['.nd-title'].textContent,'ids');
const returnId=clipEval("ED.addNodeTo('fn:Model','fn_output',0,0)");
for(let i=2;i<=4;i++)graphNodes['fn:Model'][returnId].inputs['input_'+i]={connections:[]};
clipEval(`ED.refreshSummary('fn:Model',${returnId});ED.copyNode('fn:Model',${returnId})`);
assert.equal(graphElements['fn:Model'][returnId].parts['.nd-title'].textContent,'return');
assert.equal(clipEval(`ED.inCount(ED.getNode('fn:Model',${returnId}))`),4);
const returnClone=clipEval("ED.pasteNode('fn:Model',0,0)");
assert.equal(Object.keys(graphNodes['fn:Model'][returnClone].inputs).length,4);
assert.equal(Object.keys(graphNodes['fn:Model'][returnClone].outputs).length,0);

// Preserve exact snapshots and dynamic counts for all callable variants, even obsolete ports.
for(const type of ['call','create','invoke','function_call']) {
  clipEval(`App.state.definitions.Model.kind=${JSON.stringify(type==='function_call'?'function':'model')}`);
  const id=clipEval(`ED.addNodeTo('main',${JSON.stringify(type)},0,0,'Model')`);
  const n=graphNodes.main[id];n.data.inLabels=['instance','x','extra'];n.data.outLabels=['first','second','third'];
  n.inputs={input_1:{connections:[{node:'123',input:'output_1'}]},input_2:{connections:[]},input_3:{connections:[]}};
  n.outputs={output_1:{connections:[]},output_2:{connections:[]},output_3:{connections:[]}};
  n.data.init_args='4, 8';
  // Match mock pin markup to the manually expanded graph.
  graphElements.main[id]=cardElement(id,'main');
  clipEval(`ED.refreshSummary('main',${id});ED.copyNode('main',${id})`);
  // 动态类型的黄色功能标题同样是 NODE_DEFS 的准确 label（call 由 editor.js、其余由 architecture.js 定义）。
  assert.equal(graphElements.main[id].parts['.nd-callable'].textContent,
    {call:'模型调用',create:'创建模型实例',invoke:'调用已有实例',function_call:'普通函数调用'}[type]);
  const before=JSON.stringify(n),pasted=clipEval("ED.pasteNode('main',100,200)");
  const clone=graphNodes.main[pasted];
  assert.equal(clone.data.type,type);assert.equal(clone.data.fname,'Model');assert.equal(clone.data.init_args,'4, 8');
  assert.equal(JSON.stringify(clone.data.inLabels),JSON.stringify(n.data.inLabels));
  assert.equal(JSON.stringify(clone.data.outLabels),JSON.stringify(n.data.outLabels));
  assert.equal(Object.keys(clone.inputs).length,3);assert.equal(Object.keys(clone.outputs).length,3);
  assert(Object.values({...clone.inputs,...clone.outputs}).every(p=>p.connections.length===0));
  clone.data.inLabels.push('changed');assert.equal(n.data.inLabels.length,3);assert.equal(JSON.stringify(n),before);
}
assert.equal(clipEval("ED.canPaste('main')"),true);
clipEval('App.state.fnOrder=[]');assert.equal(clipEval("ED.canPaste('main')"),false);
assert.equal(clipEval("ED.pasteNode('main',0,0)"),null);
clipEval("App.state.fnOrder=['Model'];App.project={id:'offline-b'}");
assert.equal(clipEval("ED.canPaste('main')"),false);

// Real redraw uses row centers under zoom, not title/subtitle height constants.
clipEval("App.project={id:'offline-a'}");
const firstMain=Object.keys(graphNodes.main)[0],mainCard=graphElements.main[firstMain];
mainCard.parts['.nd-pins'].children.forEach((row,i)=>{
  row.getBoundingClientRect=()=>({top:160+i*14,height:14});
});
clipEval("ED.canvases.main.redrawPending=null;ED.redraw('main')");
// Limit geometry verification to one node to keep this mock independent of browser layout.
const rest={...graphNodes.main};for(const id of Object.keys(graphNodes.main))if(id!==firstMain)delete graphNodes.main[id];
clipboardUI.frame();
assert.equal(mainCard.parts['.inputs'].children[0].style.top,'127px');
Object.assign(graphNodes.main,rest);

// Menu placement, keyboard/outside closing, node/canvas paste and native edit isolation.
clipEval(`ED.copyNode('main',${firstMain});App.state.definitions.Model.kind='model'`);
clipEval(`ED.openContextMenu('main',${firstMain},319,239)`);
let menu=body.children.at(-1);assert.equal(menu.attributes.role,'menu');
assert.equal(menu.style.left,'184px');assert.equal(menu.style.top,'164px');
assert.equal(menu.children.length,3);assert.equal(menu.children[2].textContent,'删除节点');assert.equal(activeElement,menu.children[0]);
let prevented=0;
docFire('keydown',true,{key:'ArrowDown',preventDefault:()=>prevented++,stopPropagation(){}});
assert.equal(activeElement,menu.children[1]);
docFire('keydown',true,{key:'Escape',preventDefault:()=>prevented++,stopPropagation(){}});
assert.equal(clipEval('ED.contextMenu'),null);assert.equal(body.children.length,0);
const realRemoveNode=clipEval('ED.removeNode');
clipEval('App.removedFromMenu=null;ED.removeNode=(key,id)=>{App.removedFromMenu={key,id};};');
clipEval(`ED.openContextMenu('main',${firstMain},120,60)`);
menu=body.children.at(-1);menu.children[2].onclick();
assert.equal(clipEval('App.removedFromMenu.key'),'main');
assert.equal(String(clipEval('App.removedFromMenu.id')),String(firstMain));
assert.equal(clipEval('ED.contextMenu'),null);
clipboardUI.savedRemoveNode=realRemoveNode;clipEval('ED.removeNode=savedRemoveNode');
clipEval("ED.openContextMenu('main',null,120,60)");menu=body.children.at(-1);
assert.equal(menu.children.length,1);const nodeCount=Object.keys(graphNodes.main).length;
menu.children[0].onclick();assert.equal(Object.keys(graphNodes.main).length,nodeCount+1);
const menuPaste=Object.values(graphNodes.main).at(-1);assert.equal(menuPaste.pos_x,200);assert.equal(menuPaste.pos_y,100);
clipEval("ED.openContextMenu('main',null,120,60)");docFire('pointerdown',true,{target:body});
assert.equal(clipEval('ED.contextMenu'),null);
const connectionClasses=['output_2','connection','node_out_node-17','input_1','node_in_node-9'];
clipboardUI.connectionTarget={closest:s=>s==='svg.connection'?{classList:connectionClasses}:null};
const connectionInfo=clipEval('ED.connectionInfo(connectionTarget)');
assert.deepEqual(JSON.parse(JSON.stringify(connectionInfo)),{outputId:'17',inputId:'9',outputPort:'output_2',inputPort:'input_1'});
clipboardUI.lastDeletedConnection=null;
clipEval('ED.canvases.main.editor.removeSingleConnection=(...args)=>{lastDeletedConnection=args;};');
clipboardUI.testConnection=connectionInfo;
clipEval("ED.openContextMenu('main',null,120,60,testConnection)");menu=body.children.at(-1);
assert.equal(menu.attributes['aria-label'],'连接线菜单');
assert.equal(menu.children.length,1);assert.equal(menu.children[0].textContent,'删除连接');
menu.children[0].onclick();assert.deepEqual(clipboardUI.lastDeletedConnection,['17','9','output_2','input_1']);
assert.equal(clipEval('ED.contextMenu'),null);
const contextListeners={};const contextContainer={addEventListener:(event,handler,capture)=>{(contextListeners[event]||=[]).push({handler,capture});}};
clipboardUI.document.getElementById=()=>contextContainer;
clipboardUI.Drawflow=class {start(){}on(){}};clipboardUI.ResizeObserver=class {observe(){}};
clipEval("realMakeEditor('menu-test','menu-canvas')");
const captured=contextListeners.contextmenu.filter(x=>x.capture);
let rightClick=null,stopped=0;clipboardUI.recordMenu=(...args)=>{rightClick=args;};
clipEval('ED.openContextMenu=(...args)=>recordMenu(...args)');
const headerEvent={target:{closest:s=>s==='.drawflow-node'?{id:'node-9'}:null},clientX:8,clientY:9,
  preventDefault(){},stopImmediatePropagation:()=>stopped++};
captured[1].handler(headerEvent);assert.deepEqual(rightClick,['menu-test','9',8,9]);assert.equal(stopped,1);
rightClick=null;captured[1].handler({...headerEvent,target:{closest:()=>null}});
assert.deepEqual(rightClick,['menu-test',null,8,9]);
rightClick=null;captured[1].handler({...headerEvent,target:{closest:s=>s==='.drawflow-node'?{id:'node-9'}:{}}});
assert.equal(rightClick,null); // Native input/drawer context menu is not hijacked.

// Project load clears the clipboard synchronously, without requests, storage or real files.
clipEval(`
savedImportData=ED.importData;ED.importData=()=>{};App.clearAllModels=()=>{};App.switchTab=()=>{};
App.renderCodePreview=()=>{};App.projectStatus=()=>{};
App.loadProject({id:'other',name:'Other',graph:{seed:1,main:{},functions:[]},code:''});
ED.importData=savedImportData;`);
assert.equal(clipEval('ED.clipboard'),null);
assert.equal(clipEval("ED.pasteNode('main',0,0)"),null);
// Fresh-ID serialization and source-map coverage on a valid, unconnected cloned tensor.
clipEval("ED.canvases.main=makeScope('main');App.state.fnOrder=[];App.state.definitions={};ED.openContextMenu=()=>{};");
const tensorId=clipEval("ED.addNodeTo('main','tensor',0,0)");
clipEval(`ED.copyNode('main',${tensorId})`);
const tensorCopy=clipEval("ED.pasteNode('main',100,200)");
const tensorCode=clipEval('App.generatePython()');assert.equal(clipEval('App.lastErrors.length'),0);
assert(tensorCode.includes('ids_copy = torch.tensor([1, 3, 5])'));
assert(clipEval(`App.codeSourceMap.main[${tensorCopy}].length>0`));
assert(clipEval(`App.tracePlan.some(n=>n.scope==='main'&&n.node==='${tensorCopy}')`));
const snapshot=JSON.parse(clipEval('JSON.stringify(App.snapshot())'));
assert.equal(snapshot.main[tensorCopy].id,tensorCopy);assert.notEqual(tensorId,tensorCopy);
// 导入：importData 重建卡片并为每个节点挂载常显参数表单，表单可继续编辑。
clipboardUI.importGraph={1:{id:1,name:'tensor',data:{type:'tensor',name:'imported_ids',values:'[7,8]'},pos_x:0,pos_y:0,class:'node-tensor',html:'',
  inputs:{},outputs:{output_1:{connections:[]}}}};
clipEval("ED.canvases.main=makeScope('main');ED.importData('main',importGraph)");
const importedCard=graphElements.main[1];
assert.equal(importedCard.parts['.nd-title'].textContent,'imported_ids');
assert.equal(importedCard.parts['.nd-callable'].textContent,'Tensor 输入');   // 导入后黄色功能标题同样是 NODE_DEFS.label
assert.equal(importedCard.parts['.nd-params'].dataset.mounted,'true');
const importedField=importedCard.parts['.nd-params'].controls.find(c=>c.dataset.pkey==='values');
assert.equal(importedField.value,'[7,8]');
importedField.value='[9]';importedField.events.input();
assert.equal(graphNodes.main[1].data.values,'[9]');
assert(clipEval("[...App.drawers.values()].every(d=>d.key!=='main')"));   // 导入重置目标画布的运行结果状态
assert(/nd-summary\s*\{[^}]*white-space:\s*normal;[^}]*overflow-wrap:\s*anywhere;/.test(css));
console.log('PASS: safe name headers, callable/parameters, zoomed pin centers, detached fresh-ID clones, settings/signatures/ports, input names/scope, import mounts always-on params, drawer reset, autosave/source maps, undefined targets, project isolation/reset, accessible clamped context menu and native edit isolation');

// Explicit variable references resolve compiler symbols, not raw node names or expressions.
const refs={1:node(1,'variable_ref',{name:'source',variable:'source'}),2:node(2,'tensor',{name:'source',values:'[2.,3.]'}),3:node(3,'binop',{name:'sum',op:'+'},[[1],[2]])};
result=compile({main:refs},{});assert.deepEqual(result.errors,[]);
assert(result.code.indexOf('source = torch.tensor')<result.code.indexOf('source_2 = source'));
assert.deepEqual(mapped(result,'main',1),['source_2 = source']);
run(result.code,'assert source_2 is source\nassert torch.equal(sum,source+source)');
assert.equal(vm.runInContext("App.tracePlan.find(r=>r.node==='1').sources[0].node",context),'2');
assert.equal(compile(JSON.parse(JSON.stringify({main:refs})),{}).code,result.code);
const multiRefs={...main,5:node(5,'variable_ref',{name:'picked',variable:'a_out2'})};
result=compile({main:multiRefs,'fn:ShankeModel':model},defs);assert.deepEqual(result.errors,[]);
run(result.code,'assert picked is a_out2');
assert.equal(vm.runInContext("App.tracePlan.find(r=>r.scope==='main'&&r.node==='5').sources[0].output",context),2);
for(const variable of ['missing','model','self.dim','x + 1','x[0]','x()','torch.manual_seed','self.dim.real','__import__("os")','class','']) {
  const graph={main:{1:node(1,'variable_ref',{variable})}};
  assert(compile(graph,{}).errors.some(e=>e.includes('变量引用')),variable);
}
const cycle={main:{1:node(1,'variable_ref',{name:'a',variable:'b'}),2:node(2,'variable_ref',{name:'b',variable:'a'})}};
assert(compile(cycle,{}).errors.some(e=>e.includes('循环')));
const edgeCycle={main:{1:node(1,'variable_ref',{name:'a',variable:'b'}),2:node(2,'binop',{name:'b',op:'+'},[[1],[1]])}};
assert(compile(edgeCycle,{}).errors.some(e=>e.includes('循环')));
assert(compile({main:{1:node(1,'variable_ref',{name:'a',variable:'a'})}},{}).errors.some(e=>e.includes('循环')));
const duplicated={main:{1:node(1,'tensor',{name:'v',values:'[1]'}),2:node(2,'tensor',{name:'v',values:'[2]'}),3:node(3,'variable_ref',{name:'ref',variable:'v_2'})}};
result=compile(duplicated,{});assert.deepEqual(result.errors,[]);run(result.code,'assert ref is v_2');
const parameterRef={1:node(1,'variable_ref',{name:'alias',variable:'输入'}),2:node(2,'fn_input',{argname:'输入'}),3:node(3,'fn_output',{},[[1]],0)};
for(const kind of ['model','function']) {
  result=compile({main:{},'fn:ScopedRef':parameterRef},{ScopedRef:{kind,params:''}});assert.deepEqual(result.errors,[]);
  run(result.code,kind==='model'?'x=torch.tensor(2.)\nassert ScopedRef()(x) is x':'x=torch.tensor(2.)\nassert ScopedRef(x) is x');
}
const ctorRefs={1:node(1,'variable_ref',{name:'size',variable:'dim'}),2:node(2,'variable_ref',{name:'size2',variable:'self.dim'}),3:node(3,'fn_input',{argname:'x'}),4:node(4,'linear',{name:'q',in_features:'dim',out_features:'dim',bias:false},[[3]]),5:node(5,'variable_ref',{name:'q_value',variable:'q'}),6:node(6,'variable_ref',{name:'layer',variable:'self.q'}),7:node(7,'fn_output',{},[[1],[2],[5],[6]],0)};
result=compile({main:{},'fn:CtorRef':ctorRefs},{CtorRef:{kind:'model',params:'dim'}});assert.deepEqual(result.errors,[]);
assert.equal((result.code.match(/self.dim = dim/g)||[]).length,1);
assert.deepEqual(mapped(result,'fn:CtorRef',1),['size = self.dim']);
run(result.code,'m=CtorRef(4)\na,b,y,layer=m(torch.randn(2,4))\nassert a==b==4\nassert y.shape==(2,4)\nassert layer is m.q');
for(const variable of ['self.unknown','self.q.weight','self.forward']) {
  const invalid=JSON.parse(JSON.stringify(ctorRefs));invalid[6].data.variable=variable;
  assert(compile({main:{},'fn:CtorRef':invalid},{CtorRef:{kind:'model',params:'dim'}}).errors.some(e=>e.includes('变量引用')));
}
const scopedOnly=JSON.parse(JSON.stringify(ctorRefs));scopedOnly[1].data.variable='unrelated';
assert(compile({main:{9:node(9,'tensor',{name:'unrelated',values:'[1]'})},'fn:CtorRef':scopedOnly},{CtorRef:{kind:'model',params:'dim'}}).errors.some(e=>e.includes('当前作用域')));
const reservedPromotion={1:node(1,'variable_ref',{variable:'self.forward'}),2:node(2,'fn_output',{},[[1]],0)};
assert(compile({main:{},'fn:Reserved':reservedPromotion},{Reserved:{kind:'model',params:'forward'}}).errors.some(e=>e.includes('保留成员')));
const noRef=JSON.parse(JSON.stringify(ctorRefs));delete noRef[1];delete noRef[2];delete noRef[5];delete noRef[6];noRef[7]=node(7,'fn_output',{},[[4]],0);
result=compile({main:{},'fn:NoRef':noRef},{NoRef:{kind:'model',params:'dim'}});assert.deepEqual(result.errors,[]);assert(!result.code.includes('self.dim ='));
const chineseCtor={1:node(1,'variable_ref',{variable:'维度'}),2:node(2,'fn_output',{},[[1]],0)};
result=compile({main:{},'fn:Chinese':chineseCtor},{Chinese:{kind:'model',params:'维度'}});assert.deepEqual(result.errors,[]);run(result.code,'assert Chinese(3)()==3');
assert(compile({main:{1:node(1,'variable_ref',{variable:'x'})},'fn:ScopedRef':parameterRef},{ScopedRef:{kind:'function',params:''}}).errors.some(e=>e.includes('当前作用域')));
const ambiguous=JSON.parse(JSON.stringify(ctorRefs));ambiguous[3].data.argname='dim';
assert(compile({main:{},'fn:CtorRef':ambiguous},{CtorRef:{kind:'model',params:'dim'}}).errors.length);
console.log('PASS: actual symbols, producer-first alias names, forward dependency order, multireturn ports, expression/missing/self rejection, reference and wire cycles, unique duplicate outputs, Chinese class/function parameters, constructor promotion and registered members, no unused promotion, scope isolation');

clipEval("ED.canvases.main=makeScope('main');App.state.fnOrder=[];App.state.definitions={};App.project={id:'offline-a'};");
const referenceId=clipEval("ED.addNodeTo('main','variable_ref',0,0)");
graphNodes.main[referenceId].data.variable='target_out2';
clipEval(`ED.refreshSummary('main',${referenceId});ED.copyNode('main',${referenceId})`);
const referenceClone=clipEval("ED.pasteNode('main',100,200)");
assert.equal(graphNodes.main[referenceClone].data.variable,'target_out2');assert.equal(graphNodes.main[referenceClone].data.name,'ref_copy');
assert.equal(Object.keys(graphNodes.main[referenceClone].inputs).length,0);assert.equal(Object.keys(graphNodes.main[referenceClone].outputs).length,1);
assert.equal(graphElements.main[referenceClone].parts['.nd-callable'].textContent,'变量引用 Variable Reference');   // 黄色功能标题 = NODE_DEFS.label
clipEval(`App.showInspector('main',${referenceClone})`);
const referenceForm=graphElements.main[referenceClone].parts['.nd-params'];
assert(referenceForm.html.includes('变量名'));assert(referenceForm.html.includes('param-help'));
assert(!referenceForm.html.includes('变量引用 Variable Reference'));   // 参数体不重复功能标题
assert(!referenceForm.html.includes('删除该节点'));                    // 底部无删除按钮
assert(!referenceForm.controls.some(c=>c.dataset.pkey==='__name'));
const referenceField=referenceForm.controls.find(c=>c.dataset.pkey==='variable');referenceField.value='target';referenceField.events.input();
assert.equal(graphNodes.main[referenceClone].data.variable,'target');assert.equal(graphNodes.main[referenceId].data.variable,'target_out2');
console.log('PASS: zero-input one-output reference palette/header/help/form, fresh-name clipboard with unchanged explicit reference target');

// ---- 运算 op 多输入：手填 Literal 静态来源、连线优先、左折叠显式括号、逐段运算符、错误校验 ----
vm.runInContext(`for (const s of ['4','0.5','2','1','3','7']) App.literalCache.set(s,{expression:s,type:'int'})`,context);
// 静态开方：两个操作数都手填（4 ** 0.5 = 2），旧 data.op 兼容（无 ops 数组、无连线）
const staticRoot={1:node(1,'binop',{name:'root2',op:'**',values:['4','0.5']},[null,null])};
result=compile({main:staticRoot},{});
assert.deepEqual(result.errors,[]);
assert.deepEqual(mapped(result,'main',1),['root2 = 4 ** 0.5']);
run(result.code,'assert root2 == 2');
assert.equal(compile({main:JSON.parse(JSON.stringify(staticRoot))},{}).code,result.code);   // 旧图序列化恢复不丢数据
// 接线混合输入 + 左折叠：a 连线、b 手填 2、c 连线 → ((t1 + 2) * t2)，显式括号避免优先级歧义
const folded={1:node(1,'tensor',{name:'t1',values:'[1.,2.,3.]'}),2:node(2,'tensor',{name:'t2',values:'[4.,5.,6.]'}),
  3:node(3,'binop',{name:'fold',ops:['+','*'],values:['','2','']},[[1],null,[2]])};
result=compile({main:folded},{});
assert.deepEqual(result.errors,[]);
assert.deepEqual(mapped(result,'main',3),['fold = ((t1 + 2) * t2)']);
run(result.code,'import torch\nassert torch.equal(fold,(t1+2)*t2)');
// 两个操作数保持旧格式 a op b；三个以上逐段独立运算符按左折叠
result=compile({main:{1:node(1,'binop',{name:'seg',ops:['-','*'],values:['7','3','2']},[null,null,null])}},{});
assert.deepEqual(result.errors,[]);
assert.deepEqual(mapped(result,'main',1),['seg = ((7 - 3) * 2)']);
run(result.code,'assert seg == 8');
// 各种运算符（手填字面量全静态，一次执行覆盖）
const opCases=[['+','7','3','10'],['-','7','3','4'],['*','7','3','21'],['/','7','3','7/3'],
  ['//','7','3','2'],['%','7','3','1'],['**','2','3','8'],['>','3','1','True'],['==','3','3','True']];
const opsGraph={};
opCases.forEach(([op,a,b],i)=>{opsGraph[i+1]=node(i+1,'binop',{name:'calc'+i,op,values:[a,b]},[null,null]);});
result=compile({main:opsGraph},{});
assert.deepEqual(result.errors,[]);
opCases.forEach(([op,a,b],i)=>assert(result.code.includes(`calc${i} = ${a} ${op} ${b}`),op));
run(result.code,'assert calc0==10 and calc1==4 and calc2==21 and calc3==7/3 and calc4==2 and calc5==1 and calc6==8 and calc7==True and calc8==True');
// trace_plan：手填值为静态来源（node null），连线依赖优先；source map 覆盖折叠加括号的整行
const planGraph={1:node(1,'tensor',{name:'t1',values:'[1]'}),
  3:node(3,'binop',{name:'mix',ops:['+','*'],values:['','2','']},[[1],null,[1]])};
result=compile({main:planGraph},{});
assert.deepEqual(result.errors,[]);
const mixPlan=vm.runInContext('App.tracePlan.find(r=>r.scope==="main"&&r.node==="3")',context);
assert.equal(mixPlan.sources[0].node,'1');
assert.equal(mixPlan.sources[1].node,null);
assert(mixPlan.sources[1].name.includes('手填 2'),mixPlan.sources[1].name);
assert.equal(mixPlan.sources[2].node,'1');
// 普通函数 forward 内的静态多输入运算：(x + 1) ** 2
const fnOps={1:node(1,'fn_input',{argname:'x'}),2:node(2,'binop',{name:'bigger',ops:['+','**'],values:['','1','2']},[[1],null,null]),3:node(3,'fn_output',{},[[2]],0)};
result=compile({main:{},'fn:Square':fnOps},{Square:{kind:'function',params:''}});
assert.deepEqual(result.errors,[]);
run(result.code,'assert Square(3)==16');
// 错误校验：未连接且未手填 → 准确报错，绝不静默用 0
const missing={1:node(1,'tensor',{values:'[1]'}),2:node(2,'binop',{name:'r',op:'+'},[[1],null])};
result=compile({main:missing},{});
assert(result.errors.some(e=>e.includes('输入 b 未连接且未手填值')),result.errors);
assert(!result.code.includes('\n    r ='),result.code);   // 报错时不生成该节点语句，草稿不可运行
// 非法 / 空运算符；非法与未校验的手填字面量（只信后端校验结果，不 eval）
assert(compile({main:{1:node(1,'binop',{name:'r',op:'x^2',values:['1','2']},[null,null])}},{}).errors.some(e=>e.includes('运算符「x^2」不支持')));
assert(compile({main:{1:node(1,'binop',{name:'r',ops:['','+'],values:['1','2','3']},[null,null,null])}},{}).errors.some(e=>e.includes('运算符「（空）」不支持')));
vm.runInContext(`App.literalCache.set('__import__("os")',{error:'Only numbers, strings, bool, None, lists and tuples are supported'})`,context);
result=compile({main:{1:node(1,'binop',{name:'r',op:'+',values:['1','__import__("os")']},[null,null])}},{});
assert(result.errors.some(e=>e.includes('不是合法字面量')&&e.includes('输入 b')),result.errors);
result=compile({main:{1:node(1,'binop',{name:'r',op:'+',values:['1','99']},[null,null])}},{});
assert(result.errors.some(e=>e.includes('不是合法字面量')&&e.includes('等待 Literal 校验')),result.errors);
console.log('PASS: binop multi-input static literals (sqrt 4**0.5=2), wire-first mixed fold with explicit parens, legacy 2-input form, per-segment ops, trace-plan static sources, function-scope fold, missing/invalid operator/literal errors');

// ---- 运算 op UI：行内值编辑器、行间算子、增删输入（含页面内确认）、连线禁用/恢复、复制保留参数 ----
;(async()=> {
  clipEval("ED.canvases.main=makeScope('main');App.state.fnOrder=[];App.state.definitions={};App.project={id:'offline-op'};");
  const upId=clipEval("ED.addNodeTo('main','tensor',0,0)");
  const binopId=clipEval("ED.addNodeTo('main','binop',60,0)");
  const binopNode=graphNodes.main[binopId];
  let binopCard=graphElements.main[binopId];
  const rebuildCard=()=>{graphElements.main[binopId]=cardElement(binopId,'main');binopCard=graphElements.main[binopId];};
  const refresh=()=>{rebuildCard();clipEval(`ED.refreshSummary('main',${binopId})`);};
  const rowsOf=()=>binopCard.parts['.nd-pins'].children;
  const editors=()=>rowsOf().filter(r=>String(r.className).split(' ').includes('nd-arg-row'))
    .map(r=>r.querySelector('.nd-arg-value'));
  clipEval(`ED.refreshSummary('main',${binopId})`);
  // 结构：2 个输入行 + 行间 1 个算子行；输出在第一个输入行右侧（data-in/out 供 redraw 定位）
  assert.equal(binopCard.parts['.nd-callable'].textContent,'运算 op');
  assert.equal(binopCard.parts['.nd-arguments'].textContent,'a + b');
  assert.equal(rowsOf().length,3);
  assert.equal(rowsOf()[0].dataset.in,'input_1');
  assert.equal(rowsOf()[0].dataset.out,'output_1');
  assert.equal(rowsOf()[1].className,'nd-op-row');
  assert.equal(rowsOf()[2].dataset.in,'input_2');
  assert.equal(rowsOf()[2].dataset.out,undefined);
  assert.equal(rowsOf()[1].children[0].className,'nd-op');
  assert.equal(rowsOf()[1].children[0].value,'+');
  // 参数区：＋/－ 按钮与说明常显；底部不再有 op 字段；最少 2 个时不能移除
  let formBinop=binopCard.parts['.nd-params'];
  assert.equal(formBinop.dataset.mounted,'true');
  assert(formBinop.html.includes('data-ins-action="add-input"'));
  assert(formBinop.html.includes('data-ins-action="remove-input"'));
  assert(!formBinop.controls.some(c=>c.dataset.pkey==='op'));
  assert.equal(await clipEval(`ED.removeOperand('main',${binopId})`),false);
  // ＋ 增加输入：端口 +1、值留空，继承上一段运算符
  assert.equal(await clipEval(`ED.addOperand('main',${binopId})`),true);
  assert.equal(Object.keys(binopNode.inputs).length,3);
  assert.equal(JSON.stringify(binopNode.data.ops),'["+","+"]');
  refresh();
  assert.equal(rowsOf().length,5);   // 3 输入行 + 2 算子行
  // 手填值：输入事件写入 data.values；结构未变时不重建行 → 焦点与元素实例保持
  let eds=editors();
  assert.equal(eds.length,3);
  eds[0].focus();eds[0].value='4';eds[0].events.input();
  assert.equal(binopNode.data.values[0],'4');
  assert.strictEqual(editors()[0],eds[0]);           // 未失焦、未重建
  // 算子每段独立编辑；data.op 同步首段兼容旧读取方；摘要显示左折叠
  rowsOf()[1].children[0].value='*';rowsOf()[1].children[0].events.change();
  assert.equal(JSON.stringify(binopNode.data.ops),'["*","+"]');
  assert.equal(binopNode.data.op,'*');
  assert.equal(binopCard.parts['.nd-arguments'].textContent,'((a * b) + c)');
  // 再增加：新段继承上一段（此前末段 '+'）
  assert.equal(await clipEval(`ED.addOperand('main',${binopId})`),true);
  assert.equal(JSON.stringify(binopNode.data.ops),'["*","+","+"]');
  refresh();
  // 已连线输入：编辑器禁用并注明来自上游；手填值保留，断开后恢复
  eds=editors();
  eds[1].value='7';eds[1].events.input();
  assert.equal(binopNode.data.values[1],'7');
  binopNode.inputs.input_2.connections=[{node:String(upId),input:'output_1'}];
  clipEval(`ED.refreshSummary('main',${binopId})`);
  eds=editors();
  assert.equal(eds[1].disabled,true);
  assert.equal(eds[1].value,'');
  binopNode.inputs.input_2.connections=[];
  clipEval(`ED.refreshSummary('main',${binopId})`);
  eds=editors();
  assert(!eds[1].disabled);                          // 断开后恢复可编辑
  assert.equal(eds[1].value,'7');                    // 断开恢复手填值
  // 移除已连线的末端输入：页面内确认；取消不动，确定只删末端及其连线，前面的连线不受影响
  binopNode.inputs.input_2.connections=[{node:String(upId),input:'output_1'}];
  binopNode.inputs.input_4.connections=[{node:String(upId),input:'output_1'}];
  clipEval('App.confirmAsked=null;App.confirmAnswer=null;ED.confirmDialog=async(title,message)=>{App.confirmAsked=[title,message];return App.confirmAnswer;};');
  clipEval('App.confirmAnswer=false');
  assert.equal(await clipEval(`ED.removeOperand('main',${binopId})`),false);
  assert.equal(Object.keys(binopNode.inputs).length,4);
  assert(String(clipEval('App.confirmAsked[0]')).includes('d'));   // 第 4 个输入名为 d
  clipEval('App.confirmAnswer=true');
  assert.equal(await clipEval(`ED.removeOperand('main',${binopId})`),true);
  assert.equal(Object.keys(binopNode.inputs).length,3);
  assert.equal(JSON.stringify(binopNode.inputs.input_2.connections),'[{"node":"'+upId+'","input":"output_1"}]');
  assert.equal(JSON.stringify(binopNode.data.ops),'["*","+"]');
  assert.equal(JSON.stringify(binopNode.data.values),'["4","7",""]');
  refresh();
  // 复制/粘贴：手填值与运算符数组随节点保留，连线不复制的既有行为不变
  binopNode.data.values=['4','7',''];
  binopNode.data.ops=['*','+'];
  clipEval(`ED.copyNode('main',${binopId})`);
  const pastedBinop=clipEval("ED.pasteNode('main',10,20)");
  assert.equal(JSON.stringify(graphNodes.main[pastedBinop].data.values),'["4","7",""]');
  assert.equal(JSON.stringify(graphNodes.main[pastedBinop].data.ops),'["*","+"]');
  assert.equal(Object.keys(graphNodes.main[pastedBinop].inputs).length,3);
  assert(graphElements.main[pastedBinop].parts['.nd-params'].html.includes('data-ins-action="add-input"'));
  // 上限 16：加满后 addOperand 拒绝，参数区按钮禁用
  for(let i=Object.keys(binopNode.inputs).length;i<16;i++) assert.equal(await clipEval(`ED.addOperand('main',${binopId})`),true);
  assert.equal(Object.keys(binopNode.inputs).length,16);
  assert.equal(await clipEval(`ED.addOperand('main',${binopId})`),false);
  refresh();
  assert.equal(binopCard.parts['.nd-params'].querySelector('[data-ins-action="add-input"]').disabled,true);
  assert.equal(binopCard.parts['.nd-params'].querySelector('[data-ins-action="remove-input"]').disabled,false);
  console.log('PASS: binop pin-row value editors between operator rows, in-place edits without rebuild/focus loss, op segments with legacy data.op sync, add/remove last operand with in-page confirm and untouched earlier wires, 2-16 bounds, wire disables manual value and restores on disconnect, clipboard keeps values/ops');
})().catch(e=>{console.error(e);process.exit(1);});
