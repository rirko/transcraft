const fs=require('fs'),vm=require('vm'),assert=require('assert'),cp=require('child_process'),path=require('path');
const context=vm.createContext({console,window:{addEventListener(){}},setTimeout,clearTimeout});
for(const file of ['editor.js','app.js','architecture.js']) vm.runInContext(fs.readFileSync(path.join(__dirname,'static',file),'utf8'),context);
vm.runInContext(`ED.exportData=key=>fixtures[key]||{};App.fnSignature=name=>{const ns=Object.values(fixtures['fn:'+name]||{});return {inputs:ns.filter(n=>n.data.type==='fn_input').map(n=>n.data.argname),outputs:ns.filter(n=>n.data.type==='fn_output').flatMap(n=>Object.keys(n.inputs))};};`,context);
function node(id,type,data={},inputs=[],outputs=1){return {id,name:type,data:{type,name:type,...data},pos_x:40+Math.floor((id-1)/3)*310,pos_y:40+((id-1)%3)*180,inputs:Object.fromEntries(inputs.map((c,i)=>['input_'+(i+1),{connections:c?[{node:String(c[0]),input:'output_'+(c[1]||1)}]:[]}])) ,outputs:Object.fromEntries(Array.from({length:outputs},(_,i)=>['output_'+(i+1),{connections:[]}]))};}
function compile(fixtures,defs={}){context.fixtures=fixtures;context.defs=defs;const result=vm.runInContext(`App.state.fnOrder=Object.keys(defs);App.state.definitions=defs;({code:App.generatePython(),errors:App.lastErrors,plan:App.tracePlan})`,context);result.graph={main:fixtures.main||{},functions:Object.keys(defs).map(name=>({name,data:fixtures['fn:'+name]||{}}))};return result;}
function run(result,checks){assert.deepEqual(result.errors,[]);const payload=JSON.stringify({code:result.code,plan:result.plan,graph:result.graph});const r=cp.spawnSync(path.join(__dirname,'../.venv/Scripts/python.exe'),['-c',`import sys,json,io\nsys.path.insert(0,sys.argv[1])\nfrom runtime_trace import Recorder,execute,validate_plan\np=json.loads(sys.argv[2]);p['plan']=validate_plan(p['plan'],p['graph'],p['code']);stream=io.StringIO();rec=Recorder(stream);ns=execute(p['code'],p['plan'],rec)\nrecords=[json.loads(x) for x in stream.getvalue().splitlines()]\n${checks}`,__dirname,payload],{encoding:'utf8',timeout:60000});assert.equal(r.status,0,r.stderr);}
const main={1:node(1,'tensor',{name:'scores',values:'[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]'}),2:node(2,'ones_like',{name:'ones',dtype:'inherit',device:'inherit'},[[1]]),3:node(3,'triu',{name:'upper',diagonal:1},[[2],null]),4:node(4,'dtype',{name:'mask',dtype:'bool'},[[3]]),5:node(5,'masked_fill',{name:'masked',value:'-inf'},[[1],[4],null]),6:node(6,'softmax',{name:'weights',dim:-1},[[5]]),7:node(7,'print',{label:'weights'},[[6]],0),8:node(8,'tensor',{name:'batches',values:'[[[[1.,2.,3.],[4.,5.,6.]],[[7.,8.,9.],[10.,11.,12.]]], [[[13.,14.,15.],[16.,17.,18.]],[[19.,20.,21.],[22.,23.,24.]]]]'})};
let result=compile({main});
run(result,`import torch\nassert torch.equal(ns['ones'],torch.ones_like(ns['scores']))\nassert torch.equal(ns['upper'],torch.triu(ns['ones'],diagonal=1))\nassert ns['mask'].dtype==torch.bool\nassert torch.equal(ns['masked'],ns['scores'].masked_fill(ns['mask'],float('-inf')))\nassert torch.equal(torch.triu(ns['weights'],1),torch.zeros(3,3))\nassert torch.allclose(ns['weights'].sum(-1),torch.ones(3))\nassert len(records)==7\nassert next(r for r in records if r['node']=='5')['outputs'][0]['values'][1]=='-inf'\nassert next(r for r in records if r['node']=='4')['outputs'][0]['values'][1] is True\nassert next(r for r in records if r['node']=='8')['outputs'][0]['shape']==[2,2,2,3]`);
for(const diagonal of [-2,-1,0,1,2]){const g=JSON.parse(JSON.stringify(main));g[3].data.diagonal=diagonal;delete g[7];delete g[8];run(compile({main:g}),`import torch\nassert torch.equal(ns['upper'],torch.triu(ns['ones'],diagonal=${diagonal}))`);}
vm.runInContext(`App.literalCache.set('1',{expression:'1',type:'int'});App.literalCache.set('-5.0',{expression:'-5.0',type:'float'})`,context);
const linked=JSON.parse(JSON.stringify(main));linked[9]=node(9,'literal',{name:'diag',value:'1'});linked[10]=node(10,'literal',{name:'fill',value:'-5.0'});linked[3].inputs.input_2.connections=[{node:'9',input:'output_1'}];linked[3].data.diagonal='invalid ignored';linked[5].inputs.input_3.connections=[{node:'10',input:'output_1'}];linked[5].data.value='invalid ignored';
run(compile({main:linked}),`import torch\nassert torch.equal(ns['masked'],ns['scores'].masked_fill(ns['mask'],-5.0))`);
const invalid=JSON.parse(JSON.stringify(main));invalid[3].data.diagonal='1.5';assert.throws(()=>compile({main:invalid}),/整数/);invalid[3].data.diagonal=1;invalid[5].data.value='__import__("os")';assert.throws(()=>compile({main:invalid}),/必须/);
const model={1:node(1,'fn_input',{argname:'x'}),2:node(2,'linear',{name:'q',in_features:4,out_features:4,bias:false},[[1]]),3:node(3,'transpose',{name:'kt',dim0:-2,dim1:-1},[[2]]),4:node(4,'matmul',{name:'scores'},[[2],[3]]),5:node(5,'ones_like',{name:'ones',dtype:'inherit',device:'inherit'},[[4]]),6:node(6,'triu',{name:'future',diagonal:1},[[5],null]),7:node(7,'dtype',{name:'mask',dtype:'bool'},[[6]]),8:node(8,'masked_fill',{name:'masked',value:'-inf'},[[4],[7],null]),9:node(9,'softmax',{name:'weights',dim:-1},[[8]]),10:node(10,'matmul',{name:'out'},[[9],[2]]),11:node(11,'fn_output',{},[[10],[9]],0)};
result=compile({'main':{},'fn:Causal':model},{Causal:{kind:'model',params:''}});
run(result,`import torch\nplain={};exec(p['code'],plain)\na=ns['Causal']();b=plain['Causal']();b.load_state_dict(a.state_dict())\nx=torch.randn(3,4,requires_grad=True);y=x.detach().clone().requires_grad_(True)\na(x)[0].sum().backward();b(y)[0].sum().backward()\nassert torch.equal(x.grad,y.grad)\nassert torch.equal(a.q.weight.grad,b.q.weight.grad)\noriginal=x.detach();changed=original.clone();changed[2]+=10\nassert torch.equal(a(original)[0][:2],a(changed)[0][:2])\nassert len(rec.stack)==0\nmore=[json.loads(x) for x in stream.getvalue().splitlines()]\nassert any(r['node']=='1' and r['scope']=='fn:Causal' for r in more)\nassert any(r['node']=='11' and len(r['outputs'])==2 for r in more)`);
const inner={1:node(1,'fn_input',{argname:'x'}),2:node(2,'binop',{name:'twice',op:'+'},[[1],[1]]),3:node(3,'fn_output',{},[[2],[1]],0)};
const outer={1:node(1,'fn_input',{argname:'x'}),2:node(2,'call',{name:'child',fname:'Inner',init_args:'',inLabels:['x'],outLabels:['a','b']},[[1]],2),3:node(3,'fn_output',{},[[2,1],[2,2]],0)};
const nestedMain={1:node(1,'tensor',{values:'[1.,2.]'}),2:node(2,'create',{name:'m',fname:'Outer',init_args:''}),3:node(3,'invoke',{name:'first',fname:'Outer',inLabels:['instance','x'],outLabels:['a','b']},[[2],[1]],2),4:node(4,'invoke',{name:'second',fname:'Outer',inLabels:['instance','x'],outLabels:['a','b']},[[2],[1]],2)};
run(compile({main:nestedMain,'fn:Outer':outer,'fn:Inner':inner},{Outer:{kind:'model',params:''},Inner:{kind:'model',params:''}}),`import torch\nins=[r for r in records if r['scope']=='fn:Inner' and r['node']=='1']\nassert len(ins)==2 and ins[0]['invocation']!=ins[1]['invocation']\nassert len(ins[0]['context'])==2\nassert ins[0]['context'][-1]['caller']['scope']=='fn:Outer'\nassert ins[0]['context'][-1]['caller']['node']=='2'\nassert ins[0]['context'][-1]['caller']['sources'][0]['node']=='1'\nassert any(r['scope']=='fn:Outer' and r['node']=='2' and len(r['outputs'])==2 for r in records)\nassert ns['first_out2'] is ns['tensor']\nassert ns['second_out2'] is ns['tensor']`);
if(process.argv.includes('--export-demo')) {
  const graph={version:3,seed:1234,main,functions:[]};
  for(const [id,n] of Object.entries(main))for(const [port,input] of Object.entries(n.inputs))for(const c of input.connections)main[c.node].outputs[c.input].connections.push({node:id,output:port});
  const demo=compile({main});
  fs.writeFileSync(path.join(__dirname,'backups/before-runtime-20261010/causal-viewer-demo.json'),JSON.stringify({name:'causal-viewer-demo',graph,code:demo.code,trace_plan:demo.plan},null,2));
}
console.log('PASS: eager masking ops, signed diagonal, linked scalar precedence, invalid fallbacks, causal row sums/future invariance, identical gradients, scoped nested multiple invocations and outputs');
for(const reserved of ['_canvas_trace','_canvas_return']) {
  assert.throws(()=>compile({main:{},['fn:'+reserved]:{}},{[reserved]:{kind:'model',params:''}}),/保留名/);
  assert.throws(()=>compile({main:{},'fn:Bad':{}},{Bad:{kind:'model',params:reserved}}),/构造参数名/);
}
console.log('PASS: tracing reserved definition and constructor names');
// Pure freshness predicate, no DOM required.
vm.runInContext(fs.readFileSync(path.join(__dirname,'static/tensor-viewer.js'),'utf8'),context);
assert.equal(vm.runInContext(`App.project={id:'a',revision:'r'};App.editRevision=3;App.tensorIsExpired({project_id:'a',revision:'r',editRevision:3})`,context),false);
for(const test of ["{project_id:'b',revision:'r',editRevision:3}","{project_id:'a',revision:'old',editRevision:3}","{project_id:'a',revision:'r',editRevision:2}"])assert.equal(vm.runInContext(`App.tensorIsExpired(${test})`,context),true);
console.log('PASS: project, saved code revision and unsaved graph freshness');

const refFunction={1:node(1,'variable_ref',{name:'input_alias',variable:'x'}),2:node(2,'fn_input',{argname:'x'}),3:node(3,'binop',{name:'twice',op:'+'},[[1],[1]]),4:node(4,'fn_output',{},[[5],[1]],0),5:node(5,'variable_ref',{name:'local_alias',variable:'twice'})};
const refModel={1:node(1,'variable_ref',{name:'width',variable:'dim'}),2:node(2,'variable_ref',{name:'width2',variable:'self.dim'}),3:node(3,'fn_input',{argname:'x'}),4:node(4,'linear',{name:'q',in_features:'dim',out_features:'dim',bias:false},[[3]]),5:node(5,'variable_ref',{name:'q_alias',variable:'q'}),6:node(6,'variable_ref',{name:'x_alias',variable:'x'}),7:node(7,'binop',{name:'scaled',op:'/'},[[5],[1]]),8:node(8,'fn_output',{},[[7],[6]],0)};
const referenceMain={1:node(1,'variable_ref',{name:'second_alias',variable:'answer_out2'}),2:node(2,'tensor',{name:'data',values:'[1.,2.]'}),3:node(3,'function_call',{name:'answer',fname:'ReferenceFn',inLabels:['x'],outLabels:['twice','x']},[[2]],2)};
result=compile({main:referenceMain,'fn:ReferenceFn':refFunction,'fn:ReferenceModel':refModel},{ReferenceFn:{kind:'function',params:''},ReferenceModel:{kind:'model',params:'dim'}});
assert.deepEqual(result.errors,[]);
assert.equal(result.plan.find(r=>r.scope==='main'&&r.node==='1').sources[0].output,2);
assert.equal(result.plan.find(r=>r.scope==='fn:ReferenceModel'&&r.node==='5').sources[0].node,'4');
assert(result.plan.find(r=>r.scope==='fn:ReferenceModel'&&r.node==='1').sources[0].name.includes('__init__'));
assert.equal(result.plan.find(r=>r.scope==='fn:ReferenceModel'&&r.node==='1').lines.length,1);
run(result,`import torch
assert ns['second_alias'] is ns['data']
assert next(r for r in records if r['scope']=='main' and r['node']=='1')['outputs'][0]['values']==[1.,2.]
assert next(r for r in records if r['scope']=='fn:ReferenceFn' and r['node']=='1')['outputs'][0]['values']==[1.,2.]
plain={};exec(p['code'],plain)
a=ns['ReferenceModel'](4);b=plain['ReferenceModel'](4);b.load_state_dict(a.state_dict())
x=torch.randn(2,4,requires_grad=True);y=x.detach().clone().requires_grad_(True)
a(x)[0].sum().backward();b(y)[0].sum().backward()
assert torch.equal(x.grad,y.grad)
assert torch.equal(a.q.weight.grad,b.q.weight.grad)
more=[json.loads(line) for line in stream.getvalue().splitlines()]
widths=[r for r in more if r['scope']=='fn:ReferenceModel' and r['node']=='1']
assert len(widths)==1 and widths[0]['context'][-1]['phase']=='forward'
assert widths[0]['outputs'][0]['repr']=='4'
q=next(r for r in more if r['scope']=='fn:ReferenceModel' and r['node']=='5')['outputs'][0]
assert q['requires_grad'] and q['shape']==[2,4]
assert not rec.stack`);
const instanceInner={1:node(1,'fn_input',{argname:'x'}),2:node(2,'fn_output',{},[[1]],0)};
const instanceOuter={1:node(1,'variable_ref',{name:'child_alias',variable:'self.child'}),2:node(2,'fn_input',{argname:'x'}),3:node(3,'create',{name:'child',fname:'RefInner',init_args:''}),4:node(4,'invoke',{name:'answer',fname:'RefInner',inLabels:['instance','x'],outLabels:['x']},[[1],[2]]),5:node(5,'fn_output',{},[[4]],0)};
result=compile({main:{},'fn:RefOuter':instanceOuter,'fn:RefInner':instanceInner},{RefOuter:{kind:'model',params:''},RefInner:{kind:'model',params:''}});
run(result,`import torch
m=ns['RefOuter']();x=torch.tensor([3.]);assert m(x) is x
more=[json.loads(line) for line in stream.getvalue().splitlines()]
assert next(r for r in more if r['scope']=='fn:RefOuter' and r['node']=='1')['outputs'][0]['repr']=='<RefInner Module>'`);
const wrongAlias=JSON.parse(JSON.stringify(instanceOuter));wrongAlias[4]=node(4,'binop',{op:'+'},[[1],[2]]);
assert(compile({main:{},'fn:RefOuter':wrongAlias,'fn:RefInner':instanceInner},{RefOuter:{kind:'model',params:''},RefInner:{kind:'model',params:''}}).errors.some(e=>e.includes('模型实例只能')));
console.log('PASS: backend trace-plan validation, implicit reference producer/port sources, constructor provenance, alias-only forward captures, function arguments, multireturn identity, model output aliases, registered instance references and type checks, identical input/parameter gradients');

// ---- 运算 op 多输入：静态手填值执行、sqrt(4**0.5)=2、连线优先混合、trace_plan 静态来源 ----
vm.runInContext(`App.literalCache.set('4',{expression:'4',type:'int'});App.literalCache.set('0.5',{expression:'0.5',type:'float'});App.literalCache.set('2',{expression:'2',type:'int'})`,context);
const opMain={1:node(1,'binop',{name:'root2',op:'**',values:['4','0.5']},[null,null]),
  2:node(2,'tensor',{name:'t',values:'[1.,2.,3.]'}),
  3:node(3,'binop',{name:'mix',ops:['+','*'],values:['','2','']},[[1],null,[2]])};
result=compile({main:opMain});
run(result,`import torch
assert ns['root2']==2
assert torch.equal(ns['mix'],(ns['root2']+2)*ns['t'])
rootp=next(r for r in p['plan'] if r['node']=='1')
assert all(s['node'] is None and s['input']==i+1 for i,s in enumerate(rootp['sources']))
mixp=next(r for r in p['plan'] if r['node']=='3')
assert mixp['sources'][0]['node']=='1'
assert mixp['sources'][1]['node'] is None and '手填 2' in mixp['sources'][1]['name']
assert mixp['sources'][2]['node']=='2'
rootrec=next(r for r in records if r['node']=='1')
assert rootrec['outputs'][0]['repr']=='2.0'
mixrec=next(r for r in records if r['node']=='3')
assert mixrec['outputs'][0]['values']==[4.,8.,12.]`);
console.log('PASS: binop multi-input static value execution (4**0.5=2), wire-first mixed fold, trace-plan static sources and runtime records');
