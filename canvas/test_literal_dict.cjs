/*
 * test_literal_dict.cjs —— Literal 字典支持隔离回归（浏览器核心 + 真实执行）
 * 覆盖：后端校验表达式的整数键不被 JSON 字符串化；id_to_token[0] / token_to_id['知']
 * 真实执行；缺键 KeyError 不静默；trace 记录 mapping 有界预览；Literal 值多行文本域
 * 与自定义节点 literal 参数（spectype）共用 textarea；节点卡片摘要截断。
 * 全部使用内存 fixture，不触碰真实项目文件。
 * 运行：node canvas/test_literal_dict.cjs（需要 .venv 的 python 做 /api/literal 同款校验与真实执行）
 */
const fs=require('fs'),vm=require('vm'),assert=require('assert'),cp=require('child_process'),path=require('path');
const context=vm.createContext({console,window:{addEventListener(){}},setTimeout,clearTimeout,
  document:{createElement:tag=>({textContent:'',className:''})}});
for(const file of ['editor.js','app.js','architecture.js','tensor-viewer.js']) vm.runInContext(fs.readFileSync(path.join(__dirname,'static',file),'utf8'),context);
vm.runInContext(`ED.exportData=key=>fixtures[key]||{};App.fnSignature=name=>({inputs:[],outputs:[]});`,context);
function node(id,type,data={},inputs=[],outputs=1){return {id,data:{type,name:type,...data},inputs:Object.fromEntries(inputs.map((c,i)=>['input_'+(i+1),{connections:c?[{node:String(c[0]),input:'output_'+(c[1]||1)}]:[]}])) ,outputs:Object.fromEntries(Array.from({length:outputs},(_,i)=>['output_'+(i+1),{connections:[]}]))};}
function compile(fixtures,defs={}){context.fixtures=fixtures;context.defs=defs;const result=vm.runInContext(`App.state.fnOrder=Object.keys(defs);App.state.definitions=defs;({code:App.generatePython(),errors:App.lastErrors,plan:App.tracePlan})`,context);result.graph={main:fixtures.main||{},functions:Object.keys(defs).map(name=>({name,data:fixtures['fn:'+name]||{}}))};return result;}
const python=path.join(__dirname,'../.venv/Scripts/python.exe');

/* 与浏览器 /api/literal 完全同源：直接调用后端 validate_literal 取规范表达式 */
function checked(source){
  const r=cp.spawnSync(python,['-c','import sys,json;sys.path.insert(0,sys.argv[1]);from server import validate_literal;print(json.dumps(validate_literal(sys.argv[2]),ensure_ascii=False))',__dirname,source],{encoding:'utf8'});
  assert.equal(r.status,0,r.stderr);
  return JSON.parse(r.stdout.trim().split('\n').pop());
}
function seedCache(source){const result=checked(source);vm.runInContext(`App.literalCache.set(${JSON.stringify(source)}, ${JSON.stringify(result)})`,context);return result;}

const ID_MAP="{0: '知', 1: '之', 2: '为', 3: '，', 4: '不', 5: '是', 6: '也'}";
const TOKEN_MAP="{'知': 0, '之': 1}";
const idChecked=seedCache(ID_MAP),tokenChecked=seedCache(TOKEN_MAP);
assert.equal(idChecked.type,'dict');assert.equal(tokenChecked.type,'dict');

/* 1) 生成代码：字典 repr 原样进入语句，整数键 0: 不被字符串化 */
const main={1:node(1,'literal',{name:'id_to_token',value:ID_MAP}),
            2:node(2,'literal',{name:'token_to_id',value:TOKEN_MAP}),
            3:node(3,'getitem',{name:'char',index:'0'},[[1]]),
            4:node(4,'getitem',{name:'token',index:"'知'"},[[2]]),
            5:node(5,'getitem',{name:'second',index:'1'},[[1]])};
const result=compile({main});
assert.deepEqual(result.errors,[],result.errors);
assert.equal((result.code.match(/^\s+id_to_token = \{0: '知', 1: '之', 2: '为', 3: '，', 4: '不', 5: '是', 6: '也'\}$/m)||[]).length,1,result.code);
assert.equal((result.code.match(/^\s+token_to_id = \{'知': 0, '之': 1\}$/m)||[]).length,1,result.code);
assert(result.code.includes("char = id_to_token[0]"),result.code);
assert(result.code.includes("token = token_to_id['知']"),result.code);
assert(!result.code.includes('"0:"')&&!result.code.includes('{"0"'),result.code);

/* 2) 真实执行（含 runtime_trace 记录）：getitem 0 与 '知' 都取到真实值，缺键抛 KeyError */
function traceRun(fixtures,defs){
  const compiled=compile(fixtures,defs);
  const payload=JSON.stringify({code:compiled.code,plan:compiled.plan,graph:compiled.graph});
  const r=cp.spawnSync(python,['-c',[
    "import sys,json,io",
    "sys.path.insert(0,sys.argv[1])",
    "from runtime_trace import Recorder,execute,validate_plan",
    "p=json.loads(sys.argv[2]);p['plan']=validate_plan(p['plan'],p['graph'],p['code'])",
    "stream=io.StringIO()",
    "try:",
    "    ns=execute(p['code'],p['plan'],Recorder(stream))",
    "except BaseException as exc:",
    "    print('EXEC-ERROR:'+type(exc).__name__+':'+str(exc));raise SystemExit(0)",
    "records=[json.loads(x) for x in stream.getvalue().splitlines()]",
    "print('RESULT'+json.dumps({'ns':{k:repr(v) for k,v in ns.items() if not k.startswith('_canvas')},'records':records},ensure_ascii=False))",
  ].join('\n'),__dirname,payload],{encoding:'utf8'});
  assert.equal(r.status,0,r.stderr);
  if(r.stdout.includes('EXEC-ERROR:')) return {error:r.stdout.trim().split('\n')[0]};
  return JSON.parse(r.stdout.trim().split('\n').pop().slice(6));
}
const traced=traceRun({main});
assert.equal(traced.ns.char,"'知'",traced.ns);
assert.equal(traced.ns.token,'0',traced.ns);
assert.equal(traced.ns.second,"'之'",traced.ns);
const mapping=traced.records.find(r=>r.node==='1').outputs[0];
assert.equal(mapping.kind,'mapping');
assert.equal(mapping.type,'dict');
assert.equal(mapping.count,7);
assert.equal(mapping.truncated,false);
assert.equal(mapping.entries[0][0],'0');                       // 整数键无引号
assert.equal(mapping.entries[0][1].repr,"'知'");
assert.equal(mapping.entries[3][0],'3');                       // 整数键无引号（第 4 项）
assert.equal(mapping.entries[3][1].repr,"'，'");
const stringKey=traced.records.find(r=>r.node==='2').outputs[0];
assert.equal(stringKey.entries[0][0],"'知'");                   // 字符串键带引号
assert.equal(stringKey.entries[0][1].repr,'0');
const missing=traceRun({main:{...main,3:node(3,'getitem',{name:'char',index:'999'},[[1]])}});
assert(missing.error&&missing.error.startsWith('EXEC-ERROR:KeyError:'),missing.error);

/* 直接执行同样缺键抛 KeyError，不静默 None */
const raw=cp.spawnSync(python,['-c',`id_to_token = ${idChecked.expression}\nprint(id_to_token[999])`],{encoding:'utf8'});
assert.notEqual(raw.status,0);
assert(raw.stderr.includes('KeyError'),raw.stderr);

/* 3) Literal 值参数（multiline）与自定义节点 literal 参数（spectype）共用多行文本域；
 *    其他 text 参数（如 tensor.values）保持单行 input */
let fakeHtml='';
const box={get innerHTML(){return fakeHtml;},set innerHTML(v){fakeHtml=v;},querySelectorAll:()=>[],querySelector:()=>null,hidden:true};
context.box=box;
vm.runInContext(`ED.getNode=()=>({data:{type:'literal',name:'value',value:${JSON.stringify(ID_MAP)}}});App.renderInspectorForm('main','1',box)`,context);
assert(fakeHtml.includes('<textarea data-pkey="value" rows="3"'),fakeHtml);
assert(fakeHtml.includes('placeholder='),fakeHtml);
fakeHtml='';
vm.runInContext(`ED.getNode=()=>({data:{type:'tensor',name:'ids',values:'[1, 3, 5]'}});App.renderInspectorForm('main','2',box)`,context);
assert(!fakeHtml.includes('<textarea'),fakeHtml);
assert(fakeHtml.includes('<input type="text" data-pkey="values"'),fakeHtml);
fakeHtml='';
vm.runInContext(`NODE_DEFS['custom.dictop@1']={label:'字典查找',params:[{key:'mapping',label:'映射',type:'text',def:"{'知': 0}",spectype:'literal'}]};ED.getNode=()=>({data:{type:'custom.dictop@1',name:'lookup',mapping:"{'知': 0}"}});App.renderInspectorForm('main','3',box)`,context);
assert(fakeHtml.includes('<textarea data-pkey="mapping" rows="3"'),fakeHtml);

/* 4) 节点卡片摘要：多行长字典压缩为单行并截断 */
const summary=vm.runInContext(`NODE_DEFS.literal.summary({value:${JSON.stringify(ID_MAP+'\\n  # long entry '.repeat(4))}})`,context);
assert(!summary.includes('\n')&&summary.length<=49&&summary.endsWith('…'),summary);

/* 5) 查看器渲染分支：mapping 输出标签与键值预览（纯数据函数，无 DOM 依赖部分） */
const viewerText=[];
const fakeDocBox={appendChild:el=>viewerText.push(el.textContent)};
context.fakeBox=fakeDocBox;
vm.runInContext(`App.renderTensorValue(${JSON.stringify(mapping)}, fakeBox, {slices:[],items:{}}, ()=>{})`,context);
assert(viewerText[0].includes('dict · 7 项'),viewerText);
assert(viewerText.some(t=>t.includes('[0]')),viewerText);

console.log('PASS: dict literal compile+execute (id_to_token[0], token_to_id[\'知\']), KeyError on missing key, mapping trace preview, multiline textarea for literal/spectype params, truncated card summary, viewer mapping branch');
