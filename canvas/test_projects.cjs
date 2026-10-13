// Headless persistence-contract tests, no browser or user data.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
function fixture() {
  let graph={version:3,seed:1,main:{},functions:[]};
  const storage=new Map(),calls=[];
  const status={};
  const App={state:{seed:1,fnOrder:[],definitions:{}},refreshSignatures(){},
    validateLiterals:async()=>{},generatePython:()=> 'print(1)',renderCodePreview(){}};
  const context={App,ED:{exportData:()=>graph.main},$:()=>status,
    localStorage:{setItem:(k,v)=>storage.set(k,v),getItem:k=>storage.get(k),removeItem:k=>storage.delete(k)},
    setTimeout:()=>0,clearTimeout(){},console};
  vm.runInNewContext(fs.readFileSync(__dirname+'/static/projects.js','utf8'),context);
  App.project={id:'a',version:1,name:'A'};
  App.request=async(path,data)=> {
    calls.push({path,data});
    return {project:{id:data.project_id,version:data.version+1,name:'A'},revision:data.code?'rev':null};
  };
  return {App,calls,storage,setGraph:value=>{graph=value;}};
}
(async()=> {
  {
    const {App,calls}=fixture();
    const first=App.syncNow(),second=App.syncNow();
    await Promise.all([first,second]);
    assert.deepEqual(calls.map(x=>x.data.version),[1,2]);
    assert.ok(calls.every(x=>x.data.project_id==='a'));
  }
  {
    const {App,calls}=fixture();
    App.generatePython=()=>{throw Error('incomplete');};
    const result=await App.syncNow();
    assert.equal(calls[0].data.code,null);
    assert.equal(result.revision,null);
    assert.equal(App.savedEditRevision,0);
  }
  {
    const {App,calls}=fixture();
    let release;
    App.validateLiterals=()=>new Promise(resolve=>{release=resolve;});
    const save=App.syncNow(); await Promise.resolve();
    App.editRevision++; release();
    assert.equal(await save,false);
    assert.equal(calls.length,0);
  }
  {
    const {App,calls}=fixture();
    let release;
    App.validateLiterals=()=>new Promise(resolve=>{release=resolve;});
    const save=App.syncNow(); await Promise.resolve();
    App.project={id:'b',version:1}; release();
    assert.equal(await save,false);
    assert.equal(calls.length,0);
  }
  {
    const {App,storage}=fixture();
    App.request=async()=>{const e=Error('conflict');e.status=409;throw e;};
    App.editRevision=1;
    assert.equal(await App.syncNow(),false);
    assert.ok(storage.has('canvas-draft:a'));
    assert.equal(App.project.version,1);
  }
  console.log('PASS: serialized versions, incomplete drafts, stale edit/project validation, conflict recovery');
})().catch(error=>{console.error(error);process.exitCode=1;});
