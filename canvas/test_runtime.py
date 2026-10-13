"""Bounded previews and isolated tracing HTTP contract tests."""
import io
import json
import tempfile
from pathlib import Path
import unittest
import torch
import runtime_trace
import server
from test_server import PersistenceTest, GRAPH


class PreviewTests(unittest.TestCase):
    def test_bounded_noncontiguous_and_nonfinite(self):
        recorder = runtime_trace.Recorder(io.StringIO())
        value = torch.arange(1000000.).view(1000,1000).transpose(0,1)
        preview = recorder.preview(value)
        self.assertLessEqual(len(preview['values']), runtime_trace.PER_TENSOR)
        self.assertEqual(preview['preview_shape'], [16,16])
        self.assertEqual(preview['values'][1], 1000)
        self.assertTrue(preview['truncated'])
        self.assertEqual(recorder.preview(torch.tensor([True,False]))['values'], [True,False])
        self.assertEqual(recorder.preview(torch.tensor([float('-inf'),float('inf'),float('nan')]))['values'], ['-inf','inf','NaN'])
        self.assertEqual(recorder.preview(torch.tensor([],dtype=torch.int64))['captured'], 0)
        self.assertEqual(recorder.preview(torch.tensor(7))['values'], [7])
        self.assertEqual(recorder.preview(torch.nn.Linear(100,100))['repr'], '<Linear Module>')
        self.assertEqual(recorder.preview('<script>'+('x'*10000))['truncated'], True)
        for _ in range(100):
            recorder.preview(value)
        self.assertLessEqual(recorder.values, runtime_trace.MAX_VALUES)

    def test_record_limits_and_autograd_originals(self):
        stream = io.StringIO()
        recorder = runtime_trace.Recorder(stream)
        x = torch.tensor(2.,requires_grad=True)
        recorder.record(('main','1'),[x])
        (x*x).backward()
        self.assertEqual(x.grad.item(),4.)
        for _ in range(1000):
            recorder.record(('main','1'),[True])
        records=[json.loads(s) for s in stream.getvalue().splitlines()]
        self.assertEqual(sum(r['event']=='record' for r in records), runtime_trace.MAX_RECORDS)
        self.assertEqual(sum(r['event']=='limit' for r in records),1)

    def test_special_named_function_scope_and_input_sources(self):
        for name in ('forward','__init__'):
            code=f'def {name}(x):\n    y=x+x\n    return y\nz={name}(3)\n'
            plan=[{'scope':'fn:'+name,'node':'1','lines':[1],'type':'fn_input','expressions':['x'],'sources':[]},
                  {'scope':'fn:'+name,'node':'2','lines':[2],'type':'binop'},
                  {'scope':'fn:'+name,'node':'3','lines':[3],'type':'fn_output','sources':[{'node':'2','input':1,'output':1}]},
                  {'scope':'main','node':'4','lines':[4],'type':'function_call','sources':[{'node':'9','input':1,'output':1}]}]
            stream=io.StringIO()
            ns=runtime_trace.execute(code,plan,runtime_trace.Recorder(stream))
            self.assertEqual(ns['z'],6)
            records=[json.loads(s) for s in stream.getvalue().splitlines()]
            self.assertEqual(records[0]['scope'],'fn:'+name)
            self.assertEqual(records[0]['context'][-1]['caller']['node'],'4')
            self.assertEqual(records[0]['context'][-1]['caller']['sources'][0]['node'],'9')

    def test_input_failure_captures_prior_outputs(self):
        code='import torch\nx=torch.tensor([1.,2.])\nbad=x.masked_fill(torch.tensor([1,0]),0.)\n'
        plan=[{'scope':'main','node':'1','lines':[2],'type':'tensor'}, {'scope':'main','node':'2','lines':[3],'type':'masked_fill'}]
        stream=io.StringIO()
        with self.assertRaises(RuntimeError):
            runtime_trace.execute(code,plan,runtime_trace.Recorder(stream))
        records=[json.loads(s) for s in stream.getvalue().splitlines()]
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]['node'],'1')


class RuntimeHTTPTests(PersistenceTest):
    def test_saved_trace_identity_partial_failure_and_draft_expiry(self):
        p=self.default()
        graph={**GRAPH,'main':{'1':{'data':{'type':'tensor','name':'x'}},'2':{'data':{'type':'getitem','name':'bad'}}}}
        code='import torch\nx=torch.tensor([1.,2.])\nprint("unstructured output")\nbad=x[100]\n'
        plan=[{'scope':'main','node':'1','lines':[2],'expressions':[],'sources':[]},
              {'scope':'main','node':'2','lines':[4],'expressions':[],'sources':[{'node':'1','input':1,'output':1,'name':'input'}]}]
        p=self.request('/api/save',dict(project_id=p['id'],version=p['version'],graph=graph,code=code,trace_plan=plan))['project']
        result=self.run_project(p)
        self.assertNotEqual(result['returncode'],0)
        self.assertIn('unstructured output',result['stdout'])
        trace=result['trace']
        self.assertEqual(trace['project_id'],p['id'])
        self.assertEqual(trace['revision'],p['revision'])
        self.assertIn('IndexError',trace['error'])
        self.assertEqual(len(trace['records']),1)
        self.assertEqual(trace['records'][0]['outputs'][0]['values'],[1.,2.])
        self.assertEqual((server.DATA_DIR/'projects'/p['id']/'generated_model.py').read_text(),code)
        self.assertTrue((server.DATA_DIR/'projects'/p['id']/'runs'/(trace['run_id']+'.jsonl')).exists())
        from urllib.error import HTTPError
        with self.assertRaises(HTTPError):
            self.request('/api/save',dict(project_id=p['id'],version=p['version'],graph=graph,code=code,trace_plan=[{'scope':'fn:Missing','node':'1','lines':[2]}]))
        draft=self.save(p,None,graph)
        self.assertIsNone(draft['revision'])
        with self.assertRaises(HTTPError):
            self.run_project(p)


if __name__=='__main__':
    unittest.main()
