"""Isolated HTTP project tests; never touch user's data."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
import server

GRAPH = {'version': 3, 'seed': 1234, 'main': {}, 'functions': []}


class PersistenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous = server.DATA_DIR
        server.DATA_DIR = Path(self.temp.name)
        self.legacy = json.dumps(GRAPH)
        (server.DATA_DIR / 'graph.json').write_text(self.legacy)
        (server.DATA_DIR / 'generated_model.py').write_text('print("legacy")')
        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = 'http://127.0.0.1:' + str(self.httpd.server_port)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()
        server.DATA_DIR = self.previous
        self.temp.cleanup()

    def request(self, route, data=None):
        request = Request(self.base + route, data=json.dumps(data).encode() if data is not None else None,
                          headers={'Content-Type': 'application/json'})
        with urlopen(request) as response:
            return json.load(response)

    def default(self):
        result = self.request('/api/projects')
        return self.request('/api/project?id=' + result['default_id'])['project']

    def save(self, p, code='print("saved")', graph=GRAPH):
        return self.request('/api/save', dict(project_id=p['id'], version=p['version'], code=code, graph=graph))['project']

    def run_project(self, p):
        return self.request('/api/run', dict(project_id=p['id'], version=p['version'], revision=p['revision']))

    def test_migration_is_idempotent_and_preserves_legacy(self):
        p = self.default()
        self.assertEqual(p['graph'], GRAPH)
        self.assertEqual(p['id'], self.default()['id'])
        self.save(p)
        self.assertEqual((server.DATA_DIR / 'graph.json').read_text(), self.legacy)
        self.assertEqual((server.DATA_DIR / 'generated_model.py').read_text(), 'print("legacy")')
        self.assertEqual((server.DATA_DIR / 'projects' / p['id'] / 'legacy' / 'graph.json').read_text(), self.legacy)

    def test_isolation_run_project_and_revisions(self):
        first = self.save(self.default(), 'print("first")')
        other = self.request('/api/projects/create', {'name': 'Second'})['project']
        other = self.save(other, 'print("second")')
        self.assertEqual(self.run_project(first)['stdout'].strip(), 'first')
        self.assertEqual(self.run_project(other)['stdout'].strip(), 'second')
        stale = first
        first = self.save(first, 'print("latest")')
        for action in (lambda: self.run_project(stale), lambda: self.save(stale)):
            with self.assertRaises(HTTPError) as error:
                action()
            self.assertEqual(error.exception.code, 409)
        with self.assertRaises(HTTPError):
            self.request('/api/run', dict(project_id=other['id'], version=other['version'], revision=first['revision']))
        self.assertTrue((server.DATA_DIR/'projects'/first['id']/'versions'/'2.json').exists())

    def test_draft_retains_last_code_and_blocks_run_project(self):
        p = self.save(self.default())
        bad = {**GRAPH, 'main': {'1': {'data': {'type': 'literal', 'value': 'print("bad")'}}}}
        with self.assertRaises(HTTPError):
            self.save(p, 'print("bad")', bad)
        draft = self.save(p, None, bad)
        self.assertEqual(draft['graph'], bad)
        self.assertEqual(draft['code'], p['code'])
        with self.assertRaises(HTTPError):
            self.run_project(draft)
        with self.assertRaises(HTTPError):
            self.run_project(p)

    def test_rename_save_as_duplicate_and_invalid_preservation(self):
        p = self.save(self.default())
        copy = self.request('/api/projects/save-as', dict(project_id=p['id'], name='Copy', graph=GRAPH))['project']
        renamed = self.request('/api/projects/rename', dict(project_id=p['id'], version=p['version'], name='Attention'))['project']
        self.assertEqual(renamed['id'], p['id'])
        self.assertEqual(renamed['graph'], p['graph'])
        with self.assertRaises(HTTPError) as error:
            self.request('/api/projects/rename', dict(project_id=copy['id'], version=copy['version'], name='attention'))
        self.assertEqual(error.exception.code, 409)
        before = (server.DATA_DIR/'projects'/p['id']/'project.json').read_bytes()
        for data in [dict(graph=GRAPH, code='pass'), dict(project_id='../graph', version=1, graph=GRAPH),
                     dict(project_id='f'*32, version=1, graph=GRAPH),
                     dict(project_id=p['id'], version=renamed['version'], graph={**GRAPH, 'version': 99}),
                     dict(project_id=p['id'], version=renamed['version'], graph=GRAPH, code='bad syntax !')]:
            with self.assertRaises(HTTPError):
                self.request('/api/save', data)
        self.assertEqual((server.DATA_DIR/'projects'/p['id']/'project.json').read_bytes(), before)

    def test_corrupt_legacy_not_replaced(self):
        (server.DATA_DIR/'graph.json').write_text('{broken')
        with self.assertRaises(HTTPError):
            self.request('/api/projects')
        self.assertEqual((server.DATA_DIR/'graph.json').read_text(), '{broken')
        self.assertFalse((server.DATA_DIR/'projects'/'legacy-project.json').exists())

    def test_literal_validation(self):
        for source, kind in [('42', 'int'), ('-1.5', 'float'), ('1+2j', 'complex'), ('"hi"', 'str'), ('True', 'bool'), ('None', 'NoneType'), ('[1, "x"]', 'list'), ('(1, False)', 'tuple'), ('{"x": 1}', 'dict'), ("{0: '知', 1: '之'}", 'dict')]:
            result = self.request('/api/literal', {'source': source})
            self.assertEqual(result['type'], kind)
        for source in ['__import__("os")', '1 + 2', '[x for x in []]', '{1, 2}', '{k: 1 for k in []}', '{[1]: 2}', '{**{1: 2}}', "{1e999: 'x'}", "{0: {1: 1e999}}", 'b"x"', '1e999', '', '[' * 300, 'x' * 10001]:
            with self.assertRaises(HTTPError):
                self.request('/api/literal', {'source': source})


if __name__ == '__main__':
    unittest.main()
