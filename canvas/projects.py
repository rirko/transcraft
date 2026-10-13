"""Project snapshots; project.json is authoritative, mirrors are for inspection."""
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid
from runtime_trace import validate_plan


class Conflict(ValueError):
    pass


def atomic(path, text):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temp = stream.name
        stream.write(text)
    try:
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def validate_graph(graph):
    if not isinstance(graph, dict) or graph.get('version') not in (2, 3):
        raise ValueError('Expected graph version 2 or 3')
    if not isinstance(graph.get('main'), dict) or not isinstance(graph.get('functions'), list):
        raise ValueError('Expected main object and functions list')
    names = set()
    scopes = [graph['main']]
    for function in graph['functions']:
        if not isinstance(function, dict) or not isinstance(function.get('name'), str) or not function['name'] or function['name'] in names:
            raise ValueError('Invalid or duplicate function name')
        names.add(function['name'])
        scopes.append(function.get('data'))
    for scope in scopes:
        if not isinstance(scope, dict):
            raise ValueError('Invalid graph scope')
        for node in scope.values():
            if not isinstance(node, dict) or not isinstance(node.get('data'), dict) or not isinstance(node['data'].get('type'), str):
                raise ValueError('Invalid graph node')


class Store:
    def __init__(self, root):
        self.root = Path(root) / 'projects'
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, identity):
        if not isinstance(identity, str) or not re.fullmatch(r'[a-f0-9]{32}', identity):
            raise ValueError('Missing or invalid project id')
        path = self.root / identity
        if path.is_symlink() or path.resolve().parent != self.root.resolve():
            raise ValueError('Invalid project path')
        return path

    def load(self, identity):
        path = self.path(identity) / 'project.json'
        if not path.is_file():
            raise FileNotFoundError('Project not found')
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema') != 1:
            raise ValueError('Unsupported project schema')
        return data

    def listing(self):
        return [self.load(p.name) for p in sorted(self.root.iterdir())
                if re.fullmatch(r'[a-f0-9]{32}', p.name) and (p / 'project.json').exists()]

    def name(self, name, identity=None):
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 100 or any(ord(c) < 32 for c in name):
            raise ValueError('Project name must contain 1-100 printable characters')
        name = name.strip()
        if any(p['name'].casefold() == name.casefold() and p['id'] != identity for p in self.listing()):
            raise Conflict('Project name already exists')
        return name

    def commit(self, data):
        folder = self.path(data['id'])
        folder.mkdir(exist_ok=True)
        text = json.dumps(data, ensure_ascii=False, indent=2)
        history = folder / 'versions'
        history.mkdir(exist_ok=True)
        atomic(history / (str(data['version']) + '.json'), text)
        atomic(folder / 'graph.json', json.dumps(data['graph'], ensure_ascii=False, indent=2))
        if data['code'] is not None:
            atomic(folder / 'generated_model.py', data['code'])
        atomic(folder / 'project.json', text)
        return data

    def create(self, name, graph=None, code=None):
        name = self.name(name)
        if graph is None:
            graph = {'version': 3, 'seed': 1234, 'main': {}, 'functions': []}
        validate_graph(graph)
        return self.commit(dict(schema=1, id=uuid.uuid4().hex, name=name, version=1,
                                graph=graph, code=code, revision=None, runnable_version=None))

    def migrate(self):
        marker = self.root / 'legacy-project.json'
        if marker.exists():
            return self.load(json.loads(marker.read_text())['id'])
        legacy = self.root.parent
        graph_file = legacy / 'graph.json'
        code_file = legacy / 'generated_model.py'
        graph = json.loads(graph_file.read_text(encoding='utf-8')) if graph_file.exists() else None
        code = code_file.read_text(encoding='utf-8') if code_file.exists() else None
        project = self.create('Default', graph, code)
        for source in (graph_file, code_file):
            if source.exists():
                backup = self.path(project['id']) / 'legacy'
                backup.mkdir(exist_ok=True)
                shutil.copy2(source, backup / source.name)
        atomic(marker, json.dumps({'id': project['id']}))
        return project

    def checked(self, data):
        project = self.load(data.get('project_id'))
        if type(data.get('version')) is not int or data['version'] != project['version']:
            raise Conflict('Project changed in another tab; save a copy or reload')
        return project

    def save(self, data, validate_literal):
        project = self.checked(data)
        graph = data.get('graph')
        validate_graph(graph)
        code = data.get('code')
        if code is not None:
            if graph['version'] != 3 or not isinstance(code, str):
                raise ValueError('Runnable saves require v3 graph and Python source')
            for scope in [graph['main']] + [f['data'] for f in graph['functions']]:
                for node in scope.values():
                    if node['data']['type'] == 'literal':
                        validate_literal(node['data'].get('value'))
            compile(code, 'generated_model.py', 'exec')
            trace_plan = validate_plan(data.get('trace_plan'), graph, code)
        project['version'] += 1
        project['graph'] = graph
        project['revision'] = uuid.uuid4().hex if code is not None else None
        if code is not None:
            project['code'] = code
            project['trace_plan'] = trace_plan
            project['runnable_version'] = project['version']
        return self.commit(project)
