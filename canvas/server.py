"""Local-only canvas server. Standard library, trusted local Python execution."""
import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import tempfile
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
from projects import Store, Conflict

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / 'static'
PROJECT_DIR = BASE_DIR.parent
PY_EXE = PROJECT_DIR / '.venv' / 'Scripts' / 'python.exe'
if not PY_EXE.exists():
    PY_EXE = Path(sys.executable)
DATA_DIR = Path(os.environ.get('CANVAS_DATA_DIR', str(BASE_DIR)))
LOCK = threading.RLock()
PORT = int(os.environ.get('CANVAS_PORT', '8000'))


def validate_literal(source):
    if not isinstance(source, str) or not source.strip() or len(source) > 10000:
        raise ValueError('Literal must contain 1-10000 characters')
    value = ast.literal_eval(source.strip())
    def check(item):
        if type(item) in (list, tuple):
            for child in item:
                check(child)
        elif type(item) not in (int, float, complex, str, bool, type(None)):
            raise ValueError('Only numbers, strings, bool, None, lists and tuples are supported')
        elif isinstance(item, (float, complex)) and not (math.isfinite(item.real) and math.isfinite(item.imag)):
            raise ValueError('Non-finite numbers are not supported')
    check(value)
    return {'expression': repr(value), 'type': type(value).__name__}


def summary(project):
    return {key: project[key] for key in ('id', 'name', 'version', 'revision', 'runnable_version')}


def execute_project(project, cwd):
    run_id = uuid.uuid4().hex
    folder = cwd / 'runs'
    folder.mkdir(exist_ok=True)
    trace_file = folder / (run_id + '.jsonl')
    payload = {'code': project['code'], 'plan': project.get('trace_plan', [])}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8', dir=folder, delete=False) as stream:
        json.dump(payload, stream, ensure_ascii=False)
        execution_file = Path(stream.name)
    timed_out = False
    try:
        # Output is file-backed too: user print does not inflate server capture buffers.
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                proc = subprocess.run([str(PY_EXE), str(BASE_DIR / 'runtime_trace.py'), str(execution_file), str(trace_file)],
                                      cwd=str(cwd), stdout=stdout, stderr=stderr, timeout=30)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                returncode = -1
            def read_output(stream):
                stream.seek(0)
                data = stream.read(65537)
                return data[:65536].decode('utf-8', errors='replace') + ('\n[output truncated]' if len(data)>65536 else '')
            out, err = read_output(stdout), read_output(stderr)
    finally:
        execution_file.unlink(missing_ok=True)
    records, error, limited = [], '', False
    if trace_file.exists():
        # A killed process may leave one incomplete final line; preceding records survive.
        with trace_file.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get('event') == 'record':
                    records.append(event)
                elif event.get('event') == 'error':
                    error = event['message']
                elif event.get('event') == 'limit':
                    limited = True
    if timed_out:
        error = 'Execution timed out (30 seconds)'
        err += '\n' + error
    trace = {'project_id':project['id'], 'revision':project['revision'], 'version':project['version'],
             'run_id':run_id, 'records':records, 'plan':project.get('trace_plan', []),
             'error':error, 'limited':limited, 'code':project['code']}
    return {'ok':not timed_out, 'project_id':project['id'], 'revision':project['revision'],
            'stdout':out, 'stderr':err, 'returncode':returncode, 'trace':trace}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, ctype='application/json; charset=utf-8'):
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(status, json.dumps(obj, ensure_ascii=False).encode('utf-8'))

    def _static(self, rel):
        target = (STATIC_DIR / rel).resolve()
        if not target.is_relative_to(STATIC_DIR.resolve()) or not target.is_file():
            self._send(404, b'not found', 'text/plain')
            return
        types = {'.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.png': 'image/png'}
        self._send(200, target.read_bytes(), types.get(target.suffix, 'application/octet-stream') + '; charset=utf-8')

    def _body(self):
        if self.headers.get('Origin') and self.headers['Origin'] != 'http://' + self.headers.get('Host', ''):
            raise ValueError('Cross-origin requests are not allowed')
        if self.headers.get_content_type() != 'application/json':
            raise ValueError('Expected application/json')
        length = int(self.headers.get('Content-Length', '0'))
        if not 0 < length <= 10_000_000:
            raise ValueError('Request must contain 1-10000000 bytes')
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict):
            raise ValueError('Expected JSON object')
        return data

    def _error(self, exc):
        status = 409 if isinstance(exc, Conflict) else 404 if isinstance(exc, FileNotFoundError) else 422 if isinstance(exc, (ValueError, SyntaxError, TypeError, KeyError, RecursionError)) else 500
        self._json({'ok': False, 'error': str(exc)}, status)

    def do_GET(self):
        try:
            url = urlsplit(self.path)
            if url.path in ('/', '/index.html'):
                self._static('index.html')
            elif url.path == '/api/projects':
                with LOCK:
                    store = Store(DATA_DIR)
                    default = store.migrate()
                    result = {'default_id': default['id'], 'projects': [summary(p) for p in store.listing()]}
                self._json(result)
            elif url.path == '/api/project':
                with LOCK:
                    project = Store(DATA_DIR).load(parse_qs(url.query).get('id', [None])[0])
                self._json({'project': project})
            elif url.path == '/api/graph':
                # Read-only compatibility for already-open pre-project browser tabs.
                path = DATA_DIR / 'graph.json'
                self._json({'graph': json.loads(path.read_text(encoding='utf-8')) if path.exists() else None})
            elif url.path.startswith('/static/'):
                self._static(url.path[len('/static/'):])
            else:
                self._json({'error': 'Not found'}, 404)
        except Exception as exc:
            self._error(exc)

    def do_POST(self):
        try:
            data = self._body()
            if self.path == '/api/literal':
                self._json({'ok': True, **validate_literal(data.get('source'))})
                return
            with LOCK:
                store = Store(DATA_DIR)
                if self.path == '/api/projects/create':
                    project = store.create(data.get('name'))
                elif self.path == '/api/projects/save-as':
                    # Copy the exact client draft; a conflict on the source never destroys it.
                    store.load(data.get('project_id'))
                    if 'graph' not in data:
                        raise ValueError('Missing graph')
                    project = store.create(data.get('name'), data['graph'])
                elif self.path == '/api/projects/rename':
                    project = store.checked(data)
                    project['name'] = store.name(data.get('name'), project['id'])
                    project['version'] += 1
                    project['revision'] = None
                    store.commit(project)
                elif self.path == '/api/save':
                    project = store.save(data, validate_literal)
                elif self.path == '/api/run':
                    project = store.checked(data)
                    if not project['revision'] or data.get('revision') != project['revision']:
                        raise Conflict('Save revision mismatch; save valid code before running')
                    code = project['code']
                    cwd = store.path(project['id'])
                else:
                    self._json({'ok': False, 'error': 'Unknown endpoint'}, 404)
                    return
            if self.path != '/api/run':
                self._json({'ok': True, 'project': project, 'revision': project['revision']})
                return
            self._json(execute_project(project, cwd))
        except Exception as exc:
            self._error(exc)

    def log_message(self, fmt, *args):
        pass


def main():
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    print(f'Canvas: http://127.0.0.1:{PORT}', flush=True)
    print(f'Projects: {DATA_DIR / "projects"}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
