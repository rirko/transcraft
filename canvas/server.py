"""Local-only canvas server. Standard library.

Trust model: this is a single-user local learning tool. Python execution is the
user's own trusted local interpreter running their own generated code — there is
NO sandbox and none is claimed. Custom node specs (NodeSpec) are validated here
independently of the browser, and candidate try-runs execute code that this
server generates itself from syntax-validated free targets plus declared imports,
in a temporary directory. Dependencies are never auto-installed.
"""
import ast
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import tempfile
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
import nodespecs
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


MAX_LITERAL_DEPTH = 20      # 所有容器字面量的最大嵌套深度
MAX_DICT_ENTRIES = 512      # 字典条目总数预算（嵌套字典合计）
LITERAL_SCALARS = (int, float, complex, str, bool, type(None))


def validate_literal(source):
    if not isinstance(source, str) or not source.strip() or len(source) > 10000:
        raise ValueError('Literal must contain 1-10000 characters')
    try:
        value = ast.literal_eval(source.strip())
    except SyntaxError as exc:
        raise ValueError(f'不是合法的 Python 字面量（括号或引号可能不匹配）：{exc}') from None
    except TypeError as exc:
        # literal_eval 只在这里产生不可哈希字典键（如 {[1]: 'x'}）。
        raise ValueError(f'字典键不可哈希：列表/集合/字典不能作为键（{exc}）') from None
    except ValueError as exc:
        raise ValueError(f'不是合法的 Python 字面量：{exc}') from None
    budget = {'entries': 0}

    def check(item, depth=0):
        # depth 是当前值外层容器的层数；标量不占层，20 层内嵌容器仍然可用。
        if type(item) is dict:
            if depth >= MAX_LITERAL_DEPTH:
                raise ValueError(f'字面量嵌套超过 {MAX_LITERAL_DEPTH} 层')
            for key, child in item.items():
                budget['entries'] += 1
                if budget['entries'] > MAX_DICT_ENTRIES:
                    raise ValueError(f'字典条目总数超过 {MAX_DICT_ENTRIES}')
                if type(key) in (list, set, dict):
                    raise ValueError('字典键只支持数字、字符串、布尔、None 或其元组；列表/集合/字典不能作为键')
                check(key, depth + 1)   # 标量键走同一非有限校验；元组键递归校验元素
                check(child, depth + 1)
        elif type(item) in (list, tuple):
            if depth >= MAX_LITERAL_DEPTH:
                raise ValueError(f'字面量嵌套超过 {MAX_LITERAL_DEPTH} 层')
            for child in item:
                check(child, depth + 1)
        elif type(item) is set:
            raise ValueError('不支持集合（{1, 2}）；如需键值对应写成字典 {键: 值}')
        elif type(item) not in LITERAL_SCALARS:
            raise ValueError('只支持数字、字符串、布尔、None、列表、元组和字典字面量；不接受变量、表达式或调用')
        elif isinstance(item, (float, complex)) and not (math.isfinite(item.real) and math.isfinite(item.imag)):
            raise ValueError('Non-finite numbers are not supported')
    check(value)
    return {'expression': repr(value), 'type': type(value).__name__}


def summary(project):
    return {key: project[key] for key in ('id', 'name', 'version', 'revision', 'runnable_version')}


def run_candidate(code, timeout=20, cap=65536):
    """Run candidate node code in a throwaway temp directory; never touches projects.

    Output is size-capped like project runs; the subprocess has a hard timeout.
    信任边界：候选代码由本服务从规范生成（自由目标 + 声明式 imports 都在规范里
    经语法校验），在本机受信解释器中执行；超时子进程只是兜底，不是安全沙箱。"""
    folder = tempfile.TemporaryDirectory(prefix='canvas-nodespec-')
    with folder:
        path = Path(folder.name) / 'candidate.py'
        path.write_text(code, encoding='utf-8')
        timed_out = False
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                proc = subprocess.run([str(PY_EXE), str(path)], cwd=folder.name,
                                      stdout=stdout, stderr=stderr, timeout=timeout)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                returncode = -1
            def read_output(stream):
                stream.seek(0)
                data = stream.read(cap + 1)
                return data[:cap].decode('utf-8', errors='replace') + ('\n[output truncated]' if len(data) > cap else '')
            out, err = read_output(stdout), read_output(stderr)
    preview, error = None, ''
    if timed_out:
        error = f'候选试运行超时（{timeout} 秒），已终止子进程'
        err += '\n' + error
    else:
        for line in reversed(out.splitlines()):
            text = line.strip()
            if not text.startswith('{'):
                continue
            try:
                candidate = json.loads(text)
            except json.JSONDecodeError:
                continue
            # 兼容两种预览：旧版 Tensor（shape/dtype）与新版的 kind 判别式预览。
            if isinstance(candidate, dict) and ('kind' in candidate or ('shape' in candidate and 'dtype' in candidate)):
                preview = candidate
                break
        if returncode != 0 and not error:
            error = err.strip().splitlines()[-1] if err.strip() else f'候选试运行失败（退出码 {returncode}）'
            missing = re.search(r"ModuleNotFoundError: No module named '([^']+)'", err)
            if missing:
                error = (f"依赖模块不存在：No module named '{missing.group(1)}'。"
                         f"后端不会自动安装依赖，也不支持动态加载任意文件路径；"
                         f"请在本机 .venv 中自行安装后再试运行，或改用已安装的模块。")
    return {'ok': returncode == 0 and preview is not None, 'preview': preview, 'stdout': out,
            'stderr': err, 'returncode': returncode, 'timed_out': timed_out, 'error': error, 'code': code}


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


def literal_checked(source):
    """nodespecs 期望的 literal 契约：结果为 {'expression': …} 或 {'error': …}。"""
    try:
        return validate_literal(source)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        return {'error': str(exc)}


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
            if self.path == '/api/nodespec/try':
                # Independent backend validation; the code below is generated here,
                # executed in a temp dir, and never written to any project.
                spec = nodespecs.validate_spec(data.get('spec'), literal_checked)
                samples = nodespecs.validate_samples(data.get('samples'), spec)
                code = nodespecs.build_try_code(spec, samples, literal_checked)
                self._json(run_candidate(code))
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
