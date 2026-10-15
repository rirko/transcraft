"""Execution-only AST tracing. Never replace model values or generated source files."""
import ast
import json
import math
from pathlib import Path
import sys
import traceback

MAX_RECORDS = 512
MAX_VALUES = 32768
PER_TENSOR = 2048


def validate_plan(plan, graph, code):
    if plan is None:
        return []
    if not isinstance(plan, list) or len(plan) > 5000:
        raise ValueError('Invalid trace plan')
    scopes = {'main': graph['main'], **{'fn:' + f['name']: f['data'] for f in graph['functions']}}
    result = []
    line_count = len(code.splitlines())
    seen = set()
    for item in plan:
        if not isinstance(item, dict):
            raise ValueError('Invalid trace entry')
        scope, node = item.get('scope'), str(item.get('node'))
        if scope not in scopes or node not in scopes[scope] or (scope, node) in seen:
            raise ValueError('Trace node does not belong to graph')
        seen.add((scope, node))
        lines = item.get('lines')
        if not isinstance(lines, list) or any(type(n) is not int or not 1 <= n <= line_count for n in lines):
            raise ValueError('Invalid trace source lines')
        entry = {'scope':scope, 'node':node, 'lines':lines,
                 'name':str(scopes[scope][node]['data'].get('name',''))[:100]}
        entry['type'] = scopes[scope][node]['data']['type']
        sources = item.get('sources', [])
        if not isinstance(sources, list) or len(sources)>500:
            raise ValueError('Invalid trace sources')
        normalized = []
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError('Invalid trace source')
            upstream = str(source['node']) if source.get('node') is not None else None
            if upstream is not None and upstream not in scopes[scope]:
                raise ValueError('Invalid trace upstream node')
            if type(source.get('input')) is not int or type(source.get('output')) is not int:
                raise ValueError('Invalid trace ports')
            normalized.append({'node':upstream, 'input':source['input'], 'output':source['output'], 'name':str(source.get('name','input'))[:100]})
        entry['sources'] = normalized
        index = item.get('inputIndex')
        if index is not None and (type(index) is not int or not 0 <= index < 500):
            raise ValueError('Invalid trace input index')
        entry['inputIndex'] = index
        # Parameter expressions are emitted by the compiler, but never eval arbitrary text.
        expressions = item.get('expressions', [])
        if not isinstance(expressions, list) or any(not isinstance(x, str) or not x.isidentifier() for x in expressions):
            raise ValueError('Invalid trace parameter expression')
        entry['expressions'] = expressions
        result.append(entry)
    return result


class Recorder:
    def __init__(self, stream):
        self.stream = stream
        self.count = 0
        self.values = 0
        self.serial = 0
        self.stack = []
        self.limited = False
        self.callers = []

    def begin_call(self, scope, node, sources):
        self.callers.append({'scope':scope, 'node':node, 'sources':sources})

    def end_call(self):
        self.callers.pop()

    def enter(self, scope, phase):
        self.serial += 1
        self.stack.append({'scope': scope, 'phase': phase, 'invocation': self.serial,
                           'caller': self.callers[-1] if self.callers else None})

    def leave(self):
        self.stack.pop()

    def number(self, value):
        if isinstance(value, float) and not math.isfinite(value):
            return 'NaN' if math.isnan(value) else '-inf' if value < 0 else 'inf'
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53:
            return str(value)[:200]
        if isinstance(value, complex):
            return str(value)
        return value

    def preview(self, value, depth=0):
        import torch
        kind = type(value).__name__
        if isinstance(value, torch.Tensor):
            shape = list(value.shape)
            total = value.numel()
            info = dict(kind='tensor', type=kind, shape=shape, dtype=str(value.dtype),
                        device=str(value.device), numel=total, requires_grad=value.requires_grad)
            if len(shape) > 16:
                return {**info, 'preview_shape': [], 'values': [], 'truncated': bool(total), 'unavailable':'Rank exceeds preview limit (16)'}
            if value.layout != torch.strided or value.is_quantized or value.device.type == 'meta':
                return {**info, 'preview_shape': [], 'values': [], 'truncated': bool(total), 'unavailable': 'Unsupported preview layout/device'}
            budget = min(PER_TENSOR, MAX_VALUES - self.values)
            sizes = [min(n, 2 if i < len(shape)-2 else 16) for i, n in enumerate(shape)]
            while math.prod(sizes) > budget and sizes:
                index = max(range(len(sizes)), key=lambda i: sizes[i])
                sizes[index] = max(1, sizes[index] // 2)
                if math.prod(sizes) == 1 and budget == 0:
                    break
            if budget <= 0:
                return {**info, 'preview_shape': [0]*len(shape), 'values': [], 'truncated': bool(total), 'unavailable': 'Run preview value budget reached'}
            # Slice before any GPU transfer or contiguous copy; retain originals and autograd.
            with torch.no_grad():
                small = value.detach()[tuple(slice(0, n) for n in sizes)] if shape else value.detach()
                data = small.cpu().reshape(-1).tolist()
            self.values += len(data)
            return {**info, 'preview_shape': sizes, 'values': [self.number(n) for n in data],
                    'captured': len(data), 'truncated': len(data) < total}
        if isinstance(value, torch.nn.Module):
            return dict(kind='value', type=kind, repr='<'+kind+' Module>')
        if value is None or type(value) in (bool, int, float, complex):
            return dict(kind='value', type=kind, repr=str(self.number(value))[:200])
        if isinstance(value, str):
            return dict(kind='value', type=kind, repr=repr(value[:300]), truncated=len(value)>300)
        if type(value) in (tuple, list) and depth < 2:
            return dict(kind='sequence', type=kind, count=len(value),
                        items=[self.preview(x, depth+1) for x in value[:8]], truncated=len(value)>8)
        # 字典：有界键值预览。JSON 键必须是字符串，因此用 repr(key) 保留键的真实
        # 类型面貌（整数 0 显示为 0，字符串 '知' 显示为 '知'），与值成对输出。
        if isinstance(value, dict) and depth < 2:
            entries = []
            for index, (key, item) in enumerate(value.items()):
                if index >= 8:
                    break
                try:
                    key_text = repr(key)
                except Exception:
                    key_text = '<'+type(key).__name__+' 键>'
                entries.append([key_text[:100], self.preview(item, depth+1)])
            return dict(kind='mapping', type=kind, count=len(value),
                        entries=entries, truncated=len(value)>8)
        return dict(kind='value', type=kind, repr='<'+kind+'>')

    def record(self, key, outputs):
        if self.count >= MAX_RECORDS:
            if not self.limited:
                self.emit({'event':'limit', 'message':'Record limit reached (512)'})
                self.limited = True
            return
        self.count += 1
        try:
            previews = [self.preview(value) for value in outputs[:16]]
        except Exception as exc:
            previews = [{'kind':'value', 'type':'preview error', 'repr':str(exc)[:300]}]
        self.emit({'event':'record', 'scope':key[0], 'node':key[1], 'invocation':self.stack[-1]['invocation'] if self.stack else 0,
                   'context':list(self.stack), 'outputs':previews,
                   'output_count':len(outputs), 'outputs_truncated':len(outputs)>16})

    def emit(self, event):
        self.stream.write(json.dumps(event, ensure_ascii=True, allow_nan=False)+'\n')
        self.stream.flush()


class Instrument(ast.NodeTransformer):
    def __init__(self, plan):
        self.plan = plan
        self.scope = 'main'
        self.phase = 'main'
        self.in_class = False

    def call(self, method, args):
        return ast.Expr(ast.Call(ast.Attribute(ast.Name('_canvas_trace', ast.Load()), method, ast.Load()), args, []))

    def record(self, item, outputs):
        return self.call('record', [ast.Constant((item['scope'], item['node'])), ast.List(outputs, ast.Load())])

    def matching(self, node):
        return [p for p in self.plan if p['scope'] == self.scope and node.lineno in p['lines']]

    def visit_ClassDef(self, node):
        previous, was_class = self.scope, self.in_class
        self.scope = 'fn:'+node.name
        self.in_class = True
        self.generic_visit(node)
        self.scope, self.in_class = previous, was_class
        return node

    def visit_FunctionDef(self, node):
        previous, phase = self.scope, self.phase
        if not self.in_class:
            self.scope = 'fn:'+node.name
        self.phase = 'init' if self.in_class and node.name == '__init__' else 'forward'
        original = node.body
        node.body = []
        for statement in original:
            transformed = self.visit(statement)
            node.body.extend(transformed if isinstance(transformed, list) else [transformed])
        entries = []
        if self.phase != 'init':
            for item in self.matching(node):
                if item['type'] == 'fn_input':
                    entries.append(self.record(item, [ast.Name(x, ast.Load()) for x in item['expressions']]))
        node.body = [self.call('enter',[ast.Constant(self.scope),ast.Constant(self.phase)]),
                     ast.Try(entries+node.body, [], [], [self.call('leave',[])])]
        self.scope, self.phase = previous, phase
        return node

    def visit_Assign(self, node):
        records = []
        for item in self.matching(node):
            if item['type'] == 'call' and node.lineno != max(item['lines']):
                continue
            outputs = []
            for target in node.targets:
                parts = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
                for part in parts:
                    if isinstance(part, ast.Name):
                        outputs.append(ast.Name(part.id, ast.Load()))
                    elif isinstance(part, ast.Attribute):
                        outputs.append(ast.Attribute(part.value, part.attr, ast.Load()))
            if outputs:
                records.append(self.record(item, outputs))
        statements = [node]+records
        callers = [item for item in self.matching(node) if item['type'] in ('call','invoke','function_call')
                   and node.lineno == max(item['lines']) and self.phase != 'init']
        if callers:
            item = callers[0]
            sources = item.get('sources', [])[1:] if item['type'] == 'invoke' else item.get('sources', [])
            # Literals are AST-parsed from JSON repr, never evaluated by the recorder.
            begin = self.call('begin_call',[ast.Constant(item['scope']),ast.Constant(item['node']),ast.parse(repr(sources),mode='eval').body])
            return [begin,ast.Try(statements,[],[],[self.call('end_call',[])])]
        return statements

    def visit_Return(self, node):
        items = self.matching(node)
        if not items:
            return node
        # Return value is evaluated once; all output nodes partition the tuple by input count.
        values = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value or ast.Constant(None)]
        assignment = ast.Assign([ast.Name('_canvas_return',ast.Store())], ast.Tuple(values,ast.Load()))
        records = []
        offset = 0
        for item in items:
            count = len(item.get('sources', []))
            records.append(self.record(item,[ast.Subscript(ast.Name('_canvas_return',ast.Load()),ast.Constant(i),ast.Load()) for i in range(offset,min(offset+count,len(values)))]))
            offset += count
        value = ast.Name('_canvas_return',ast.Load()) if isinstance(node.value,ast.Tuple) else ast.Subscript(ast.Name('_canvas_return',ast.Load()),ast.Constant(0),ast.Load())
        return [assignment]+records+[ast.Return(value)]


def instrument(code, plan):
    tree = Instrument(plan).visit(ast.parse(code, filename='generated_model.py'))
    return compile(ast.fix_missing_locations(tree), 'generated_model.py', 'exec')


def execute(code, plan, recorder, namespace=None):
    namespace = {'__name__':'__main__'} if namespace is None else namespace
    namespace['_canvas_trace'] = recorder
    exec(instrument(code, plan), namespace)
    return namespace


def main():
    payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    with Path(sys.argv[2]).open('w', encoding='utf-8') as stream:
        recorder = Recorder(stream)
        try:
            execute(payload['code'],payload['plan'],recorder)
        except BaseException:
            recorder.emit({'event':'error','message':traceback.format_exc()[-8000:]})
            traceback.print_exc()
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
