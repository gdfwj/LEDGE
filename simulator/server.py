import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import ROOT, catalog, read_json
from .runs import RunService, validate_configuration


class Handler(BaseHTTPRequestHandler):
    def log_request(self, code='-', size='-'):
        if isinstance(code, int) and code >= 400:
            super().log_request(code, size)

    def respond(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == '/api/runs':
                return self.respond(self.server.runs.history())
            if path.startswith('/api/runs/'):
                return self.respond(self.server.runs.get(path.removeprefix('/api/runs/')))
            if path == '/api/catalog':
                return self.respond(catalog())
            if path == '/api/parameters':
                return self.respond(read_json(ROOT / 'configs/parameters.json'))
            examples = {p.name: p for p in (ROOT / 'examples').glob('*.json')}
            if path == '/api/examples':
                return self.respond(sorted(examples))
            if path.startswith('/api/example/'):
                name = path.removeprefix('/api/example/')
                if name not in examples:
                    return self.respond({'error': 'Example not found.'}, 404)
                return self.respond(read_json(examples[name]))
            file = (ROOT / 'web' / ('index.html' if path == '/' else path.lstrip('/'))).resolve()
            if ROOT / 'web' not in file.parents or not file.is_file():
                return self.respond({'error': 'Not found.'}, 404)
            data = file.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8'}.get(file.suffix, 'application/octet-stream'))
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (ValueError, KeyError, OSError) as error:
            self.respond({'error': str(error)}, 400)

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 10000000:
                raise ValueError('Configuration must be between 1 byte and 10 MB.')
            body = json.loads(self.rfile.read(length))
            config = body.get('experiment', body)
            if self.path == '/api/validate':
                return self.respond(validate_configuration(config))
            if self.path == '/api/runs':
                return self.respond(self.server.runs.start(config), 202)
            if self.path.startswith('/api/cancel/'):
                return self.respond(self.server.runs.cancel(self.path.removeprefix('/api/cancel/')))
            if self.path == '/api/simulate':
                from .simulation import run_simulation
                return self.respond(run_simulation(config, body.get('mode', 'fast')))
            self.respond({'error': 'Not found.'}, 404)
        except Exception as error:
            self.respond({'error': str(error)}, 400)


def serve(port=8765, run_root=None):
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.runs = RunService(run_root)
    print(f'LEDGE: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
