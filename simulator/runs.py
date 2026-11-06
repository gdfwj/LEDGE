import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

from .config import PROJECT, load_bundle, read_json, validate

TERMINAL = {'complete', 'failed', 'cancelled'}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_configuration(config):
    from .adapter import prepare_requests
    experiment, graph, warnings = validate(config)
    mapping, artifacts, interference, _ = load_bundle(experiment)
    from .simulation import validate_interference
    validate_interference(interference['parameters'])
    prepare_requests(experiment, graph, mapping, artifacts)
    return {'valid': True, 'warnings': warnings}


def process_identity(pid):
    try:
        if sys.platform == 'linux':
            fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
            return None if fields[0] == 'Z' else fields[19]
        return subprocess.check_output(['ps', '-p', str(pid), '-o', 'lstart='], text=True).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


class RunService:
    def __init__(self, root=None):
        self.root = Path(root or PROJECT / 'data/runs').resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.children = {}
        self.lock = threading.RLock()

    def start(self, config):
        config = copy.deepcopy(config)
        validation = validate_configuration(config)
        with self.lock:
            rid = time.strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
            folder = self.root / rid
            folder.mkdir()
            mapping, artifacts, interference, _ = load_bundle(config)
            for precision, artifact in artifacts.items():
                target = folder / 'parameters' / (precision + '.json')
                write(target, artifact)
                mapping['baseline_files'][precision] = str(target)
            target = folder / 'parameters/interference.json'
            write(target, interference)
            mapping['interference_file'] = str(target)
            runtime = dict(config, parameters=mapping)
            runtime.pop('parameters_file', None)
            write(folder / 'config.json', config)
            write(folder / 'runtime.json', runtime)
            write(folder / 'validation.json', validation)
            write(folder / 'status.json', dict(id=rid, state='starting', created_unix=time.time()))
            with (folder / 'stdout.log').open('wb') as stream:
                child = subprocess.Popen([sys.executable, '-m', 'simulator.runs', str(folder)],
                    cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.children[rid] = child
            write(folder / 'process.json', dict(pid=child.pid, identity=process_identity(child.pid)))
            def reap():
                child.wait()
                with self.lock:
                    self.children.pop(rid, None)
            threading.Thread(target=reap, daemon=True).start()
            return self.get(rid)

    def folder(self, rid):
        if not rid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in rid):
            raise ValueError('Invalid run ID.')
        folder = self.root / rid
        if not folder.is_dir():
            raise ValueError('Run not found.')
        return folder

    def get(self, rid, include_result=True):
        folder = self.folder(rid)
        state = read_json(folder / 'status.json')
        if state['state'] not in TERMINAL and (folder / 'process.json').exists():
            owner = read_json(folder / 'process.json')
            dead = self.children[rid].poll() is not None if rid in self.children else process_identity(owner['pid']) != owner['identity'] or owner['identity'] is None
            if dead:
                state = read_json(folder / 'status.json')
                if state['state'] not in TERMINAL:
                    state.update(state='cancelled' if (folder / 'cancel.request').exists() else 'failed', ended_unix=time.time())
                    if state['state'] == 'failed':
                        state['error'] = 'Simulation process exited. See stdout.log.'
                    write(folder / 'status.json', state)
        config = read_json(folder / 'config.json')
        state.update(config_name=config.get('name', rid), elapsed_s=(state.get('ended_unix') or time.time()) - state['created_unix'])
        if include_result:
            state['config'] = config
            if (folder / 'result.json').exists():
                state['result'] = read_json(folder / 'result.json')
        return state

    def history(self):
        return [self.get(p.name, False) for p in sorted(self.root.iterdir(), reverse=True) if (p / 'status.json').exists()]

    def cancel(self, rid):
        with self.lock:
            state = self.get(rid)
            if state['state'] in TERMINAL:
                return state
            folder = self.folder(rid)
            owner = read_json(folder / 'process.json')
            if not owner['identity'] or process_identity(owner['pid']) != owner['identity']:
                raise ValueError('Run process is no longer available.')
            (folder / 'cancel.request').touch()
            try:
                os.kill(owner['pid'], signal.SIGTERM)
            except ProcessLookupError:
                return self.get(rid)
            return dict(id=rid, state='cancelling')


def worker(folder):
    folder = Path(folder)
    state = read_json(folder / 'status.json')

    def interrupted(*args):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if (folder / 'cancel.request').exists():
            raise KeyboardInterrupt()
        state['state'] = 'running'
        write(folder / 'status.json', state)
        from .simulation import run_simulation
        config = read_json(folder / 'runtime.json')
        result = run_simulation(config, 'log' if config.get('logging_level', 'trace') == 'trace' else 'fast')
        result['input'] = read_json(folder / 'config.json')
        write(folder / 'result.json', result)
        state['state'] = 'complete'
    except KeyboardInterrupt:
        state['state'] = 'cancelled'
    except Exception as error:
        state.update(state='failed', error=str(error))
    finally:
        state['ended_unix'] = time.time()
        write(folder / 'status.json', state)


if __name__ == '__main__':
    worker(sys.argv[1])
