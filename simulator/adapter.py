import copy, math
from collections import defaultdict
GPU = {'Orin', 'Xavier'}
from simulator.cost_model.shape_model import ShapeFormula, ShapeWork4
from simulator.cost_model.models import ModelSpec
from simulator.cost_model.advanced_transport import collective_rounds
from simulator.network.route_cost import Network
from .config import ConfigError, catalog
STATS = lambda w: {k: v for k, v in vars(w).items() if k.startswith(('F', 'D')) and isinstance(v, (int, float))}

class AnnotatedWork(ShapeWork4):

    def __init__(self, *args):
        super().__init__(*args)
        self.segments = []
        self.depth = 0

    def capture(self, name, fn, *args):
        if self.depth:
            return fn(*args)
        before = STATS(self)
        self.depth += 1
        try:
            result = fn(*args)
        finally:
            self.depth -= 1
        part = copy.copy(self)
        for k, v in STATS(self).items():
            setattr(part, k, v - before.get(k, 0))
        self.segments.append((name, part))
        return result

    def add(self, *a):
        return self.capture('other', super().add, *a)

    def attention(self, *a):
        return self.capture('attention', super().attention, *a)

    def ffn(self, *a):
        return self.capture('ffn', super().ffn, *a)

    def norm(self, *a):
        return self.capture('norm', super().norm, *a)

    def mm(self, *a):
        return self.capture('output', super().mm, *a)

def mapped_parameters(graph, mapping, artifact, quantization):
    p = {k: v for k, v in artifact['parameters'].items() if not k.startswith('network')}
    for id in [l['id'] for l in graph['links']]:
        link = mapping['_links'][id]
        profile = copy.deepcopy(mapping['link_types'].get(link.get('parameter_profile', 'ethernet'), {}))
        profile.update(mapping.get('link_overrides', {}).get(id, {}))
        ref = profile.get('baseline_ref', 'uplink')
        ref = ref.get(quantization, 'uplink') if isinstance(ref, dict) else ref
        for term in ('alpha_ms', 'b_s_per_GB'):
            value = profile.get('parameters', {}).get(quantization, {}).get(term, artifact['parameters'].get('network1.' + ref + '.' + term))
            if value is None or not math.isfinite(value) or value < 0 or (term == 'b_s_per_GB' and value == 0):
                raise ConfigError('Invalid link parameter: ' + id + '.' + term)
            p['network1.' + id + '.' + term] = value
    return p

def discover_shared(graph, groups):
    network = Network(graph)
    owners = defaultdict(set)
    for g in groups:
        nodes = g['devices']
        ctrl = g['controller']
        pairs = {(ctrl, h) for h in nodes} | {(h, ctrl) for h in nodes}
        if g['parallelism'] == 'tp':
            pairs |= {(a, b) for a in nodes for b in nodes if a != b}
        for a, b in pairs:
            for link in network.cost(a, b, 4)['link_ids']:
                owners[link].add(g['id'])
    return sorted((k for k, v in owners.items() if len(v) > 1))

class ExecutionTrace(ShapeFormula):

    def __init__(self, row, model, graph, parameters, mapping, resources, profile='small4_amortized'):
        self.saved = parameters
        self.mapping = mapping
        self.resources = resources
        self.row = row
        self.trace = []
        self.token = -1
        self.microbatch = -1
        self.layer = 0
        self.stage = -1
        self.last_kind = ''
        self.actual_paths = []
        super().__init__([row], {row['model']: model}, {'network1': graph}, profile)
        self.mark()

    def _case(self, *args):
        from .routing import ExecutionNetwork
        self.network = ExecutionNetwork(self.network.graph)
        self.networks[self.topology] = self.network
        self.work_type = AnnotatedWork
        return super()._case(*args)

    def check_network_cost(self, message, base, service):
        if service <= base + 1e-10:
            return
        q = self.row['quantization']
        baseline = []
        shared = []
        for link in message['link_ids']:
            profile_name = self.mapping['_links'][link].get('parameter_profile', 'ethernet')
            profile = {**self.mapping['link_types'].get(profile_name, {}), **self.mapping.get('link_overrides', {}).get(link, {})}
            ref = profile.get('baseline_ref', 'uplink')
            ref = ref.get(q, 'uplink') if isinstance(ref, dict) else ref
            override = profile.get('parameters', {}).get(q, {})
            baseline.append(f'{link}: baseline_ref={ref}' + (' (with explicit parameter overrides)' if override else '') + f", alpha={self.saved['network1.' + link + '.alpha_ms']:.9g} ms, b={self.saved['network1.' + link + '.b_s_per_GB']:.9g} s/GB")
            if link in self.resources:
                resource = self.resources[link]
                shared.append(f"{link}: service_ref={resource['parameter_ref']}" + (' (with explicit service parameter overrides)' if 'b_s_per_GB' in profile.get('service_parameters', {}) else '') + f", B={resource['bandwidth'] / 1000000.0:.9g} MB/s")
        raise ConfigError('Shared-resource bandwidth mapping is inconsistent with the full-path cost: ' + str(message['link_ids']) + f"; request {self.row['id']}, model {self.row['model']} ({q}), message {message['M_bytes']} bytes" + ', path ' + ' -> '.join(message['path']) + f'; full-path cost={base * 1000:.9g} ms, shared service time={service * 1000:.9g} ms, private={(base - service) * 1000:.9g} ms < 0. ' + 'Baseline mapping: ' + '; '.join(baseline) + '. Shared resources: ' + '; '.join(shared) + '. Check baseline_ref, parameters, service_ref, and service_parameters in link_types / link_overrides. ' + 'New access links inherit the ethernet uplink reference; device types such as RPi5 do not automatically select access-port parameters.')

    def mark(self):
        if self.token >= 0:
            self.trace.append({'kind': 'token', 'token': self.token})

    def duration(self, w, include_dispatch=True):
        vector = self.work_vector(w, 1)
        host = w.host or self.row['controller']
        node = self.mapping['_nodes'][host]
        profile = self.mapping['device_types'][node.get('parameter_profile', node['device_type'])]
        q = self.row['quantization']
        override = {**profile.get('parameters', {}).get(q, {}), **self.mapping.get('device_overrides', {}).get(host, {}).get('parameters', {}).get(q, {})}

        def coefficient(name):
            return override.get(name.split('.', 1)[1], self.saved[name]) if name.startswith(w.cls + '.') else self.saved[name]
        count = len(vector) // 2
        values = [coefficient(self.names[int(vector[k + count])]) for k in range(count)]
        if any((not math.isfinite(v) or v < 0 for v in values)):
            raise ConfigError('Device parameter overrides must be finite and nonnegative.')
        return sum((vector[k] * values[k] for k in range(count) if include_dispatch or not self.names[int(vector[k + count])].endswith('.graph_dispatch_ms')))

    def group(self, works, target, weight, g):
        if g != self.token:
            self.mark()
            self.token = g
            self.microbatch = -1
            self.layer = 0
            self.stage = -1
            self.last_kind = ''
            self.embedding_count = 0
        super().group(works, target, weight, g)
        role = next((name for w in works for name, part in w.segments if name in ('attention', 'ffn', 'output')), 'other')
        if all((w.host is None for w in works)):
            chunks = math.ceil(self.row['batch'] / (self.row['microbatch_tokens'] // self.row['prompt_tokens'])) if g == 0 else 1
            role = 'embedding' if self.embedding_count < chunks else 'sampling'
            if role == 'embedding':
                self.embedding_count += 1
            if role == 'embedding':
                self.microbatch += 1
                self.layer = 0
                self.stage = -1
        durations = []
        for w in works:
            node = w.host or self.row['controller']
            duration = self.duration(w)
            segments = []
            if self.row['mode'] == 'pp' or self.row['mode'] == 'single':
                stage = self.row['nodes'].index(node) if node in self.row['nodes'] else -1
                assigned = self.layouts[0].get('stages', [[]] * len(self.row['nodes']))
                layers = assigned[stage] if stage >= 0 else []
                cursor = -1
                layer = None
                for name, part in w.segments:
                    part.host = w.host
                    d = self.duration(part, include_dispatch=False)
                    if w.host is not None:
                        if name == 'attention':
                            cursor += 1
                            layer = layers[cursor] if cursor < len(layers) else None
                        elif name in ('norm', 'output'):
                            layer = w.model.layers
                    segments.append({'kind': name if name != 'other' else role if w.host is None else 'residual', 'layer': layer, 'duration': d})
                if self.graph_dispatch and w.host is not None:
                    overhead = duration - self.duration(w, include_dispatch=False)
                    segments.insert(0, {'kind': 'graph_dispatch', 'layer': None, 'duration': overhead})
            else:
                stage = self.row['nodes'].index(node) if node in self.row['nodes'] else -1
            durations.append({'device': node, 'rank': stage if self.row['mode'] == 'tp' else None, 'stage': stage if self.row['mode'] == 'pp' else None, 'duration': duration, 'segments': segments})
        self.trace.append({'kind': 'compute', 'token': g, 'layer': self.layer if role in ('attention', 'ffn') else None, 'work': role, 'microbatch': self.microbatch, 'ranks': durations, 'duration': max((d['duration'] for d in durations))})
        if role == 'ffn':
            self.layer += 1
        self.last_kind = role

    def net(self, mode, nodes, size, target, weight, controller=None):
        super().net(mode, nodes, size, target, weight, controller)
        if mode == 'tp':
            rounds = collective_rounds(self.network, list(nodes), int(size), self.classes, self.collective_policy) if len(nodes) > 1 else []
        elif mode == 'pp':
            rounds = [{'name': 'controller_relay', 'messages': [m]} for a, b in zip(nodes, nodes[1:]) for m in self.network.pp_copy(a, b, controller, int(size)) if m['link_ids']]
        else:
            m = self.network.cost(nodes[0], nodes[1], int(size))
            rounds = [{'name': 'controller_io', 'messages': [m]}] if m['link_ids'] else []
        prepared = []
        for rd in rounds:
            messages = []
            demand = defaultdict(float)
            for m in rd['messages']:
                if not m['link_ids'] or not m['M_bytes']:
                    continue
                base = sum((self.saved['network1.' + l + '.alpha_ms'] / 1000 for l in m['link_ids'])) + m['M_bytes'] / 1000000000.0 * max((self.saved['network1.' + l + '.b_s_per_GB'] for l in m['link_ids']))
                resources = []
                for a, b, l in zip(m['path'], m['path'][1:], m['link_ids']):
                    if l in self.resources:
                        resources.append(f'{l}:half' if self.resources[l]['duplex'] == 'half' else f'{l}:{a}->{b}')
                service = max((m['M_bytes'] / self.resources[q.split(':')[0]]['bandwidth'] for q in resources), default=0.0)
                self.check_network_cost(m, base, service)
                spec = {'source': m['source'], 'target': m['target'], 'source_class': self.classes[m['source']], 'path': m['path'], 'links': m['link_ids'], 'bytes': m['M_bytes'], 'resources': resources, 'service': service, 'private': max(0.0, base - service), 'base': base}
                if 'command' in m:
                    spec['command'] = m['command']
                messages.append(spec)
                for q in m['resources']:
                    demand[q] += base
                self.actual_paths.append({k: spec[k] for k in ('source', 'target', 'path', 'links')})
            if messages:
                prepared.append({'messages': messages, 'floor': max(demand.values()), 'name': rd['name']})
        if prepared:
            self.trace.append({'kind': 'collective' if mode == 'tp' else 'messages', 'tensor_bytes': int(size), 'rounds': prepared, 'nodes': list(nodes), 'gpu': [h for h in nodes if self.classes[h] in GPU], 'hierarchical': mode == 'tp' and self.collective_policy == 'nccl_hierarchical', 'token': self.token, 'layer': self.trace[-1].get('layer') if self.trace else None, 'microbatch': self.microbatch})

def prepare_requests(exp, graph, mapping, artifacts):
    mapping = copy.deepcopy(mapping)
    mapping['_links'] = exp['topology']['links']
    mapping['_nodes'] = exp['topology']['nodes']
    for id, n in exp['topology']['nodes'].items():
        if n['kind'] == 'device':
            profile = n.get('parameter_profile', n['device_type'])
            d = mapping['device_types'].get(profile)
            if not d or d.get('source_class') not in ('RPi4', 'RPi5', 'Orin', 'Xavier'):
                raise ConfigError('Missing device parameter profile or invalid source_class: ' + profile)
            graph['nodes'][id]['device_class'] = d['source_class']
    from .routing import resolve_routes
    resolve_routes(graph, exp['groups'])
    shared = discover_shared(graph, exp['groups'])
    declared = mapping.get('shared_links', 'auto')
    if declared != 'auto' and (not set(shared) <= set(declared)):
        raise ConfigError('Actual communication paths contain undeclared shared links: ' + ','.join(set(shared) - set(declared)))
    active = shared if declared == 'auto' else [id for id in declared if id in exp['topology']['links']]
    resources = {}
    for id in active:
        l = exp['topology']['links'][id]
        profile = {**mapping['link_types'].get(l.get('parameter_profile', 'ethernet'), {}), **mapping.get('link_overrides', {}).get(id, {})}
        ref = profile.get('service_ref', 'uplink')
        b = profile.get('service_parameters', {}).get('b_s_per_GB', artifacts['Q8_0']['parameters'].get('network1.' + ref + '.b_s_per_GB'))
        if b is None or not math.isfinite(b) or b <= 0:
            raise ConfigError('Invalid shared-link bandwidth parameter: ' + id)
        resources[id] = {'bandwidth': 1000000000.0 / b, 'endpoints': l['endpoints'], 'parameter_ref': ref, 'duplex': l.get('duplex', 'full')}
    groups = {g['id']: g for g in exp['groups']}
    out = []
    bindings = {}
    cat = catalog(exp)
    for r in exp['requests']:
        group = groups[r['instance']]
        mid = group['model']
        q = cat[mid]['quantization']
        a = artifacts[q]
        model = ModelSpec.from_dict(a['model_specs'][mid])
        params = mapped_parameters(graph, mapping, a, q)
        row = {'id': r['id'], 'model': mid, 'quantization': q, 'mode': group['parallelism'], 'nodes': group['devices'], 'split': group['split'], 'controller': group['controller'], 'topology': 'network1', 'collective_policy': group.get('execution', {}).get('collective_policy', 'nccl_hierarchical'), 'prompt_tokens': r['input_tokens'], 'output_tokens': r['output_tokens'], 'batch': r['batch'], 'requests': 1, 'microbatch_tokens': r['microbatch_tokens']}
        f = ExecutionTrace(row, model, graph, params, mapping, resources, profile=a['profile'])
        paths = list({tuple(x['path']): x for x in f.actual_paths}.values())
        out.append({'request': r, 'group': group, 'ops': f.trace, 'gpu_devices': {h for h, c in f.classes.items() if c in ('Orin', 'Xavier')}})
        bindings[r['id']] = {'row': row, 'layout': f.layouts[0], 'paths': paths, 'parameters': params, 'operation_count': len(f.trace), 'execution_adapter': 'general-ranks/0.2.0', 'mathematical_model': 'rank-ready-network-interference/0.1.0'}
    return (out, resources, bindings)
