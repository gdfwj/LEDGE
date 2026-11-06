import copy, json, pathlib, math
ROOT = pathlib.Path(__file__).resolve().parent
PROJECT = ROOT.parent

class ConfigError(ValueError):
    pass

def read_json(path):
    return json.loads(pathlib.Path(path).read_text())

def resolve(path):
    p = pathlib.Path(path).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()

def catalog(experiment=None):
    result = read_json(PROJECT / 'data/models/catalog.json')
    for name, value in (experiment or {}).get('models', {}).items():
        if name in result:
            raise ConfigError('Model ID already exists: ' + name)
        from simulator.cost_model.models import ModelSpec
        spec = ModelSpec.from_dict(value['structure'])
        if value['structure'].get('expert_count', 0):
            raise ConfigError('Only dense models are supported')
        if spec.family not in ('qwen3', 'llama') and value.get('operator_contract') != 'dense_gqa_swiglu_rmsnorm':
            raise ConfigError('Custom models require operator_contract=dense_gqa_swiglu_rmsnorm')
        if any((getattr(spec, k) <= 0 for k in ('layers', 'hidden_size', 'intermediate_size', 'attention_heads', 'kv_heads', 'head_dim', 'vocab_size'))):
            raise ConfigError('Model dimensions must be positive')
        result[name] = dict(value, quantization=spec.quantization, model_id=name)
    return result

def validate_topology(top):
    nodes = top.get('nodes', {})
    links = top.get('links', {})
    adj = top.get('adjacency', {})
    if not nodes:
        raise ConfigError('The topology must contain at least one node.')
    for id, n in nodes.items():
        if not id or ':' in id or '->' in id or (n.get('kind') not in ('device', 'router', 'switch')):
            raise ConfigError('Invalid node ID or kind: ' + id)
        if n['kind'] == 'device' and (not isinstance(n.get('device_type'), str)):
            raise ConfigError('Invalid device type: ' + id)
    seen = {id: [] for id in links}
    pairs = set()
    for src, entries in adj.items():
        if src not in nodes:
            raise ConfigError('Unknown node in adjacency list: ' + src)
        for e in entries:
            if e.get('link') not in links or e.get('node') not in nodes:
                raise ConfigError('An adjacency entry references an unknown node or link.')
            seen[e['link']].append((src, e['node']))
    for id, l in links.items():
        if not id or ':' in id or '->' in id:
            raise ConfigError('Link IDs must be nonempty and cannot contain : or ->.')
        ep = l.get('endpoints', [])
        if len(ep) != 2 or len(set(ep)) != 2 or (not set(ep) <= set(nodes)):
            raise ConfigError('Invalid link endpoints: ' + id)
        pair = tuple(sorted(ep))
        if pair in pairs:
            raise ConfigError('The routing model does not support parallel physical links between the same endpoints.')
        pairs.add(pair)
        if sorted(seen[id]) != sorted([(ep[0], ep[1]), (ep[1], ep[0])]):
            raise ConfigError('Each link must have exactly two reciprocal adjacency entries: ' + id)
        if l.get('duplex', 'full') not in ('full', 'half'):
            raise ConfigError('duplex must be full or half.')
    return {'schema_version': 1, 'id': 'simulation', 'version': '1', 'nodes': {id: {'kind': 'device', 'device_class': n['device_type']} if n['kind'] == 'device' else {'kind': 'fabric'} for id, n in nodes.items()}, 'links': [{'id': id, 'endpoints': l['endpoints'], 'duplex': l.get('duplex', 'full'), 'alpha_s': None, 'B_bytes_per_s': None} for id, l in links.items()], 'routes': top.get('routes', {}), 'cost_rule': {}}

def load_bundle(exp):
    mapping = copy.deepcopy(exp.get('parameters'))
    if mapping is None:
        mapping = read_json(resolve(exp.get('parameters_file', 'configs/parameters.json')))
    artifacts = {}
    provenance = {}
    for q, path in mapping['baseline_files'].items():
        file = resolve(path)
        a = read_json(file)
        if a['profile'] not in ('small4_amortized', 'small4_amortized_graph_dispatch'):
            raise ConfigError('Unsupported shape-aware baseline profile: ' + a['profile'])
        for name, model in exp.get('models', {}).items():
            if model['structure'].get('quantization', 'Q8_0') == q:
                a['model_specs'][name] = copy.deepcopy(model['structure'])
        artifacts[q] = a
        provenance[q] = {'path': str(file)}
    path = resolve(mapping['interference_file'])
    interference = read_json(path)
    if interference.get('model') != 'rank-ready-network-interference/0.1.0':
        raise ConfigError('Unsupported communication model version.')
    provenance['interference'] = {'path': str(path)}
    return (mapping, artifacts, interference, provenance)

def validate(exp):
    if exp.get('execution_mode', 'simulation') != 'simulation' or exp.get('task') or exp.get('clock_policy', 'offline') != 'offline':
        raise ConfigError('Only simulation configurations are supported.')
    e = copy.deepcopy(exp)
    graph = validate_topology(e['topology'])
    models = catalog(e)
    groups = e.get('groups', [])
    ids = set()
    owned = set()
    warnings = []
    if not groups:
        raise ConfigError('Create at least one model instance/device group.')
    for g in groups:
        if not g.get('id') or g['id'] in ids:
            raise ConfigError('Model instance IDs must be unique.')
        ids.add(g['id'])
        if g.get('model') not in models:
            raise ConfigError('The model is not in the built-in or experiment model catalog.')
        nodes = g.get('devices', [])
        ctrl = g.get('controller')
        ownership = set(nodes) | {ctrl}
        if not nodes or len(set(nodes)) != len(nodes):
            raise ConfigError('The ordered device list is empty or contains duplicates: ' + g['id'])
        if any((x not in graph['nodes'] or graph['nodes'][x]['kind'] != 'device' for x in ownership)):
            raise ConfigError('The deployment or controller references a non-device node: ' + g['id'])
        if owned & ownership:
            raise ConfigError('The model requires disjoint compute resources, including controllers; duplicate devices: ' + ','.join(sorted(owned & ownership)))
        owned |= ownership
        mode = g.get('parallelism')
        if mode == 'single' and len(nodes) != 1:
            raise ConfigError('single requires exactly one device.')
        if mode not in ('tp', 'pp', 'single'):
            raise ConfigError('Parallelism must be tp, pp, or single.')
        counts = g.get('pp_layer_counts', [])
        if mode == 'pp' and counts:
            total = models[g['model']]['structure']['layers'] + 1
            if len(counts) != len(nodes) or any((not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in counts)) or sum(counts) != total:
                raise ConfigError('PP layer counts must be positive integers, including the output head, and must sum to ' + str(total))
            bounds = [(sum(counts[:i]) - 0.5) / total for i in range(1, len(counts))]
            edges = [0.0] + bounds + [1.0]
            g['split'] = [b - a for a, b in zip(edges, edges[1:])]
        policy = g.get('execution', {}).get('collective_policy', 'nccl_hierarchical')
        if policy not in ('native_mpi', 'nccl_hierarchical'):
            raise ConfigError('collective_policy must be native_mpi or nccl_hierarchical.')
        split = g.get('split', [])
        if len(split) != len(nodes) or any((not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in split)):
            raise ConfigError('Each device requires a positive TP/PP split ratio.')
        if g.get('execution', {}).get('pp_rule', 'synchronous_cohort') != 'synchronous_cohort':
            raise ConfigError('Set pp_rule to synchronous_cohort.')
        if g.get('scheduling', 'fcfs') != 'fcfs':
            raise ConfigError('Set scheduling to fcfs.')
    reqids = set()
    for r in e.get('requests', []):
        if not r.get('id') or r['id'] in reqids:
            raise ConfigError('Request IDs must be unique.')
        reqids.add(r['id'])
        if r.get('instance') not in ids:
            raise ConfigError('Unknown target instance for request: ' + r['id'])
        if not isinstance(r.get('arrival_s'), (int, float)) or not math.isfinite(r['arrival_s']) or r['arrival_s'] < 0:
            raise ConfigError('Arrival time must be a nonnegative number of seconds.')
        for k in ('input_tokens', 'output_tokens', 'batch', 'microbatch_tokens'):
            if k not in r:
                if k in ('input_tokens', 'output_tokens'):
                    raise ConfigError('Request is missing ' + k)
                r[k] = 1 if k == 'batch' else r['input_tokens'] * r.get('batch', 1)
            if not isinstance(r[k], int) or isinstance(r[k], bool) or r[k] < 1:
                raise ConfigError('Token counts, batch, and U must be positive integers: ' + r['id'])
        if r['microbatch_tokens'] < r['input_tokens'] or r['microbatch_tokens'] > r['input_tokens'] * r['batch']:
            raise ConfigError('The cohort features require P≤U≤P×B.')
    return (e, graph, warnings)
