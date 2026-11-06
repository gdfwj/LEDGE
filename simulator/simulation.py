import math, time
from .config import validate, load_bundle, ConfigError
from .adapter import prepare_requests
from .general_core import GeneralSimulator

def summarize(rows):

    def mean(rr, key):
        return sum((r[key] for r in rr)) / len(rr) if rr else None

    def group(rr):
        return {'requests': len(rr), 'tpot_requests': sum((r['tpot_s'] is not None for r in rr)), 'mean_service_s': mean(rr, 'service_s'), 'mean_response_s': mean(rr, 'response_s'), 'mean_queue_s': mean(rr, 'queue_s'), 'mean_ttft_s': mean(rr, 'ttft_s'), 'mean_tpot_s': mean([r for r in rr if r['tpot_s'] is not None], 'tpot_s')}
    return {'all': group(rows), 'instances': {id: group([r for r in rows if r['instance'] == id]) for id in sorted({r['instance'] for r in rows})}, 'models': {id: group([r for r in rows if r['model'] == id]) for id in sorted({r['model'] for r in rows})}}

def validate_interference(p):
    for key in ('quantum_bytes', 'eager_bytes', 'Orin_weight', 'Xavier_weight', 'prefill_gain', 'decode_gain'):
        if not math.isfinite(p[key]) or p[key] < 0:
            raise ConfigError('Invalid communication model parameter: ' + key)
    if min(p['quantum_bytes'], p['Orin_weight'], p['Xavier_weight']) <= 0:
        raise ConfigError('Service quantum and weights must be positive.')
    if p['readiness'] not in (0, 1):
        raise ConfigError('readiness must be 0 or 1.')

def run_simulation(experiment, mode='fast'):
    if mode not in ('fast', 'log'):
        raise ConfigError('Run mode must be fast or log.')
    started = time.monotonic()
    exp, graph, warnings = validate(experiment)
    mapping, artifacts, interference, provenance = load_bundle(exp)
    p = interference['parameters']
    validate_interference(p)
    jobs, resources, bindings = prepare_requests(exp, graph, mapping, artifacts)
    engine = GeneralSimulator(jobs, resources, p, mode == 'log')
    rows, log = engine.run()
    result = {'schema': 2, 'engine': 'general_dag', 'mode': mode, 'input': exp, 'requests': rows, 'summary': summarize(rows), 'warnings': warnings, 'resolved': {'parameter_mapping': mapping, 'artifacts': artifacts, 'interference_artifact': interference, 'provenance': provenance, 'shared_resources': resources, 'requests': bindings, 'graph': graph}, 'duration_s': max((r['completion_s'] for r in rows), default=0.0), 'engine_event_count': log['engine_event_count']}
    if mode == 'log':
        result['log'] = log
    result['wall_time_s'] = time.monotonic() - started
    return result
