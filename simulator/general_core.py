import heapq
from collections import defaultdict

class GeneralSimulator:

    def __init__(self, jobs, resources, parameters, logging=False):
        self.jobs = jobs
        self.resources = resources
        self.p = parameters
        self.logging = logging
        self.intervals = []
        self.points = []
        self.events = []
        self.serial = 0
        self.count = 0
        self.pending = []
        self.busy = {}
        self.messages = {}
        self.message_id = 0
        self.clock = 0.0
        self.follow = {}
        self.states = []
        bygroup = defaultdict(list)
        for r, j in enumerate(jobs):
            from .general_adapter import compile_dag
            tasks = compile_dag(j)
            successors = [[] for t in tasks]
            for i, t in enumerate(tasks):
                for dep in t['deps']:
                    successors[dep].append(i)
            self.states.append({'tasks': tasks, 'successors': successors, 'left': [len(t['deps']) for t in tasks], 'done': {}, 'began': {}, 'tokens': [], 'start': None, 'end': None, 'rounds': {}})
            bygroup[j['request']['instance']].append(r)
            self.point(j['request']['arrival_s'], 'arrival', r)
        for ids in bygroup.values():
            ids.sort(key=lambda i: (jobs[i]['request']['arrival_s'], i))
            self.follow.update(zip(ids, ids[1:]))
            self.push(jobs[ids[0]]['request']['arrival_s'], -1, ids[0])

    def point(self, t, kind, r, **extra):
        if self.logging:
            self.points.append({'time': t, 'kind': kind, 'request': self.jobs[r]['request']['id'], 'instance': self.jobs[r]['request']['instance'], **extra})

    def interval(self, start, end, kind, r, lane, **extra):
        if self.logging and end > start + 1e-15:
            self.intervals.append({'id': len(self.intervals), 'start': start, 'end': end, 'kind': kind, 'request': self.jobs[r]['request']['id'], 'instance': self.jobs[r]['request']['instance'], 'lane': lane, **extra})

    def record_compute(self, r, op, now):
        if not self.logging:
            return
        for rank in op['ranks']:
            lane = 'device:' + rank['device']
            common = {'device': rank['device'], 'rank': rank['rank'], 'stage': rank['stage'], 'token': op['token'], 'microbatch': op['microbatch']}
            if rank['segments']:
                t = now
                for seg in rank['segments']:
                    self.interval(t, t + seg['duration'], 'compute', r, lane, work=seg['kind'], layer=seg['layer'], **common)
                    t += seg['duration']
            else:
                self.interval(now, now + rank['duration'], 'compute', r, lane, work=op['work'], layer=op['layer'], **common)

    def push(self, t, kind, r, n=-1, m=-1):
        self.serial += 1
        heapq.heappush(self.events, (t, kind, r, n, m, self.serial))

    def complete_node(self, r, n, now):
        s = self.states[r]
        t = s['tasks'][n]
        if n in s['done']:
            raise RuntimeError('DAG node completed twice')
        s['done'][n] = now
        if t['kind'] == 'token':
            s['tokens'].append(now)
            self.point(now, 'token', r, token=t['token'])
        for work in t.get('sync', []):
            op = s['tasks'][work]['op']
            rank = op['ranks'][0]
            self.interval(s['done'][work], now, 'synchronization', r, 'device:' + rank['device'], device=rank['device'], rank=rank['rank'], stage=rank['stage'], token=t['token'], layer=t.get('layer'), reason='Compute / collective synchronization')
        for nxt in s['successors'][n]:
            s['left'][nxt] -= 1
            if s['left'][nxt] == 0:
                self.push(now, 0, r, nxt)
        if len(s['done']) == len(s['tasks']):
            s['end'] = now
            self.point(now, 'complete', r)
            if r in self.follow:
                nxt = self.follow[r]
                self.push(max(now, self.jobs[nxt]['request']['arrival_s']), -1, nxt)

    def advance_wait(self, now):
        delta = now - self.clock
        if delta > 0:
            earlier = []
            for mid, queued in self.pending:
                m = self.messages[mid]
                instance = self.jobs[m['r']]['request']['instance']
                qs = set(m['spec']['resources'])
                blockers = [self.messages[self.busy[q]] for q in qs if q in self.busy] + [self.messages[v] for v in earlier if qs.intersection(self.messages[v]['spec']['resources'])]
                if any((self.jobs[x['r']]['request']['instance'] != instance for x in blockers)):
                    m['external_wait'] += delta
                earlier.append(mid)
        self.clock = now

    def dispatch(self, now):
        remaining = []
        blocked = set()
        for mid, queued in self.pending:
            m = self.messages[mid]
            spec = m['spec']
            qs = spec['resources']
            if any((q in self.busy or q in blocked for q in qs)):
                remaining.append((mid, queued))
                blocked.update(qs)
                continue
            weight = {'RPi4': 1.0, 'RPi5': 1.0, 'Orin': self.p['Orin_weight'], 'Xavier': self.p['Xavier_weight']}[spec['source_class']]
            amount = min(m['remaining'], self.p['quantum_bytes'] * weight)
            if m['sent'] < self.p['eager_bytes'] - 1e-09 and now < m['ready']:
                amount = min(amount, self.p['eager_bytes'] - m['sent'])
            bandwidth = min((self.resources[q.split(':')[0]]['bandwidth'] for q in qs))
            duration = amount / bandwidth
            m['slice'] = amount
            for q in qs:
                self.busy[q] = mid
                common = {'resource': q, 'link': q.split(':')[0], 'source': spec['source'], 'target': spec['target'], 'path': spec['path'], 'links': spec['links'], 'message': mid, 'token': m['task']['token'], 'layer': m['task'].get('layer'), 'work': m['task']['work']}
                self.interval(queued, now, 'queue_wait', m['r'], 'link:' + q, reason='Waiting for shared network resources', **common)
                self.interval(now, now + duration, 'communication', m['r'], 'link:' + q, bytes=amount, route_reservation=True, **common)
            self.push(now + duration, 2, m['r'], m['node'], mid)
        self.pending = remaining

    def start_node(self, r, n, now):
        s = self.states[r]
        t = s['tasks'][n]
        s['began'][n] = now
        if t['kind'] == 'work':
            self.record_compute(r, t['op'], now)
            nominal = t['nominal_start']
            start = s['done'][nominal] if nominal is not None else s['start']
            rank = t['op']['ranks'][0]
            self.interval(start, now, 'synchronization', r, 'device:' + rank['device'], device=rank['device'], rank=rank['rank'], stage=rank['stage'], token=t['op']['token'], layer=t['op']['layer'], reason='Waiting for GPU broadcast')
            self.push(now + t['duration'], 5, r, n)
        elif t['kind'] == 'round':
            s['rounds'][n] = {'pending': len(t['messages']), 'floor': now + t['duration']}
            barrier = max((max(m['source_ready'], m['target_ready']) for m in t['messages']))
            for spec in t['messages']:
                mid = self.message_id
                self.message_id += 1
                src = spec['source_ready'] if self.p['readiness'] else barrier
                dst = spec['target_ready'] if self.p['readiness'] else barrier
                m = {'spec': spec, 'r': r, 'node': n, 'task': t, 'remaining': float(spec['bytes']), 'sent': 0.0, 'external_wait': 0.0, 'ready': now + dst}
                self.messages[mid] = m
                release = now + src + spec['private']
                self.interval(now + src, release, 'private_path', r, 'path:' + spec['source'] + '→' + spec['target'], source=spec['source'], target=spec['target'], path=spec['path'], links=spec['links'], token=t['token'], layer=t.get('layer'), reason='Path startup and non-shared transfer')
                self.push(release, 1 if spec['resources'] else 3, r, n, mid)
        else:
            self.complete_node(r, n, now)

    def step(self):
        if not self.events:
            return None
        due, kind, r, n, mid, _ = heapq.heappop(self.events)
        now = due
        self.advance_wait(now)
        self.count += 1
        if self.count > 50000000:
            raise RuntimeError('Event budget exceeded; reduce the input size or output length.')
        s = self.states[r]
        if kind == -1:
            s['start'] = now
            self.point(now, 'start', r)
            self.interval(self.jobs[r]['request']['arrival_s'], now, 'request_queue', r, 'request:' + self.jobs[r]['request']['id'], reason='Waiting for the previous request')
            for i, t in enumerate(s['tasks']):
                if not t['deps']:
                    self.push(now, 0, r, i)
        elif kind == 0:
            self.start_node(r, n, now)
        elif kind in (1, 4):
            m = self.messages[mid]
            if m['sent'] >= self.p['eager_bytes'] - 1e-09 and now < m['ready']:
                self.interval(now, m['ready'], 'receiver_wait', r, 'path:' + m['spec']['source'] + '→' + m['spec']['target'], source=m['spec']['source'], target=m['spec']['target'], token=m['task']['token'], layer=m['task'].get('layer'), reason='Waiting for receiver after eager transfer')
                self.push(m['ready'], 4, r, n, mid)
            else:
                self.pending.append((mid, now))
        elif kind == 2:
            m = self.messages[mid]
            for q in m['spec']['resources']:
                del self.busy[q]
            m['remaining'] -= m['slice']
            m['sent'] += m['slice']
            if m['remaining'] > 1e-07:
                self.push(now, 4, r, n, mid)
            else:
                gain = self.p['prefill_gain'] if m['task']['token'] == 0 else self.p['decode_gain']
                ready = max(now, m['ready'])
                end = ready + gain * m['external_wait']
                common = {'source': m['spec']['source'], 'target': m['spec']['target'], 'token': m['task']['token'], 'layer': m['task'].get('layer'), 'message': mid}
                self.interval(now, ready, 'receiver_wait', r, 'path:' + common['source'] + '→' + common['target'], reason='Waiting for receiver readiness', **common)
                self.interval(ready, end, 'progress_wait', r, 'path:' + common['source'] + '→' + common['target'], reason='Phase-scaled cross-instance contention delay', **common)
                self.push(end, 3, r, n, mid)
        elif kind == 3:
            round = s['rounds'][n]
            round['pending'] -= 1
            del self.messages[mid]
            if round['pending'] == 0:
                end = max(now, round['floor'])
                t = s['tasks'][n]
                self.interval(now, end, 'floor_wait', r, 'instance:' + self.jobs[r]['request']['instance'], token=t['token'], layer=t.get('layer'), reason='Collective critical-path completion')
                self.push(end, 5, r, n)
        else:
            self.complete_node(r, n, now)
        self.dispatch(now)

    def run(self):
        while self.events:
            self.step()
        return self.result()

    def result(self):
        rows = []
        for j, s in zip(self.jobs, self.states):
            r = j['request']
            tokens = s['tokens']
            if s['end'] is None or len(tokens) != r['output_tokens']:
                raise RuntimeError('Incomplete DAG or missing token boundary')
            first, last = (tokens[0], tokens[-1])
            start = s['start']
            arrival = r['arrival_s']
            rows.append({'request_id': r['id'], 'instance': r['instance'], 'model': j['group']['model'], 'arrival_s': arrival, 'start_s': start, 'first_token_s': first, 'completion_s': last, 'queue_s': start - arrival, 'service_s': last - start, 'response_s': last - arrival, 'ttft_s': first - arrival, 'tpot_s': (last - first) / (len(tokens) - 1) if len(tokens) > 1 else None, 'token_times_s': tokens, 'output_tokens': len(tokens)})
        return (rows, {'intervals': self.intervals, 'events': sorted(self.points, key=lambda x: x['time']), 'engine_event_count': self.count})
