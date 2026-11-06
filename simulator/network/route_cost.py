import math
from .mpi_accounting import collective

class Network:

    def __init__(self, graph):
        self.graph = graph
        self.adj = {n: [] for n in graph['nodes']}
        self.edges = {}
        self.links = {}
        for link in graph['links']:
            a, b = link['endpoints']
            if a == b or link['id'] in self.links or (a, b) in self.edges:
                raise ValueError('duplicate link/endpoints or self-loop')
            if link['duplex'] not in ('full', 'half'):
                raise ValueError('duplex must be full or half')
            for k in ('alpha_s', 'B_bytes_per_s'):
                v = link[k]
                if v is not None and (not math.isfinite(v) or v < 0 or (k == 'B_bytes_per_s' and v == 0)):
                    raise ValueError('invalid link parameter: ' + k)
            self.links[link['id']] = link
            for src, dst in ((a, b), (b, a)):
                self.adj[src].append(dst)
                self.edges[src, dst] = link['id']

    def path(self, source, target):
        if source not in self.adj or target not in self.adj:
            raise ValueError('unknown endpoint')
        if source == target:
            return [source]
        override = self.graph.get('routes', {}).get(source + '->' + target)
        if override is not None:
            if not override or override[0] != source or override[-1] != target or (len(set(override)) != len(override)) or any(((a, b) not in self.edges for a, b in zip(override, override[1:]))) or any((self.graph['nodes'][n]['kind'] == 'device' for n in override[1:-1])):
                raise ValueError('invalid explicit route')
            return override
        paths = []

        def search(nodes):
            if nodes[-1] == target:
                paths.append(nodes)
                return
            for nxt in sorted(self.adj[nodes[-1]]):
                if len(paths) == 2:
                    return
                if nxt not in nodes and (nxt == target or self.graph['nodes'][nxt]['kind'] != 'device'):
                    search(nodes + [nxt])
        search([source])
        if len(paths) != 1:
            raise ValueError('disconnected endpoints' if not paths else 'multiple physical paths; specify routes[source->target]')
        return paths[0]

    def cost(self, source, target, M=None):
        if M is not None and (type(M) is not int or M < 0):
            raise ValueError('M must be a nonnegative integer byte count or None')
        nodes = self.path(source, target)
        ids = [self.edges[a, b] for a, b in zip(nodes, nodes[1:])]
        links = [self.links[i] for i in ids]
        known = all((l['alpha_s'] is not None and l['B_bytes_per_s'] is not None for l in links))
        seconds = 0.0 if not links else sum((l['alpha_s'] for l in links)) + M / min((l['B_bytes_per_s'] for l in links)) if known and M is not None else None
        resources = ['link:' + a + '->' + b if l['duplex'] == 'full' else 'link-half:' + l['id'] for a, b, l in zip(nodes, nodes[1:], links)]
        return {'source': source, 'target': target, 'path': nodes, 'link_ids': ids, 'resources': resources, 'M_bytes': M, 'cost_s': seconds, 'formula': '0' if not ids else ' + '.join(('alpha[' + i + ']' for i in ids)) + ' + ' + (str(M) if M is not None else 'M') + '/min(' + ','.join(('B[' + i + ']' for i in ids)) + ')', 'cost_terms': {'alpha_link_ids': ids, 'bottleneck_link_ids': ids, 'M_bytes': M}}

    def group(self, nodes, M=None):
        if not nodes or len(nodes) != len(set(nodes)) or (not set(nodes) <= self.adj.keys()):
            raise ValueError('group must contain distinct endpoints')
        return [self.cost(a, b, M) for a in nodes for b in nodes if a != b]

    def pp_copy(self, source, target, controller, M):
        if type(M) is not int or M < 0:
            raise ValueError('M must be nonnegative integer bytes')
        if source == target or M == 0:
            return []
        return [dict(self.cost(controller, source, 321), command='GET_request'), dict(self.cost(source, controller, M + 8), command='GET_reply'), dict(self.cost(controller, target, M + 314), command='SET')]

    def tp_collective(self, nodes, M):
        if type(M) is not int or M <= 0 or M % 4 or (not 2 <= len(nodes) <= 10) or (len(set(nodes)) != len(nodes)) or (not set(nodes) <= self.adj.keys()):
            raise ValueError('bound MPI generator requires 2..10 ranks and positive F32 bytes divisible by 4')
        plan = collective(nodes, M // 4)
        rounds = [{'name': phase['name'], 'messages': [dict(self.cost(m['source'], m['destination'], m['M']), source_rank=m['source_rank'], destination_rank=m['destination_rank'], offset_elements=m['offset_elements']) for m in phase['messages']]} for phase in plan['phases'] if phase['messages']]
        return {'algorithm': plan['algorithm'], 'rank_order': nodes, 'rounds': rounds, 'total_tx_user_bytes': sum((m['M_bytes'] for r in rounds for m in r['messages'])), 'scope': 'MPI logical messages only. Rounds are rank-local, not added global barriers. Not a collective elapsed-time prediction.'}
