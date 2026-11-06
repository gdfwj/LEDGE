from collections import deque
from simulator.network.route_cost import Network
from simulator.network.mpi_accounting import collective

class ExecutionNetwork(Network):

    def tp_collective(self, nodes, M):
        if type(M) is not int or M <= 0 or M % 4 or (len(nodes) < 2) or (len(set(nodes)) != len(nodes)):
            raise ValueError('TP requires distinct ranks and a positive F32 payload')
        plan = collective(nodes, M // 4, max_ranks=None)
        rounds = [{'name': p['name'], 'messages': [dict(self.cost(m['source'], m['destination'], m['M']), source_rank=m['source_rank'], destination_rank=m['destination_rank'], offset_elements=m['offset_elements']) for m in p['messages'] if m['M'] > 0]} for p in plan['phases'] if p['messages']]
        return {'algorithm': plan['algorithm'], 'rank_order': nodes, 'rounds': rounds, 'total_tx_user_bytes': sum((m['M_bytes'] for rd in rounds for m in rd['messages']))}

def resolve_routes(graph, groups):
    network = Network(graph)
    routes = graph.setdefault('routes', {})
    for group in groups:
        endpoints = set(group['devices']) | {group['controller']}
        for src in sorted(endpoints):
            for dst in sorted(endpoints):
                if src == dst:
                    continue
                key = src + '->' + dst
                if key in routes:
                    network.path(src, dst)
                    continue
                queue = deque([[src]])
                seen = {src}
                found = None
                while queue:
                    path = queue.popleft()
                    node = path[-1]
                    if node == dst:
                        found = path
                        break
                    for nxt in sorted(network.adj[node]):
                        if nxt not in seen and (nxt == dst or graph['nodes'][nxt]['kind'] != 'device'):
                            seen.add(nxt)
                            queue.append(path + [nxt])
                if found is None:
                    raise ValueError('Deployment endpoints are disconnected: ' + key)
                routes[key] = found
    return graph
