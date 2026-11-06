import math
import numpy as np
WORKLOAD_COLUMNS = ('prompt_tokens', 'output_tokens', 'batch', 'requests', 'microbatch_tokens')
from .workload import Work, layout
from simulator.network.route_cost import Network
CLASSES = ('RPi4', 'RPi5', 'Orin', 'Xavier')

class BaseFormula:
    tier = 'base'
    version = 'base_compute_network_v1'
    profile = 'matrix'
    work_type = Work

    def __init__(self, rows, models, topologies):
        self.rows, self.models = (rows, models)
        self.networks = {name: Network(graph) for name, graph in topologies.items()}
        self.topology_classes = {name: {h: v['device_class'] for h, v in network.graph['nodes'].items() if v['kind'] == 'device'} for name, network in self.networks.items()}
        classes = {c for hosts in self.topology_classes.values() for c in hosts.values()}
        self.class_names = [c for c in CLASSES if c in classes] + sorted(classes - set(CLASSES))
        self.links = [f'{name}.{link}' for name, network in self.networks.items() for link in network.links]
        self.names = self.parameter_names()
        self.index = {n: i for i, n in enumerate(self.names)}
        self.work = []
        self.group_starts = []
        self.group_targets = []
        self.group_weights = []
        self.operations = []
        self.work_hosts = []
        self.work_classes = []
        self.host_names = [(name, h) for name, hosts in self.topology_classes.items() for h in hosts]
        self.network_templates = []
        self.network_uses = []
        self.template_keys = {}
        self.layouts = []
        for i, row in enumerate(rows):
            self.topology = row['topology']
            self.network = self.networks[self.topology]
            self.classes = self.topology_classes[self.topology]
            self._case(i, row, models[row['model']])
        self.work = np.array(self.work, dtype=float)
        self.group_starts = np.array(self.group_starts, dtype=int)
        self.group_targets = np.array(self.group_targets, dtype=int)
        self.group_weights = np.array(self.group_weights)
        self.group_ids = np.repeat(np.arange(len(self.group_starts)), np.diff(np.r_[self.group_starts, len(self.work)]))
        self.w_index = self.work[:, 6:12].astype(int)
        self.w_features = self.work[:, :6]
        self.uses = np.array(self.network_uses, dtype=float).reshape(-1, 3)
        self.operations = np.array(self.operations, dtype=float)

    def parameter_names(self):
        names = [c + '.' + term for c in self.class_names for term in ('x', 'y')]
        names += [c + '.x_matrix' for c in self.class_names if c in ('RPi4', 'RPi5', 'Xavier')]
        return names + [f'{link}.{term}' for link in self.links for term in ('alpha_ms', 'b_s_per_GB')]

    def work_vector(self, w, g):
        c = w.cls
        ix, iy = (self.index[c + '.x'], self.index[c + '.y'])
        im = self.index.get(c + '.x_matrix', ix)
        return ((w.F - w.Fmat) / 1000000000.0, w.D / 1000000000.0, w.Fmat / 1000000000.0, 0.0, 0.0, 0.0, ix, iy, im, im, im, iy)

    def group(self, works, target, weight, g):
        self.operations.append((0, len(self.group_starts), target, weight))
        self.group_starts.append(len(self.work))
        self.group_targets.append(target)
        self.group_weights.append(weight)
        for w in works:
            c = w.cls
            self.work.append(self.work_vector(w, g))
            self.work_hosts.append(self.host_names.index((self.topology, w.host)) if w.host is not None else -1)
            self.work_classes.append(self.class_names.index(c))

    def net(self, mode, nodes, size, target, weight, controller=None):
        if mode == 'tp' and len(nodes) == 1:
            return
        key = (self.topology, mode, tuple(nodes), int(size), controller)
        if key not in self.template_keys:
            template = []
            if mode == 'tp':
                rounds = self.network.tp_collective(list(nodes), int(size))['rounds']
                for rd in rounds:
                    resources = {}
                    for msg in rd['messages']:
                        for resource in msg['resources']:
                            resources.setdefault(resource, []).append(msg['cost_terms'])
                    template.append(list(resources.values()))
            elif mode == 'pp':
                for a, b in zip(nodes, nodes[1:]):
                    for msg in self.network.pp_copy(a, b, controller, int(size)):
                        if msg['link_ids']:
                            template.append([[msg['cost_terms']]])
            else:
                msg = self.network.cost(nodes[0], nodes[1], int(size))
                if msg['link_ids']:
                    template.append([[msg['cost_terms']]])
            compiled = []
            for candidates in template:
                compiled.append([[(np.array([self.index[self.topology + '.' + l + '.alpha_ms'] for l in term['alpha_link_ids']]), np.array([self.index[self.topology + '.' + l + '.b_s_per_GB'] for l in term['bottleneck_link_ids']]), term['M_bytes'] / 1000000000.0) for term in terms] for terms in candidates])
            self.template_keys[key] = len(self.network_templates)
            self.network_templates.append(compiled)
        self.network_uses.append((target, self.template_keys[key], weight))
        self.operations.append((1, self.template_keys[key], target, weight))

    def _case(self, i, row, m):
        H = m.hidden_size
        V = m.vocab_size
        a = m.activation_bytes
        P, G, B, R, U = (int(row[k]) for k in WORKLOAD_COLUMNS)
        N = B * R
        hosts = row['nodes']
        ctrl = row['controller']
        assigned = layout(row, m)
        self.layouts.append(assigned)
        for g in range(G):
            target = 2 * i + (g > 0)
            weight = 1 if g == 0 else 1 / (G - 1)
            chunks = [(s, min(U // P, N - s)) for s in range(0, N, U // P)] if g == 0 else [(0, N)]
            for start, seqs in chunks:
                t = P * seqs if g == 0 else seqs
                nt = seqs
                mask = math.ceil(((start + seqs) * P if g == 0 else N * (P + g)) / 256) * 256
                inp = self.work_type(m, self.classes[ctrl])
                inp.add(H * t, (H * m.weight_bytes * m.weight_scale + a * H) * t)
                self.group([inp], target, weight, g)
                if row['mode'] == 'tp':
                    for h in hosts:
                        self.net('message', [ctrl, h], round(a * H * t), target, weight)
                    for l in range(m.layers):
                        attention = []
                        feedforward = []
                        ft = nt if l == m.layers - 1 else t
                        for j, h in enumerate(hosts):
                            w = self.work_type(m, self.classes[h])
                            w.host = h
                            w.attention(assigned['q'][l][j], t, seqs, P, g, mask)
                            if l:
                                w.add(H * t, 3 * a * H * t)
                            attention.append(w)
                            w = self.work_type(m, self.classes[h])
                            w.host = h
                            w.ffn(assigned['ff'][l][j], ft)
                            if l == m.layers - 1:
                                w.add(0, 4 * a * H * nt + 8 * nt)
                            feedforward.append(w)
                        self.group(attention, target, weight, g)
                        self.net('tp', hosts, round(a * H * t), target, weight)
                        self.group(feedforward, target, weight, g)
                        self.net('tp', hosts, round(a * H * ft), target, weight)
                    head = []
                    for j, h in enumerate(hosts):
                        w = self.work_type(m, self.classes[h])
                        w.host = h
                        w.add(H * nt, 3 * a * H * nt)
                        w.norm(H, nt)
                        w.mm(H, assigned['output'][j], nt)
                        head.append(w)
                    self.group(head, target, weight, g)
                    for j, h in enumerate(hosts):
                        self.net('message', [h, ctrl], round(a * assigned['output'][j] * nt), target, weight)
                else:
                    active = [j for j, ll in enumerate(assigned['stages']) if ll]
                    previous = None
                    for j in active:
                        h = hosts[j]
                        ll = assigned['stages'][j]
                        if previous is None:
                            self.net('message', [ctrl, h], round(a * H * t), target, weight)
                        else:
                            size = round(a * H * (nt if m.layers - 1 in assigned['stages'][previous] else t))
                            self.net('pp', [hosts[previous], h], size, target, weight, ctrl)
                        w = self.work_type(m, self.classes[h])
                        w.host = h
                        for l in ll:
                            if l == m.layers:
                                w.norm(H, nt)
                                w.mm(H, V, nt)
                            else:
                                ft = nt if l == m.layers - 1 else t
                                w.attention(m.attention_heads * m.head_dim, t, seqs, P, g, mask)
                                w.ffn(m.intermediate_size, ft)
                                w.add(H * ft, 3 * a * H * ft)
                                if l == m.layers - 1:
                                    w.add(0, 4 * a * H * nt + 8 * nt)
                        self.group([w], target, weight, g)
                        previous = j
                    self.net('message', [hosts[active[-1]], ctrl], round(a * V * nt), target, weight)
            sample = self.work_type(m, self.classes[ctrl])
            sample.add(N * (V - 1), N * (a * V + 4))
            if g:
                sample.add(0, 24 * N)
            self.group([sample], target, weight, g)
