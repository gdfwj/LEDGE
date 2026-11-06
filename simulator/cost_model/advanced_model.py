import numpy as np
from .base_model import BaseFormula
from .advanced_transport import collective_rounds, policy_for_row
from .workload import Work
PROFILES = {'matrix_shape_pipeline_plain': ('matrix', ('small', 'medium', 'transport', 'overlap', 'broadcast_overlap'))}

class AdvancedWork(Work):

    def __init__(self, model, cls):
        super().__init__(model, cls)
        self.Ffloat_large = self.Ffloat_medium = 0.0
        self.Dfloat_mmvf = 0.0
        self.Ffloat_xavier_mid = 0.0

    def mm(self, k, n, t):
        before = self.F
        super().mm(k, n, t)
        if self.model.quantization == 'F16' and self.cls == 'Xavier' and (3 < t < 64):
            self.Ffloat_xavier_mid += self.F - before
        if self.model.quantization == 'F16' and self.cls in ('RPi4', 'RPi5') and (t > 8):
            if t < 64:
                self.Ffloat_medium += self.F - before
            else:
                self.Ffloat_large += self.F - before
        if self.model.quantization == 'F16' and (self.cls == 'Orin' and t == 1 or (self.cls == 'Xavier' and t <= 3)):
            self.Dfloat_mmvf += k * n * self.model.weight_bytes * self.model.weight_scale

class AdvancedFormula(BaseFormula):
    tier = 'advanced'
    version = 'advanced_compute_network_v1'
    work_type = AdvancedWork

    def __init__(self, rows, models, topologies, profile='matrix_shape_pipeline_plain'):
        self.profile = profile
        self.base_profile, self.flags = PROFILES[profile]
        self.response = bool(set(self.flags) & {'cpu', 'gpu', 'cpu4', 'overlap'})
        self.small_matrix = 'small' in self.flags
        self.medium_matrix = 'medium' in self.flags
        self.matrix_memory = 'memory' in self.flags
        self.transport_updated = 'transport' in self.flags
        self.float_cpu_matrix = 'float_cpu_matrix' in self.flags
        self.float_gpu_mmvf = 'float_gpu_mmvf' in self.flags
        self.overlap_gpu_reduce = 'overlap' in self.flags
        self.overlap_gpu_broadcast = 'broadcast_overlap' in self.flags
        self.float_xavier_dispatch = 'float_xavier_dispatch' in self.flags
        self.group_network = {}
        self.collective_round_cache = {}
        if self.transport_updated:
            self.version = 'advanced_compute_network_v2'
        if self.float_cpu_matrix:
            self.version = 'advanced_compute_network_v3'
        if self.float_gpu_mmvf:
            self.version = 'advanced_compute_network_v4'
        if self.overlap_gpu_reduce:
            self.version = 'advanced_compute_network_v5'
        if self.overlap_gpu_broadcast:
            self.version = 'advanced_compute_network_v6'
        if self.float_xavier_dispatch:
            self.version = 'advanced_compute_network_v7'
        super().__init__(rows, models, topologies)

    def _case(self, i, row, m):
        self.collective_policy = policy_for_row(row) if self.transport_updated else 'native_mpi'
        super()._case(i, row, m)

    def net(self, mode, nodes, size, target, weight, controller=None):
        if not self.transport_updated or mode != 'tp' or len(nodes) < 2:
            return super().net(mode, nodes, size, target, weight, controller)
        key = (self.topology, mode, tuple(nodes), int(size), controller, self.collective_policy)
        if key not in self.collective_round_cache:
            self.collective_round_cache[key] = collective_rounds(self.network, list(nodes), int(size), self.classes, self.collective_policy)
        rounds = self.collective_round_cache[key]
        gpu = [h for h in nodes if self.classes[h] in ('Orin', 'Xavier')]
        pending = None
        if self.overlap_gpu_reduce and self.collective_policy == 'nccl_hierarchical' and (1 < len(gpu) < len(nodes)):
            cut = next((i for i, rd in enumerate(rounds) if not rd['name'].startswith('nccl_')))
            assert self.operations[-1][0] == 0
            self.group_network[self.operations[-1][1]] = self.compile_rounds(key + ('gpu_reduce',), rounds[:cut])
            rounds = rounds[cut:]
            key += ('bridge_broadcast',)
            if self.overlap_gpu_broadcast:
                cut = next((i for i, rd in enumerate(rounds) if rd['name'].startswith('nccl_')))
                pending = self.compile_rounds(key + ('deferred_broadcast',), rounds[cut:])
                rounds = rounds[:cut]
                key += ('mpi_only',)
        template_id = self.compile_rounds(key, rounds)
        self.network_uses.append((target, template_id, weight))
        self.operations.append((1, template_id, target, weight))
        if pending is not None:
            self.operations.append((2, pending, target, weight))

    def compile_rounds(self, key, rounds):
        if key not in self.template_keys:
            compiled = []
            for rd in rounds:
                resources = {}
                for msg in rd['messages']:
                    for resource in msg['resources']:
                        resources.setdefault(resource, []).append(msg['cost_terms'])
                compiled.append([[(np.array([self.index[self.topology + '.' + l + '.alpha_ms'] for l in term['alpha_link_ids']]), np.array([self.index[self.topology + '.' + l + '.b_s_per_GB'] for l in term['bottleneck_link_ids']]), term['M_bytes'] / 1000000000.0) for term in terms] for terms in resources.values()])
            self.template_keys[key] = len(self.network_templates)
            self.network_templates.append(compiled)
        return self.template_keys[key]

    def parameter_names(self):
        names = []
        flags = self.flags
        for c in self.class_names:
            if self.base_profile == 'phase':
                names += [f'{c}.{phase}.{term}' for phase in ('prefill', 'decode') for term in ('x', 'y')]
            else:
                names += [c + '.x', c + '.y']
        if self.base_profile == 'matrix':
            names += [c + '.x_matrix' for c in self.class_names if c in ('RPi4', 'RPi5', 'Xavier')]
        if self.small_matrix:
            names += [c + '.x_small_matrix' for c in self.class_names if c in ('RPi4', 'RPi5')]
        if self.medium_matrix:
            names += [c + '.x_medium_matrix' for c in self.class_names if c in ('RPi4', 'RPi5')]
        if self.matrix_memory:
            names += [c + '.y_matrix' for c in self.class_names if c in ('RPi4', 'RPi5', 'Xavier')]
        if self.float_cpu_matrix:
            names += [c + '.x_f16_matrix' for c in self.class_names if c in ('RPi4', 'RPi5')]
        if self.float_gpu_mmvf:
            names += [c + '.y_f16_mmvf' for c in self.class_names if c in ('Orin', 'Xavier')]
        names += [f'{l}.{term}' for l in self.links for term in ('alpha_ms', 'b_s_per_GB')]
        if 'cpu' in flags:
            names += ['RPi5.idle_slowdown', 'RPi5.util_tau_ms']
        if 'gpu' in flags:
            names += ['Orin.wake_ms', 'Xavier.wake_ms', 'GPU.idle_tau_ms']
        if 'cpu4' in flags:
            names += ['RPi4.idle_slowdown', 'RPi4.util_tau_ms']
        return names

    def work_vector(self, w, g):
        c = w.cls
        if self.base_profile == 'phase':
            stem = c + ('.prefill' if g == 0 else '.decode')
            ix, iy = (self.index[stem + '.x'], self.index[stem + '.y'])
            im = ix
            Fm = 0
        else:
            ix, iy = (self.index[c + '.x'], self.index[c + '.y'])
            im = self.index.get(c + '.x_matrix', ix)
            Fm = w.Fmat if self.base_profile == 'matrix' else 0
        if self.float_xavier_dispatch and self.base_profile == 'matrix':
            Fm += w.Ffloat_xavier_mid
        Fs = w.Fsmall if self.small_matrix else 0
        Fmid = w.Fmedium if self.medium_matrix else 0
        ism = self.index.get(c + '.x_small_matrix', im)
        imm = self.index.get(c + '.x_medium_matrix', im)
        Dm = w.Dmat if self.matrix_memory else 0
        idm = self.index.get(c + '.y_matrix', iy)
        vector = [(w.F - Fm) / 1000000000.0, (w.D - Dm) / 1000000000.0, (Fm - Fs - Fmid) / 1000000000.0, Fs / 1000000000.0, Fmid / 1000000000.0, Dm / 1000000000.0, ix, iy, im, ism, imm, idm]
        if self.float_cpu_matrix and w.model.quantization == 'F16' and (c in ('RPi4', 'RPi5')):
            vector[2] -= w.Ffloat_large / 1000000000.0
            vector[4] -= w.Ffloat_medium / 1000000000.0
            vector[5] = (w.Ffloat_large + w.Ffloat_medium) / 1000000000.0
            vector[11] = self.index[c + '.x_f16_matrix']
        if self.float_gpu_mmvf and c in ('Orin', 'Xavier'):
            vector[1] -= w.Dfloat_mmvf / 1000000000.0
            vector[5] = w.Dfloat_mmvf / 1000000000.0
            vector[11] = self.index[c + '.y_f16_mmvf']
        return tuple(vector)
