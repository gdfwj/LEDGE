from .advanced_model import AdvancedFormula, AdvancedWork

class ShapeWork(AdvancedWork):

    def __init__(self, model, cls):
        super().__init__(model, cls)
        self.Fsmall_bulk = 0.0

    def mm(self, k, n, t):
        before = self.F
        super().mm(k, n, t)
        if self.cls == 'RPi5' and 2 < t <= 8:
            fraction = (0.5 - 1 / t) / (0.5 - 1 / 8)
            self.Fsmall_bulk += (self.F - before) * fraction

class ShapeWork4(AdvancedWork):

    def __init__(self, model, cls):
        super().__init__(model, cls)
        self.Fsmall_bulk = 0.0

    def mm(self, k, n, t):
        before = self.F
        super().mm(k, n, t)
        if self.cls == 'RPi5' and 4 < t <= 8:
            self.Fsmall_bulk += (self.F - before) * (0.25 - 1 / t) / (0.25 - 0.125)

class ShapeFormula(AdvancedFormula):
    work_type = ShapeWork

    def __init__(self, rows, models, topologies, profile='small_amortized'):
        if profile not in ('small_blend', 'small_amortized', 'small4_amortized', 'small4_amortized_graph_dispatch'):
            raise ValueError(profile)
        self.shape_profile = profile
        self.graph_dispatch = profile == 'small4_amortized_graph_dispatch'
        if profile.startswith('small4_amortized'):
            self.work_type = ShapeWork4
        super().__init__(rows, models, topologies, 'matrix_shape_pipeline_plain')
        self.profile = profile
        self.version = 'advanced_compute_network_shape_v1'
        if self.graph_dispatch:
            self.w_index = self.work[:, 7:14].astype(int)
            self.w_features = self.work[:, :7]
            self.version = 'advanced_compute_network_shape_graph_v1'

    def parameter_names(self):
        names = super().parameter_names()
        if self.shape_profile in ('small_amortized', 'small4_amortized', 'small4_amortized_graph_dispatch') and 'RPi5' in self.class_names:
            names += ['RPi5.x_small_bulk']
        if self.graph_dispatch:
            names += [c + '.graph_dispatch_ms' for c in self.class_names]
        return names

    def work_vector(self, w, g):
        v = list(super().work_vector(w, g))
        if w.cls == 'RPi5':
            feature = w.Fsmall_bulk / 1000000000.0
            v[3] -= feature
            if self.shape_profile == 'small_blend':
                v[4] += feature
            else:
                v[5] = feature
                v[11] = self.index['RPi5.x_small_bulk']
        if self.graph_dispatch:
            return tuple(v[:6] + [0.001 if w.host is not None else 0.0] + v[6:] + [self.index[w.cls + '.graph_dispatch_ms']])
        return tuple(v)
