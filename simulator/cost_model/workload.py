import bisect
import math
import struct

def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]

def split_axis(size, alignment, ratios, rotation=0):
    n = len(ratios)
    scan = []
    total = 0.0
    for i in range(n):
        total = f32(total + f32(ratios[(i + rotation) % n]))
        scan.append(total)
    out = [0] * n
    lo = 0
    for i in range(n - 1):
        hi = int(f32(f32(size * scan[i]) / total))
        hi -= hi % alignment
        out[(i + rotation) % n] = hi - lo
        lo = hi
    out[(n - 1 + rotation) % n] = size - lo
    return out

def layout(row, model):
    nodes, ratios = (row['nodes'], row['split'])
    if row['mode'] == 'tp':
        attention_alignment, ffn_alignment = (model.attention_split_alignment, model.ffn_split_alignment)
        if model.native_split_alignment:
            block = 32 if model.quantization == 'Q8_0' else 1
            group_width = model.attention_heads // model.kv_heads * model.head_dim
            while block < 128 and block * len(nodes) < group_width:
                block *= 2
            attention_alignment = math.lcm(group_width, block)
            ffn_alignment = 128
        q = [split_axis(model.attention_heads * model.head_dim, attention_alignment, ratios, l % len(nodes)) for l in range(model.layers)]
        ff = [split_axis(model.intermediate_size, ffn_alignment, ratios, l % len(nodes)) for l in range(model.layers)]
        output = split_axis(model.vocab_size, 1, ratios, model.layers % len(nodes))
        return dict(q=q, ff=ff, output=output)
    scan = []
    total = 0.0
    for r in ratios:
        total = f32(total + f32(r))
        scan.append(total)
    scan = [f32(r / total) for r in scan]
    assigned = [[] for _ in nodes]
    for l in range(model.layers + 1):
        i = 0 if row['mode'] == 'single' else bisect.bisect_right(scan, f32(l / (model.layers + 1)))
        assigned[i].append(l)
    return dict(stages=assigned)

class Work:

    def __init__(self, model, cls):
        self.model, self.cls = (model, cls)
        self.F = self.D = self.Fmat = self.Fsmall = self.Fmedium = self.Dmat = 0.0
        self.host = None

    def add(self, F, D):
        self.F += F
        self.D += D

    def mm(self, k, n, t):
        if not k or not n:
            return
        m = self.model
        F = 2 * k * n * t * m.weight_scale
        D = k * n * m.weight_bytes * m.weight_scale + m.activation_bytes * (k + n) * t
        if m.quantization == 'Q8_0':
            ae = k * t
            if self.cls in ('RPi4', 'RPi5'):
                F += 4 * ae + 2 * ae / 32 + 2 * (k / 32) * n * t
                D += 4 * ae + 2 * (ae / 32) * 34
            elif self.cls == 'Orin' or (self.cls == 'Xavier' and t < 64):
                F += 5 * ae + 3 * ae / 32 + 2 * (k / 32) * n * t
                qe = ae if t <= 8 else math.ceil(k / 128) * 128 * t
                D += 4 * ae + 2 * (qe / 32) * 36
            elif self.cls == 'Xavier':
                F += k * n * m.weight_scale + ae
                D += 4 * (k * n * m.weight_scale + ae)
        self.add(F, D)
        if self.cls in ('RPi4', 'RPi5') and t > 1 or (self.cls == 'Xavier' and t >= 64):
            self.Fmat += F
            self.Dmat += D
        if self.cls in ('RPi4', 'RPi5') and 1 < t <= 8:
            self.Fsmall += F
        if self.cls in ('RPi4', 'RPi5') and 8 < t < 64:
            self.Fmedium += F

    def norm(self, width, t, rows=None):
        rows = t if rows is None else rows
        self.add(4 * width * t + 3 * rows, 4 * self.model.activation_bytes * width * t + 4 * width)

    def attention(self, q, t, seqs, P, g, mask):
        m = self.model
        H = m.hidden_size
        h = m.head_dim
        a = m.activation_bytes
        self.norm(H, t)
        if not q:
            return
        kv = q * m.kv_heads / m.attention_heads
        self.mm(H, q, t)
        self.mm(H, kv, t)
        self.mm(H, kv, t)
        if m.qk_norm:
            self.norm(q, t, q / h * t)
            self.norm(kv, t, kv / h * t)
        pair = seqs * P * (P + 1) / 2 if g == 0 else t * (P + g)
        live = seqs * P if g == 0 else t * (P + g)
        self.add(4 * (q + kv) * t, 2 * a * (q + kv) * t + 8 * t)
        self.add(2 * kv * t, (2 * a + 2 * m.kv_bytes) * kv * t + 16 * t)
        self.add((4 * h + 5) * (q / h) * pair + q * t, 2 * a * q * t + 2 * m.kv_bytes * kv * live + m.kv_bytes * mask * t)
        self.mm(q, H, t)

    def ffn(self, width, t):
        H = self.model.hidden_size
        a = self.model.activation_bytes
        self.add(H * t, 3 * a * H * t)
        self.norm(H, t)
        if width:
            self.mm(H, width, t)
            self.mm(H, width, t)
            self.add(5 * width * t, 3 * a * width * t)
            self.mm(width, H, t)
