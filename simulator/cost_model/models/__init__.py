from dataclasses import dataclass, asdict
QUANTIZATION = {'F32': 4.0, 'F16': 2.0, 'BF16': 2.0, 'Q8_0': 34 / 32, 'Q4_0': 18 / 32, 'Q4_1': 20 / 32, 'Q5_0': 22 / 32, 'Q5_1': 24 / 32, 'Q2_K': 84 / 256, 'Q3_K': 110 / 256, 'Q4_K': 144 / 256, 'Q5_K': 176 / 256, 'Q6_K': 210 / 256, 'Q8_K': 292 / 256}

@dataclass(frozen=True)
class ModelSpec:
    name: str
    layers: int
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    quantization: str = 'Q8_0'
    parameter_count: int | None = None
    weight_bytes_per_parameter: float | None = None
    activation_bytes: float = 4.0
    kv_bytes: float = 2.0
    tie_word_embeddings: bool = False
    attention_split_alignment: int = 1
    ffn_split_alignment: int = 1
    family: str = 'qwen3'
    qk_norm: bool = True
    native_split_alignment: bool = False

    @property
    def weight_bytes(self):
        if self.weight_bytes_per_parameter is not None:
            return self.weight_bytes_per_parameter
        return QUANTIZATION[self.quantization]

    @property
    def shape_parameter_count(self):
        H, W, V = (self.hidden_size, self.intermediate_size, self.vocab_size)
        q, kv = (self.attention_heads * self.head_dim, self.kv_heads * self.head_dim)
        return self.layers * (H * (2 * q + 2 * kv) + 3 * H * W + 2 * H + (2 * self.head_dim if self.qk_norm else 0)) + H + H * V * (1 if self.tie_word_embeddings else 2)

    @property
    def weight_scale(self):
        return (self.parameter_count or self.shape_parameter_count) / self.shape_parameter_count

    @classmethod
    def from_dict(cls, value):
        return cls(**{k: v for k, v in value.items() if k in cls.__dataclass_fields__})

    def to_dict(self):
        return {**asdict(self), 'effective_weight_bytes_per_parameter': self.weight_bytes, 'shape_parameter_count': self.shape_parameter_count, 'weight_scale': self.weight_scale}
