"""Chinese span extraction with schema-scoped attributes and document decisions.

This separate experiment model does not change existing service checkpoints.
"""
import math
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


def unfreeze_encoder_tail(encoder, count=4):
    """Use the same last-N-layer policy for BERT and mmBERT encoders."""
    if hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layer'):
        layers = encoder.encoder.layer
        path = 'encoder.layer'
    elif hasattr(encoder, 'layers'):
        layers = encoder.layers
        path = 'layers'
    else:
        raise ValueError('Unsupported encoder layer layout')
    if not 0 < count <= len(layers):
        raise ValueError('Trainable encoder layer count exceeds available layers')
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    for layer in layers[-count:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    return {'layer_path': path, 'total_layers': len(layers),
            'trainable_layers': list(range(len(layers) - count, len(layers))),
            'policy': f'last_{count}_layers_only; embeddings and other encoder modules frozen'}


def save_schema_checkpoint(model, path, base_file_hashes=None):
    """Store a frozen-base delta when requested, otherwise preserve legacy state dicts."""
    state = model.state_dict()
    if base_file_hashes is not None:
        mutable = {name for name, p in model.named_parameters() if p.requires_grad}
        buffers = {name for name, _ in model.named_buffers()}
        state = {'format': 'frozen_encoder_delta_v1', 'base_file_hashes': base_file_hashes,
                 'updated_encoder_keys': sorted(name for name in mutable if name.startswith('encoder.')),
                 'state_dict': {name: value.detach().cpu() for name, value in state.items()
                                if not name.startswith('encoder.') or name in mutable or name in buffers}}
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_schema_checkpoint(model, path, device, base_path):
    """Strictly reconstruct the model; a compact checkpoint requires the exact base."""
    saved = torch.load(path, map_location=device, weights_only=True)
    compact = saved.get('format') == 'frozen_encoder_delta_v1'
    if compact:
        for name, expected in saved['base_file_hashes'].items():
            h = hashlib.sha256()
            with (Path(base_path) / name).open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError(f'Compact checkpoint base file mismatch: {name}')
        updates = saved['state_dict']
        required = {name for name in model.state_dict() if not name.startswith('encoder.')}
        if 'updated_encoder_keys' in saved:
            required.update(saved['updated_encoder_keys'])
        else:
            # Initial v1 files used the fixed last-four-layer policy without key metadata.
            if hasattr(model.encoder, 'layers'):
                layers, prefix = model.encoder.layers, 'encoder.layers'
            elif hasattr(model.encoder, 'encoder') and hasattr(model.encoder.encoder, 'layer'):
                layers, prefix = model.encoder.encoder.layer, 'encoder.encoder.layer'
            else:
                raise ValueError('Unsupported legacy compact encoder layout')
            required.update(f'{prefix}.{i}.{name}' for i in range(len(layers) - 4, len(layers))
                            for name, _ in layers[i].named_parameters())
        if not required.issubset(updates):
            raise ValueError('Compact checkpoint is missing updated encoder or task-head state')
        state = model.state_dict()
        if not set(updates).issubset(state):
            raise ValueError('Unexpected compact checkpoint state keys')
        state.update(updates)
    else:
        state = saved
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite checkpoint tensors')
    model.load_state_dict(state, strict=True)
    return {'format': 'frozen_encoder_delta_v1' if compact else 'full_state_dict',
            'strict_reconstructed_state': True, 'all_tensors_finite': True}


class SchemaMultitaskModel(nn.Module):
    def __init__(self, base_path, schema, max_span_width=32, rank=32, boundary_refinement=False, max_boundary_trim=32):
        super().__init__()
        self.schema = schema
        self.max_span_width = max_span_width
        self.rank = rank
        self.boundary_refinement = boundary_refinement
        self.max_boundary_trim = max_boundary_trim
        self.encoder = AutoModel.from_pretrained(base_path, local_files_only=True)
        h = self.encoder.config.hidden_size
        n = len(schema['labels'])
        self.start = nn.Linear(h, n * rank)
        self.end = nn.Linear(h, n * rank)
        self.span_bias = nn.Parameter(torch.full((n,), -3.0))
        if boundary_refinement:
            self.width_bias = nn.Parameter(torch.zeros(max_span_width,n))
        self.type_embedding = nn.Embedding(n, 32)
        self.span_projection = nn.Sequential(nn.Linear(h * 3 + 32, 256), nn.GELU(), nn.Dropout(.1))
        self.attribute_heads = nn.ModuleDict({f: nn.Linear(256, len(r['options']))
                                             for f, r in schema['fields'].items() if r['per_region']})
        if boundary_refinement:
            self.boundary_heads = nn.ModuleList([nn.Linear(256,max_boundary_trim+1),nn.Linear(256,max_boundary_trim+1)])
            for head in self.boundary_heads:
                nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
                with torch.no_grad():head.bias[0]=2.
        self.case_head = nn.Sequential(nn.Dropout(.1), nn.Linear(h, len(schema['fields']['case_decision']['options'])))
        # The relation head can learn only from complete, typed supervision.
        self.relation_head = nn.Sequential(nn.Linear(512 + h, 256), nn.GELU(), nn.Dropout(.1),
                                           nn.Linear(256, len(schema['relations']) + 1))

    def encode(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def span_logits(self, hidden):
        b, length, _ = hidden.shape
        n = len(self.schema['labels'])
        starts = self.start(hidden).reshape(b, length, n, self.rank)
        ends = self.end(hidden).reshape(b, length, n, self.rank)
        logits = []
        for width in range(1, self.max_span_width + 1):
            if width > length:
                logits.append(hidden.new_full((b, length, n), -1e4))
                continue
            shifted = F.pad(ends[:, width - 1:], (0, 0, 0, 0, 0, width - 1)) if width > 1 else ends
            value=(starts * shifted).sum(-1) / math.sqrt(self.rank) + self.span_bias
            if self.boundary_refinement:value=value+self.width_bias[width-1]
            logits.append(value)
        return torch.stack(logits, dim=2)

    def entity_representation(self, hidden, spans):
        if not spans:
            return hidden.new_empty((0, 256))
        rows = []
        for begin, end, label in spans:
            rows.append(torch.cat([hidden[begin], hidden[end - 1], hidden[begin:end].mean(0),
                                   self.type_embedding.weight[label]]))
        return self.span_projection(torch.stack(rows))

    def relation_logits(self, hidden, representations, pairs, spans):
        if not pairs:
            return hidden.new_empty((0, len(self.schema['relations']) + 1))
        rows = []
        for a, b in pairs:
            lo = min(spans[a][0], spans[b][0])
            hi = max(spans[a][1], spans[b][1])
            rows.append(torch.cat([representations[a], representations[b], hidden[lo:hi].mean(0)]))
        return self.relation_head(torch.stack(rows))


def span_loss(logits, targets, valid, negative_weight=1.):
    """Sample explicit negative candidates; preserve every valid overlapping target."""
    positive = logits[targets & valid]
    negative_mask = valid & ~targets
    negative = logits[negative_mask]
    if not len(positive) and not len(negative):
        return logits.sum() * 0
    # Include hard false spans as well as random negatives.
    count = min(len(negative), max(128, len(positive) * 16))
    if count:
        hard_count = count // 2
        hard = negative.topk(hard_count).values if hard_count else negative[:0]
        random = negative[torch.randint(len(negative), (count - hard_count,), device=logits.device)]
        negative = torch.cat([hard, random])
    values = []
    if len(positive):
        values.append(F.softplus(-positive).mean())
    if len(negative):
        values.append(negative_weight*F.softplus(negative).mean())
    return sum(values)
