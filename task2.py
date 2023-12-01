import torch
import torch.nn as nn
import torch.nn.functional as F
from task1 import GPT2

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(RotaryPositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.inv_freq = 1. / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))

    def forward(self, x):
        sinusoid_inp = torch.ger(x, self.inv_freq)
        sinusoid_inp = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        return self.embedding(x) + self.alpha * sinusoid_inp

class GroupQueryMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_groups=4):
        super(GroupQueryMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_groups = n_groups

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.group_linear = nn.Linear(d_model, d_model * n_groups)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        # Group queries
        q = self.group_linear(q).view(q.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model / self.n_heads, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, v)

        out = out.transpose(1, 2).contiguous().view(out.size(0), -1, self.d_model)
        out = self.out(out)

        return out

class SlidingWindowMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size=128):
        super(SlidingWindowMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model / self.n_heads, dtype=torch.float32))

        # Mask for sliding window
        window_mask = torch.tril(torch.ones(self.window_size, self.window_size)).view(1, 1, self.window_size, self.window_size)
        window_mask = window_mask.to(device=mask.device, dtype=mask.dtype)

        # Apply window mask
        scores = scores * window_mask + (1.0 - window_mask) * float('-inf')

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, v)

        out = out.transpose(1, 2).contiguous().view(out.size(0), -1, self.d_model)
        out = self.out(out)

        return out

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class GPT2Layer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(GPT2Layer, self).__init__()
        self.self_attention = GroupQueryMultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feedforward = PositionwiseFeedforward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.self_attention(x, x, x, mask)
        x = self.norm1(x)
        x = x + self.feedforward(x)
        x = self.norm2(x)
        return x

class GPT2WithRotaryPositionalAndGroupQuery(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_heads=12, d_ff=3072, n_layers=12):
        super(GPT2WithRotaryPositionalAndGroupQuery, self).__init__()
        self.embedding = RotaryPositionalEmbedding(d_model)
        self.layers = nn.ModuleList([GPT2Layer(d_model, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, x, mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

vocab_size = 10000  # Adjust as needed
model = GPT2(vocab_size)
input_sequence = torch.randint(0, vocab_size, (1, 10))  # Adjust sequence length as needed
attention_mask = torch.ones_like(input_sequence)

model_with_rotary_and_group_query = GPT2WithRotaryPositionalAndGroupQuery(vocab_size)
output = model_with_rotary_and_group_query(input_sequence, attention_mask)
print(output.shape)
