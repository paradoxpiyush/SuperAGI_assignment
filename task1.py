import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        # Apply linear transformations for each head
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        # Split the input into multiple heads
        q = q.view(q.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        k = k.view(k.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        v = v.view(v.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model / self.n_heads, dtype=torch.float32))

        # Apply mask to prevent attending to future tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        # Apply attention weights to values
        out = torch.matmul(attention_weights, v)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(out.size(0), -1, self.d_model)

        # Apply linear transformation to get the final output
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
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feedforward = PositionwiseFeedforward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.self_attention(x, x, x, mask)
        x = self.norm1(x)
        x = x + self.feedforward(x)
        x = self.norm2(x)
        return x


class GPT2(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_heads=12, d_ff=3072, n_layers=12):
        super(GPT2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([GPT2Layer(d_model, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, x, mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


# Example usage
vocab_size = 10000  # Adjust as needed
model = GPT2(vocab_size)
input_sequence = torch.randint(0, vocab_size, (1, 10))  # Adjust sequence length as needed
attention_mask = torch.ones_like(input_sequence)

output = model(input_sequence, attention_mask)
print(output.shape)  # Should print torch.Size([1, 10, 768])
