# GPT-2 Implementation and Alterations

## Task-1: GPT-2 Implementation

### Overview

In Task-1, we implemented a simplified version of the GPT-2 model in PyTorch. The model includes the multi-head self-attention mechanism, position-wise feed-forward networks, and token/positional embeddings.

### Code Implementation

1. **Multi-Head Self-Attention Mechanism:** Implemented the `MultiHeadAttention` class.
2. **Position-wise Feed-Forward Networks:** Implemented the `PositionwiseFeedforward` class.
3. **Token and Positional Embeddings:** Implemented basic token and positional embeddings.

### Approach
The implementation followed a modular approach, organizing the code into separate classes for each major component: MultiHeadAttention, PositionwiseFeedforward, RotaryPositionalEmbedding, GPT2Layer, and GPT2.
This modular design improves code readability and maintainability.
Layer Stacking:

GPT-2 consists of multiple layers of attention and feed-forward networks. We used nn.ModuleList to stack these layers in the GPT2 model.
This stacking allows the model to capture hierarchical features and relationships within the data.
Validation:

To validate the implementation, we generated a random input sequence and passed it through the model.
The output shape was compared against expectations to ensure correctness.
Attention Masking:

Attention masking to prevent attending to future tokens required careful indexing and tensor operations.
Following the Transformer architecture, we ensured proper masking by considering both padding and future tokens.
Positional Embeddings:

For positional embeddings, we implemented a basic sinusoidal encoding following the original Transformer architecture.
Positional embeddings were implemented as a separate class (RotaryPositionalEmbedding) for clarity.
- Modular code structure for better organization.
- Validation using randomly generated input sequences.

### Difficulties Encountered and Solutions

- **Attention Masking:** Careful tensor indexing and operations were crucial.
- **Positional Embeddings:** Considerable thought into different positional encoding strategies.

### Reference Materials

- [Attention is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [nanoGPT Repository by Andrej Karpathy](https://github.com/karpathy/nanoGPT)

---

## Task-2: GPT-2 Model Alterations

### Overview

In Task-2, we introduced three alterations to the GPT-2 model: Rotary Positional Embedding, Group Query Attention, and Sliding Window Attention.

### 1. Rotary Positional Embedding

#### Approach

- Followed insights from Su et al.'s RoFormer.
- Created the `RotaryPositionalEmbedding` class.
- Integrated rotary positional embeddings into the GPT-2 model.

#### Results

- Successfully replaced original positional embeddings with rotary embeddings.
- Maintained the model's ability to capture sequential information.

### 2. Group Query Attention



#### Approach

Understanding Rotary Embeddings:

Studied Su et al.'s RoFormer paper to grasp the concept of rotary positional embeddings.
Rotary embeddings introduce a rotational factor to positional embeddings, enhancing the model's ability to capture sequential information.
Implementation of Rotary Positional Embeddings:

Created the RotaryPositionalEmbedding class, incorporating the rotational factor into positional embeddings.
Integrated this class into the GPT-2 model (GPT2WithRotaryPositionalEmbedding).
Validation:

Ensured that the model with rotary positional embeddings maintained its capability to capture sequential information.
Compared outputs with and without rotary embeddings to validate their impact.

- Implemented the `GroupQueryMultiHeadAttention` class.
- Modified the attention mechanism to include group queries.

#### Results

- The model exhibited a modified operation compared to standard attention.
- Group query attention allowed for capturing different aspects of the input sequence simultaneously.

### 3. Sliding Window Attention

#### Approach

Understanding Sliding Window Attention:

Referred to Beltagy et al.'s Longformer paper to understand the concept and benefits of sliding window attention.
Sliding window attention restricts attention to a local window, potentially improving scalability.
Implementation of Sliding Window Attention:

Implemented the SlidingWindowMultiHeadAttention class, adjusting the attention mechanism to consider only a local window of tokens.
Integrated this class into the GPT-2 model (GPT2WithSlidingWindowAttention).
Validation:

Observed the model's performance with sliding window attention, particularly in tasks requiring long-range dependencies.
Checked if the model exhibited improvements in computation efficiency.


- Implemented the `SlidingWindowMultiHeadAttention` class.
- Applied a sliding window mask to restrict attention to a local window.

#### Results

- Sliding window attention successfully limited computation to a local context.
- Performance improvements were observed, particularly on tasks requiring long-range dependencies.

### Overall Observations and Conclusion

- The implemented alterations showcased the adaptability of the GPT-2 architecture.
- Each modification affected the model's operation, offering different trade-offs.
- Rotary embeddings preserved sequential information, group query attention enhanced parallel processing, and sliding window attention improved scalability.

### Reference Materials

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864.pdf)
- [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/pdf/2305.13245v2.pdf)
- [Longformer: The Long-Document Transformer](https://arxiv.org/pdf/2004.05150v2.pdf)


## Overall Observations and Conclusion
The alterations were implemented with a focus on preserving the model's original capabilities while introducing enhancements.
Each modification was validated to ensure its impact on the model's performance and behavior.
The overall goal was to showcase the adaptability of the GPT-2 architecture and analyze the trade-offs introduced by each alteration.
