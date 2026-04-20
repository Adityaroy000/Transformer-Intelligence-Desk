
"""
capstone_streamlit.py - Research Paper Q&A - Attention Is All You Need Agent
Run: streamlit run capstone_streamlit.py
"""
import uuid

import streamlit as st
from dotenv import load_dotenv
from agent import load_agent as load_backend_agent

load_dotenv(override=True)

DOMAIN_NAME = "Research Paper Q&A - Attention Is All You Need"
DOMAIN_DESCRIPTION = "A grounded research Q&A agent that answers questions about the 'Attention Is All You Need paper'"
DOCUMENTS = [{'id': 'doc_001', 'topic': 'Abstract — What the Transformer Is', 'text': "The paper 'Attention Is All You Need' by Vaswani et al. (2017) introduces the Transformer, a new neural network architecture for sequence-to-sequence tasks. Before this paper, the dominant models for tasks like machine translation were based on complex recurrent neural networks (RNNs) or convolutional neural networks (CNNs) arranged in an encoder-decoder structure. The best of these models also used an attention mechanism connecting the encoder and decoder.\nThe Transformer is different because it is based solely on attention mechanisms. It completely dispenses with recurrence and convolution. This design makes the model significantly more parallelizable during training and requires substantially less time to train compared to recurrent models.\nThe Transformer was evaluated on two machine translation benchmarks. On the WMT 2014 English-to-German translation task, the model achieves 28.4 BLEU, improving over the previous best results including ensembles by more than 2 BLEU points. On the WMT 2014 English-to-French translation task, it establishes a new single-model state-of-the-art BLEU score of 41.8, trained for 3.5 days on eight GPUs — a small fraction of the training cost of prior best models. The paper also demonstrates that the Transformer generalizes well to other tasks, such as English constituency parsing."}, {'id': 'doc_002', 'topic': 'Introduction — Why the Transformer Was Needed', 'text': 'Before the Transformer, recurrent neural networks (RNNs), long short-term memory networks (LSTMs), and gated recurrent units (GRUs) were the dominant approaches for sequence modeling and transduction tasks such as machine translation and language modeling.\nRecurrent models process sequences one token at a time. The hidden state h_t is computed as a function of the previous hidden state h_{t-1} and the current input. This inherently sequential nature means operations cannot be parallelized across positions within a training example. At longer sequence lengths, memory constraints also limit the batch size that can be used during training. These two problems together make training slow and expensive.\nAttention mechanisms had already become an important part of sequence models by 2017. They allow the model to focus on relevant parts of the input sequence regardless of the distance between positions. However, attention was almost always used in conjunction with a recurrent network — it was an add-on, not the foundation.\nThe paper proposes the Transformer as a model architecture that completely eliminates recurrence. Instead, the Transformer relies entirely on attention mechanisms to capture dependencies between all positions in the input and output sequences simultaneously. This design makes the model highly parallelizable. As a result, the Transformer can be trained to state-of-the-art quality in as little as 12 hours on 8 NVIDIA P100 GPUs for the base configuration.'}, {'id': 'doc_003', 'topic': 'Background — Prior Work and Self-Attention', 'text': 'The Background section of the paper discusses prior attempts to reduce sequential computation and the concept of self-attention.\n\n    Earlier work such as the Extended Neural GPU, ByteNet, and ConvS2S tried to reduce sequential computation by using convolutional neural networks to compute hidden representations in parallel. However, none of them fully solved the problem of relating positions far apart in a sequence. The number of operations required to relate signals from two arbitrary positions in the input or output sequence grows with distance: O(n) sequential operations for ConvS2S and O(log n) for ByteNet, where n is the sequence length. In the Transformer, this is reduced to a constant O(1) number of operations regardless of the distance between positions, because self-attention directly connects all positions.\n\n    Self-attention, also called intra-attention, is an attention mechanism that relates different positions within a single sequence to compute a representation of that sequence. Self-attention had already been used successfully in tasks like reading comprehension, abstractive summarization, textual entailment, and learning task-independent sentence representations before the Transformer paper.\n\n    End-to-end memory networks, another prior approach, use a recurrent attention mechanism instead of sequence-aligned recurrence and perform well on simple question answering and language modeling tasks.\n\n    The Transformer is the first transduction model that relies entirely on self-attention to compute representations of its input and output, without using any sequence-aligned recurrent layers or convolutional layers. This is the key architectural innovation of the paper.'}, {'id': 'doc_004', 'topic': 'Model Architecture Overview — Encoder-Decoder Structure', 'text': 'The Transformer follows an encoder-decoder structure, which is the standard design for sequence-to-sequence tasks like machine translation.\n\nThe encoder takes an input sequence of symbol representations (x1, x2, ..., xn) and maps it to a sequence of continuous representations z = (z1, z2, ..., zn). Given z, the decoder then generates an output sequence (y1, y2, ..., ym) one symbol at a time. At each step, the decoder consumes the previously generated symbols as additional input — this is called auto-regressive generation.\n\nBoth the encoder and the decoder are composed of stacked self-attention layers and point-wise fully connected layers. The key architectural parameters are:\n\n- N = 6 identical layers in both the encoder stack and the decoder stack\n- d_model = 512, which is the fixed output dimensionality of all sub-layers and all embedding layers throughout the model\n\nThe encoder can process the entire input sequence in parallel because it has no auto-regressive dependency. The decoder, however, must generate tokens one at a time because each token depends on all previously generated tokens.\n\nEvery sub-layer in both the encoder and decoder uses two design techniques: residual connections and layer normalization. A residual connection adds the input of a sub-layer directly to its output before normalization. This helps gradients flow during training. The output of each sub-layer is therefore LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function computed by the sub-layer itself. These two techniques are applied consistently throughout the entire architecture.'}, {'id': 'doc_005', 'topic': 'Encoder Stack — Structure and Sub-Layers', 'text': 'The encoder in the Transformer is composed of a stack of N = 6 identical layers. Each layer contains exactly two sub-layers:\n\n1. Multi-head self-attention mechanism — The first sub-layer allows each position in the input sequence to attend to all other positions in the same sequence. There is no masking, so every position can freely look at every other position. This is what allows the encoder to process the entire input sequence in parallel.\n\n2. Position-wise fully connected feed-forward network — The second sub-layer applies the same feed-forward network independently to each position.\n\nBoth sub-layers use the same two design patterns: a residual connection followed by layer normalization. The output of each sub-layer is computed as:\n\n    Output = LayerNorm(x + Sublayer(x))\n\nwhere x is the input to the sub-layer and Sublayer(x) is the function the sub-layer computes. The residual connection adds the original input back to the output, which helps gradient flow during training. Layer normalization then stabilizes the activations.\n\nTo make residual connections work cleanly throughout the model, all sub-layers and embedding layers are designed to produce outputs of the same fixed dimension: d_model = 512. This consistent dimensionality is maintained through every layer of the encoder stack.\n\nThe encoder has no positional masking — any position can attend to any other position in the input. This is in contrast to the decoder, which uses masking to prevent positions from attending to future tokens.'}, {'id': 'doc_006', 'topic': 'Decoder Stack — Structure, Masking, and Encoder-Decoder Attention', 'text': "The decoder in the Transformer is also composed of a stack of N = 6 identical layers. However, each decoder layer has three sub-layers instead of the encoder's two:\n\n1. Masked multi-head self-attention — The first sub-layer allows each position in the output sequence to attend to all previous positions. A masking mechanism is applied to prevent any position from attending to future positions (positions that have not been generated yet). This ensures the auto-regressive property: the prediction for position i can only depend on outputs at positions less than i.\n\n2. Multi-head encoder-decoder attention — The second sub-layer attends over the encoder's output. The queries come from the previous decoder sub-layer, while the keys and values come from the final output of the encoder stack. This allows every decoder position to attend over all positions in the input sequence, giving the decoder full access to the encoded input representation.\n\n3. Position-wise fully connected feed-forward network — The third sub-layer is identical to the one used in the encoder: applied independently to each position.\n\nAll three sub-layers use residual connections followed by layer normalization: LayerNorm(x + Sublayer(x)), exactly as in the encoder.\n\nThe decoder generates output tokens one at a time. At each step, it takes all previously generated tokens as input. The output embeddings are offset by one position to the right, and combined with the masking, this guarantees the auto-regressive property across the entire generation process."}, {'id': 'doc_007', 'topic': 'Scaled Dot-Product Attention — Formula and Why Scaling Matters', 'text': 'Scaled Dot-Product Attention is the core building block of the Transformer. It takes three inputs: queries (Q), keys (K), and values (V).\n\nThe attention formula is:\n\n    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V\n\nHere, d_k is the dimension of the queries and keys, and d_v is the dimension of the values. The computation works as follows: the query matrix Q is multiplied by the transpose of the key matrix K^T to produce a matrix of dot-product scores. Each score measures how relevant a key is to a given query. These scores are then divided by the square root of d_k (sqrt(d_k)), and a softmax function is applied to convert them into attention weights that sum to 1. Finally, the weights are used to compute a weighted sum of the value vectors V, producing the output.\n\nThere are two main types of attention functions: additive attention and dot-product (multiplicative) attention. Dot-product attention is identical to Scaled Dot-Product Attention except without the scaling factor. The authors add the scaling by 1/sqrt(d_k) because for large values of d_k, the dot products grow very large in magnitude, which pushes the softmax function into regions with extremely small gradients and slows down training.\n\nIn practice, Q, K, and V are matrices, so multiple queries are computed simultaneously using efficient matrix multiplication operations. Scaled Dot-Product Attention is both faster and more memory-efficient than additive attention because it relies entirely on matrix multiplication rather than separate feed-forward networks.'}, {'id': 'doc_008', 'topic': 'Multi-Head Attention — h Heads, Dimensions, and Three Use Cases', 'text': 'Multi-Head Attention extends Scaled Dot-Product Attention by running multiple attention operations in parallel. Instead of performing a single attention function with full d_model-dimensional queries, keys, and values, the model linearly projects Q, K, and V into h different lower-dimensional spaces using learned projection matrices, performs attention in each space independently, then concatenates and projects the results.\n\nThe formula is:\n    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O\n    where head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)\n\nIn the paper, the specific values used are:\n- h = 8 parallel attention heads\n- d_k = d_v = d_model / h = 512 / 8 = 64\n- Each head projects to dimensions: d_k = 64 for queries and keys, d_v = 64 for values\n- The concatenated output has dimension h * d_v = 8 * 64 = 512, which is then projected by W^O back to d_model = 512\n\nBecause each head operates on a reduced dimension (64 instead of 512), the total computational cost of multi-head attention is similar to single-head attention with full dimensionality.\n\nMulti-head attention allows the model to jointly attend to information from different representation subspaces at different positions. A single attention head would average these, losing the ability to focus on distinct patterns simultaneously.\n\nThe Transformer uses multi-head attention in three different ways:\n1. Encoder self-attention: every position attends to all positions in the previous encoder layer.\n2. Decoder masked self-attention: each position attends to all previous positions in the decoder.\n3. Encoder-decoder attention: queries come from the decoder, keys and values come from the encoder output.'}, {'id': 'doc_009', 'topic': 'Position-wise Feed-Forward Networks — FFN Structure and Dimensions', 'text': 'In addition to the multi-head attention sub-layer, each encoder and decoder layer contains a fully connected feed-forward network (FFN). This network is applied to each position separately and identically — meaning the same network weights are applied to every token position, but each position is processed independently of all others. This is why it is called position-wise.\n\nThe feed-forward network consists of two linear transformations with a ReLU activation in between:\n\n    FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2\n\nThe key dimensions are:\n- Input dimension: d_model = 512\n- Inner (hidden) layer dimension: d_ff = 2048\n- Output dimension: d_model = 512\n\nThe network expands from 512 to 2048 (four times larger), applies ReLU, then projects back down to 512. This expansion gives the model significant additional capacity to learn complex non-linear transformations at each position after the attention mechanism has mixed information across positions.\n\nWhile the same W_1, b_1, W_2, b_2 matrices are shared across all positions within a single layer, each of the 6 encoder layers and each of the 6 decoder layers has its own separate set of parameters.\n\nThe position-wise FFN can also be interpreted as two convolution operations with a kernel size of 1. This equivalence means the FFN is computationally efficient and easy to implement using standard matrix operations, while still providing the model with the capacity to transform representations at each position after attention has been applied.'}, {'id': 'doc_010', 'topic': 'Positional Encoding — How the Transformer Knows Token Order', 'text': "Since the Transformer contains no recurrence and no convolution, it has no built-in sense of the order or position of tokens in a sequence. Without any positional information, the model would treat the sentence 'The cat sat on the mat' identically regardless of word order. To solve this, the paper injects positional encodings into the input embeddings.\n\nPositional encodings are added directly to the input embeddings at the bottom of both the encoder and decoder stacks. They have the same dimension as the embeddings (d_model = 512) so they can be summed together.\n\nThe paper uses fixed sine and cosine functions of different frequencies to compute positional encodings:\n\n    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))\n    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))\n\nwhere pos is the position of the token in the sequence and i is the dimension index. Each dimension of the positional encoding corresponds to a sinusoid with a different wavelength, forming a geometric progression from 2π to 10000 × 2π.\n\nThe authors chose sinusoidal functions because for any fixed offset k, the positional encoding PE(pos + k) can be represented as a linear function of PE(pos). This means the model can easily learn to attend by relative positions.\n\nThe paper also experimented with learned positional embeddings as an alternative and found nearly identical results. Sinusoidal encoding was kept because it may allow the model to generalize to sequence lengths longer than those seen during training.\n\nAdditionally, dropout is applied to the sum of the embeddings and positional encodings in both the encoder and decoder."}, {'id': 'doc_011', 'topic': 'Training Setup — Data, Hardware, Optimizer, and Regularization', 'text': 'The Transformer was trained on two large machine translation datasets.\n\nTraining Data:\n- WMT 2014 English-German: approximately 4.5 million sentence pairs, encoded using byte-pair encoding (BPE) with a shared vocabulary of approximately 37,000 tokens.\n- WMT 2014 English-French: approximately 36 million sentence pairs, encoded using a word-piece vocabulary of 32,000 tokens.\n\nHardware and Training Duration:\n- All models were trained on 8 NVIDIA P100 GPUs.\n- Base model: trained for 100,000 steps, taking approximately 12 hours.\n- Big model: trained for 300,000 steps, taking approximately 3.5 days.\n\nOptimizer:\nThe Adam optimizer was used with the following hyperparameters: β₁ = 0.9, β₂ = 0.98, and ε = 10⁻⁹. A custom learning rate schedule was used:\n\n    lrate = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))\n    warmup_steps = 4000\n\nThis schedule increases the learning rate linearly for the first 4,000 training steps, then decreases it proportionally to the inverse square root of the step number.\n\nRegularization — two techniques were applied:\n\n1. Residual Dropout: Dropout with rate P_drop = 0.1 applied to the output of every sub-layer before it is added to the residual connection and normalized. Dropout is also applied to the sums of the embeddings and the positional encodings in both the encoder and decoder.\n\n2. Label Smoothing: A label smoothing value of ε_ls = 0.1 was used during training. This hurts perplexity slightly because the model learns to be less confident, but it improves accuracy and BLEU score on the translation benchmarks.'}, {'id': 'doc_012', 'topic': 'Results and Benchmarks — BLEU Scores and Model Configurations', 'text': 'The paper evaluates the Transformer on two machine translation benchmarks and reports results for two model sizes: base and big.\n\nWMT 2014 English-to-German Translation:\n- Transformer (base): 27.3 BLEU\n- Transformer (big): 28.4 BLEU\n- Previous best (ConvS2S ensemble): 26.36 BLEU\nThe Transformer (big) improves over the previous best results, including ensembles, by more than 2 BLEU points.\n\nWMT 2014 English-to-French Translation:\n- Transformer (base): 38.1 BLEU\n- Transformer (big): 41.8 BLEU — a new single-model state-of-the-art\nThe big model achieves this score after training for 3.5 days on 8 GPUs, at a small fraction of the training cost of all previous state-of-the-art models.\n\nModel Configurations (Table 1 in the paper):\n\nBase model:\n- d_model = 512, d_ff = 2048, h = 8, d_k = d_v = 64\n- N = 6 layers, dropout = 0.1, label smoothing = 0.1\n- Parameters: 65 million\n- Training: 100,000 steps (~12 hours on 8 P100 GPUs)\n\nBig model:\n- d_model = 1024, d_ff = 4096, h = 16, d_k = d_v = 64\n- N = 6 layers, dropout = 0.3, label smoothing = 0.1\n- Parameters: 213 million\n- Training: 300,000 steps (~3.5 days on 8 P100 GPUs)\n\nThe paper demonstrates that the Transformer achieves superior translation quality at significantly lower training cost compared to all prior recurrent and convolutional sequence-to-sequence models.'}, {'id': 'doc_013', 'topic': 'Ablation Study — Effect of Architecture Choices (Table 3)', 'text': 'Section 6.3 of the paper investigates the importance of different Transformer components by varying one aspect at a time and measuring the impact on translation quality. All ablation experiments are evaluated on the WMT 2014 English-to-German development set (newstest2013). The base model scores 25.8 BLEU on this set.\n\nNumber of Attention Heads (h):\nReducing to a single attention head (h=1, d_k=d_v=512) drops performance significantly to 23.3 BLEU. Increasing heads too far (h=16, h=32) with correspondingly smaller d_k also slightly reduces quality. The optimal is h=8 with d_k=d_v=64, confirming the base model choice.\n\nKey Dimension Size (d_k):\nReducing d_k while keeping h=8 hurts model quality. The paper concludes that determining compatibility between queries and keys is not easy, and that more sophisticated compatibility functions than dot product may be beneficial.\n\nModel Size:\nLarger models (bigger d_model, d_ff, or more layers) consistently perform better, as shown in the big model results (213M parameters vs 65M for base).\n\nDropout:\nRemoving dropout hurts performance. The base model uses P_drop = 0.1. The big model requires higher dropout of P_drop = 0.3 to avoid overfitting given its larger capacity.\n\nLabel Smoothing:\nUsing ε_ls = 0.1 improves BLEU scores compared to no label smoothing, even though it increases perplexity. Label smoothing encourages the model to be less overconfident in its predictions.\n\nPositional Encoding:\nReplacing the sinusoidal positional encodings with learned positional embeddings produces nearly identical results, confirming both approaches are viable.'}, {'id': 'doc_014', 'topic': 'Conclusion and Impact — Summary of Contributions and Future Work', 'text': "The conclusion of 'Attention Is All You Need' summarises the key contributions of the Transformer and outlines directions for future work.\n\nKey Contributions:\nThe Transformer is the first sequence transduction model based entirely on attention mechanisms, completely replacing the recurrent layers that were standard in encoder-decoder architectures. Compared to models based on recurrent or convolutional layers, the Transformer can be trained significantly faster due to its high parallelizability. On machine translation benchmarks, the Transformer achieves new state-of-the-art results: 28.4 BLEU on WMT 2014 English-to-German and 41.8 BLEU on WMT 2014 English-to-French.\n\nThe paper also demonstrates that the Transformer generalises beyond machine translation. It was successfully applied to English constituency parsing with both large and limited training data, showing competitive results with task-specific architectures.\n\nFuture Work Mentioned in the Paper:\nThe authors planned to extend attention-based models to other modalities including images, audio, and video. They also intended to explore local, restricted attention mechanisms to handle very long sequences more efficiently. Another goal was to make the generation process less sequential, potentially enabling faster inference.\n\nThe code used to train and evaluate all models in the paper was made available at the TensorFlow tensor2tensor repository on GitHub.\n\nBroader Impact:\nThe Transformer architecture introduced in this paper became the direct foundation for virtually all subsequent large language models, including BERT (2018), GPT-2 (2019), T5 (2019), and GPT-4. It fundamentally shifted the field of natural language processing and deep learning from recurrent architectures to attention-based models, making 'Attention Is All You Need' one of the most cited and influential papers in AI history."}]

st.set_page_config(
    page_title=DOMAIN_NAME,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ─── Accent & Layout Tokens (theme-independent) ─── */
    :root {
        --accent-1:   #4f9ffe;
        --accent-2:   #a259ff;
        --accent-3:   #00e5c4;
        --success:    #22d3a0;

        --radius-sm:   8px;
        --radius-md:   14px;
        --radius-lg:   20px;
        --radius-xl:   28px;
        --radius-pill: 999px;

        --font-body:    'Inter', system-ui, sans-serif;
        --font-heading: 'Space Grotesk', system-ui, sans-serif;
        --font-mono:    'JetBrains Mono', monospace;

        --shadow-sm: 0 2px 8px rgba(0,0,0,0.18);
        --shadow-md: 0 8px 32px rgba(0,0,0,0.28);
    }

    /* ── DARK theme ── */
    [data-theme="dark"] {
        --page-bg:        linear-gradient(160deg,#080c18 0%,#0a0e1a 40%,#0c1020 100%);
        --glow-a:         rgba(79,159,254,0.16);
        --glow-b:         rgba(162,89,255,0.12);
        --glow-c:         rgba(0,229,196,0.09);
        --panel-bg:       rgba(10,14,26,0.78);
        --panel-border:   rgba(255,255,255,0.07);
        --panel-border-m: rgba(255,255,255,0.13);
        --sidebar-bg:     linear-gradient(180deg,rgba(12,16,28,0.97) 0%,rgba(10,14,24,0.99) 100%);
        --bottom-bg:      rgba(8,12,22,0.88);
        --chat-bg:        rgba(15,21,37,0.65);
        --chat-user-bg:   rgba(79,159,254,0.055);
        --chat-ai-bg:     rgba(162,89,255,0.045);
        --input-bg:       rgba(15,21,37,0.85);
        --card-bg:        rgba(255,255,255,0.038);
        --hero-bg:        linear-gradient(135deg,rgba(79,159,254,0.12) 0%,rgba(162,89,255,0.08) 50%,rgba(0,229,196,0.06) 100%);
        --text-hi:        #e8edf8;
        --text-mid:       rgba(232,237,248,0.62);
        --text-lo:        rgba(232,237,248,0.38);
        --title-grad:     linear-gradient(135deg,#e8edf8 0%,#a0c4ff 55%,#c4a0ff 100%);
        --sidebar-sticky-bg: rgba(10,14,24,0.97);
    }

    /* ── LIGHT theme ── */
    [data-theme="light"] {
        --page-bg:        linear-gradient(160deg,#eef2fb 0%,#f0f4fc 40%,#e8edf8 100%);
        --glow-a:         rgba(79,159,254,0.10);
        --glow-b:         rgba(162,89,255,0.07);
        --glow-c:         rgba(0,229,196,0.06);
        --panel-bg:       rgba(255,255,255,0.72);
        --panel-border:   rgba(0,0,0,0.07);
        --panel-border-m: rgba(79,159,254,0.22);
        --sidebar-bg:     linear-gradient(180deg,rgba(240,244,255,0.97) 0%,rgba(235,240,255,0.99) 100%);
        --bottom-bg:      rgba(240,244,255,0.92);
        --chat-bg:        rgba(255,255,255,0.75);
        --chat-user-bg:   rgba(79,159,254,0.07);
        --chat-ai-bg:     rgba(162,89,255,0.06);
        --input-bg:       rgba(255,255,255,0.90);
        --card-bg:        rgba(79,159,254,0.05);
        --hero-bg:        linear-gradient(135deg,rgba(79,159,254,0.09) 0%,rgba(162,89,255,0.06) 50%,rgba(0,229,196,0.05) 100%);
        --text-hi:        #1a1f36;
        --text-mid:       rgba(26,31,54,0.65);
        --text-lo:        rgba(26,31,54,0.40);
        --title-grad:     linear-gradient(135deg,#1a1f36 0%,#2563eb 55%,#7c3aed 100%);
        --sidebar-sticky-bg: rgba(240,244,255,0.97);
    }

    /* ─── Global fonts — only plain text elements, NOT icon spans ─── */
    body, p, h1, h2, h3, h4, h5, h6,
    button, input, textarea, label, li, a, td, th {
        font-family: var(--font-body);
    }

    /* ─── Animated Mesh Background ─── */
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse 900px 600px at 8%  -8%,  var(--glow-a) 0%, transparent 65%),
            radial-gradient(ellipse 700px 500px at 95%  5%,  var(--glow-b) 0%, transparent 60%),
            radial-gradient(ellipse 600px 400px at 50% 95%,  var(--glow-c) 0%, transparent 55%),
            var(--page-bg);
        background-attachment: fixed;
        color: var(--text-hi);
        min-height: 100vh;
        transition: background 0.35s ease;
    }

    /* Subtle animated starfield */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            radial-gradient(1px 1px at 18% 22%, var(--glow-a) 0%, transparent 100%),
            radial-gradient(1px 1px at 73% 14%, var(--glow-b) 0%, transparent 100%),
            radial-gradient(1px 1px at 42% 67%, var(--glow-a) 0%, transparent 100%),
            radial-gradient(1px 1px at 88% 54%, var(--glow-b) 0%, transparent 100%),
            radial-gradient(1.5px 1.5px at 62% 38%, var(--glow-a) 0%, transparent 100%),
            radial-gradient(1.5px 1.5px at 30% 91%, var(--glow-c) 0%, transparent 100%);
        pointer-events: none;
        z-index: 0;
    }

    /* ─── stHeader — transparent, same height as navbar, toolbar stays inside ─── */
    [data-testid="stHeader"] {
        background: transparent !important;
        height: 3.4rem !important;
        min-height: 3.4rem !important;
        padding: 0 !important;
        border: none !important;
        overflow: visible !important;
        z-index: 999 !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
    }

    /* ─── Streamlit toolbar — stays in stHeader, styled with glass buttons ─── */
    [data-testid="stToolbar"] {
        display: flex !important;
        align-items: center !important;
        gap: 0.25rem !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 1rem 0 0 !important;
    }
    /* Deploy button + icon button styling */
    [data-testid="stToolbar"] button,
    [data-testid="stToolbar"] a {
        background: rgba(79,159,254,0.09) !important;
        border: 1px solid rgba(79,159,254,0.25) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-hi) !important;
        padding: 0.22rem 0.72rem !important;
        font-family: var(--font-heading) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.18s ease !important;
        height: 1.85rem !important;
        display: inline-flex !important;
        align-items: center !important;
    }
    [data-testid="stToolbar"] button:hover,
    [data-testid="stToolbar"] a:hover {
        background: rgba(79,159,254,0.18) !important;
        border-color: rgba(79,159,254,0.55) !important;
        box-shadow: 0 0 12px rgba(79,159,254,0.2) !important;
    }
    /* Preserve Material icon font for ⋮ */
    [data-testid="stToolbar"] [title] {
        font-family: 'Material Icons', 'Material Symbols Outlined' !important;
    }

    /* ─── Custom Navbar — visual layer behind transparent stHeader ─── */
    .custom-navbar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 998;  /* below stHeader (999) so toolbar shows on top */
        display: flex;
        align-items: center;
        height: 3.4rem;
        padding: 0 1.5rem;
        background: var(--panel-bg);
        backdrop-filter: blur(20px) saturate(160%);
        -webkit-backdrop-filter: blur(20px) saturate(160%);
        border-bottom: 1px solid var(--panel-border-m);
        transition: background 0.35s ease;
    }

    /* Accent gradient on bottom border */
    .custom-navbar::after {
        content: "";
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg,
            transparent 0%,
            var(--accent-1) 20%,
            var(--accent-2) 55%,
            var(--accent-3) 82%,
            transparent 100%
        );
        opacity: 0.6;
    }

    .navbar-left {
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 0.05rem;
    }

    .navbar-title {
        font-family: var(--font-heading);
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: -0.01em;
        color: var(--text-hi);
        line-height: 1.15;
        margin: 0;
    }

    .navbar-desc {
        font-size: 0.72rem;
        color: var(--text-mid);
        line-height: 1.3;
        margin: 0;
        transition: color 0.35s ease;
    }

    /* ─── Push all main content below the navbar ─── */
    [data-testid="stMainBlockContainer"] {
        max-width: 1100px !important;
        padding: 4.8rem 1.8rem 2rem !important;
        background: var(--panel-bg) !important;
        backdrop-filter: blur(24px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(160%) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: var(--radius-xl) !important;
        box-shadow:
            0 0 0 1px rgba(255,255,255,0.04),
            0 32px 80px rgba(0,0,0,0.35),
            inset 0 1px 0 rgba(255,255,255,0.06) !important;
        position: relative;
        z-index: 1;
        transition: background 0.35s ease, border-color 0.35s ease;
    }


    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--panel-border) !important;
        box-shadow: 4px 0 40px rgba(0,0,0,0.3) !important;
        backdrop-filter: blur(16px) !important;
        transition: background 0.35s ease;
    }

    /* Pull sidebar content flush to top */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
    }

    /* Accent gradient edge on sidebar */
    [data-testid="stSidebar"]::after {
        content: "";
        position: absolute;
        top: 0; right: 0;
        width: 1px; height: 100%;
        background: linear-gradient(
            180deg,
            transparent 0%,
            rgba(79,159,254,0.55) 20%,
            rgba(162,89,255,0.40) 60%,
            rgba(0,229,196,0.25) 85%,
            transparent 100%
        );
    }

    /* ─── Bottom Chat Bar ─── */
    [data-testid="stBottomBlockContainer"] {
        background: var(--bottom-bg) !important;
        backdrop-filter: blur(20px) !important;
        border-top: 1px solid var(--panel-border) !important;
        transition: background 0.35s ease;
    }

    /* ─── Page Header (ChatGPT-style top strip) ─── */
    /* Neutralize Streamlit's element wrapper for the very first block */
    [data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"]
        > div:first-child [data-testid="stMarkdownContainer"],
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]
        > div:first-child [data-testid="stMarkdownContainer"] {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    .page-header {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        padding: 0.3rem 0 0.85rem;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid var(--panel-border-m);
        position: relative;
    }

    /* Accent gradient on bottom border */
    .page-header::after {
        content: "";
        position: absolute;
        bottom: -1px; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg,
            var(--accent-1) 0%,
            var(--accent-2) 50%,
            var(--accent-3) 100%
        );
        opacity: 0.55;
    }

    .page-header-title {
        font-family: var(--font-heading);
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: -0.01em;
        color: var(--text-hi);
        line-height: 1.2;
        margin: 0;
    }

    .page-header-sub {
        font-size: 0.76rem;
        color: var(--text-mid);
        line-height: 1.45;
        margin: 0.18rem 0 0;
        transition: color 0.35s ease;
    }

    /* ─── Legacy hero-shell (kept for ::before/::after reference) ─── */
    .hero-shell {
        position: relative;
        overflow: hidden;
        background: var(--hero-bg);
        border: 1px solid var(--panel-border-m);
        border-radius: var(--radius-lg);
        padding: 1.4rem 1.6rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 0 rgba(255,255,255,0.07) inset, var(--shadow-md);
        transition: background 0.35s ease;
    }

    /* Stat badges in hero */
    .hero-stats {
        display: flex;
        gap: 0.65rem;
        margin-top: 1rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }

    .hero-stat {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(255,255,255,0.05);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 0.28rem 0.7rem;
        font-size: 0.82rem;
        font-weight: 500;
        color: var(--text-secondary);
    }

    .hero-stat span.val {
        color: var(--accent-1);
        font-weight: 600;
        font-family: var(--font-mono);
        font-size: 0.8rem;
    }

    /* ─── Sidebar Cards ─── */
    .sidebar-top {
        position: sticky;
        top: 3.4rem;      /* align below fixed navbar */
        z-index: 10;
        background: var(--sidebar-sticky-bg);
        padding: 0.6rem 0 0.55rem;
        backdrop-filter: blur(12px);
    }

    .sidebar-top h3 {
        font-family: var(--font-heading) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: var(--text-lo) !important;
        margin: 0 0 0.5rem !important;   /* no top margin — flush to top */
    }

    .sidebar-card {
        background: var(--card-bg);
        border: 1px solid var(--panel-border);
        border-radius: var(--radius-md);
        padding: 0.72rem 0.82rem;
        color: var(--text-mid);
        margin-bottom: 0.6rem;
        font-size: 0.87rem;
        line-height: 1.55;
        transition: background 0.35s ease;
    }

    .session-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(79,159,254,0.08);
        border: 1px solid rgba(79,159,254,0.2);
        color: var(--accent-1);
        border-radius: var(--radius-pill);
        padding: 0.22rem 0.6rem;
        font-size: 0.77rem;
        font-family: var(--font-mono);
        margin-bottom: 0.6rem;
        letter-spacing: 0.04em;
    }
    .session-pill::before { content: "#"; opacity: 0.5; }

    /* ─── Topic Chips ─── */
    .topic-chip {
        background: rgba(128,128,128,0.04);
        border: 1px solid var(--panel-border);
        border-left: 2px solid var(--accent-1);
        border-radius: var(--radius-sm);
        color: var(--text-mid);
        padding: 0.36rem 0.5rem 0.36rem 0.65rem;
        margin-bottom: 0.32rem;
        font-size: 0.81rem;
        line-height: 1.4;
        transition: all 0.18s ease;
        cursor: default;
    }

    .topic-chip:hover {
        background: rgba(79,159,254,0.07);
        border-left-color: var(--accent-2);
        color: var(--text-hi);
        transform: translateX(2px);
        box-shadow: -3px 0 10px rgba(79,159,254,0.14);
    }

    /* ─── BUTTONS — Animated Premium Style ─── */
    .stButton > button {
        position: relative;
        overflow: hidden;
        width: 100%;
        border-radius: var(--radius-md) !important;
        border: 1px solid rgba(79,159,254,0.35) !important;
        background: linear-gradient(135deg,
            rgba(79,159,254,0.15) 0%,
            rgba(162,89,255,0.10) 100%
        ) !important;
        color: var(--text-hi) !important;
        font-family: var(--font-heading) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.02em !important;
        min-height: 2.55rem !important;
        cursor: pointer !important;
        transition:
            transform 0.22s cubic-bezier(0.34,1.56,0.64,1),
            box-shadow 0.22s ease,
            border-color 0.22s ease !important;
        box-shadow:
            0 0 0 1px rgba(255,255,255,0.05),
            0 4px 14px rgba(0,0,0,0.18),
            inset 0 1px 0 rgba(255,255,255,0.10) !important;
    }

    /* Shimmer sweep on hover */
    .stButton > button::before {
        content: "";
        position: absolute;
        top: 0; left: -100%;
        width: 60%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(255,255,255,0.12) 50%,
            transparent 100%
        );
        transform: skewX(-18deg);
        transition: left 0.45s ease;
        pointer-events: none;
    }

    .stButton > button:hover::before {
        left: 160%;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.01) !important;
        border-color: rgba(79,159,254,0.72) !important;
        box-shadow:
            0 0 0 1px rgba(79,159,254,0.15),
            0 0 22px rgba(79,159,254,0.25),
            0 8px 28px rgba(0,0,0,0.22),
            inset 0 1px 0 rgba(255,255,255,0.14) !important;
        background: linear-gradient(135deg,
            rgba(79,159,254,0.24) 0%,
            rgba(162,89,255,0.18) 100%
        ) !important;
    }

    .stButton > button:active {
        transform: translateY(-1px) scale(0.99) !important;
        transition-duration: 0.08s !important;
    }

    /* ─── Chat Messages ─── */
    [data-testid="stChatMessage"] {
        background: var(--chat-bg) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: var(--radius-md) !important;
        padding: 0.5rem 0.65rem !important;
        margin-bottom: 0.5rem !important;
        backdrop-filter: blur(8px) !important;
        animation: chat-enter 280ms cubic-bezier(0.22,1,0.36,1) both;
        transition: background 0.35s ease, box-shadow 0.2s ease;
    }

    [data-testid="stChatMessage"]:hover {
        box-shadow: 0 0 0 1px var(--panel-border-m), var(--shadow-sm) !important;
    }

    /* User — blue tint */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        border-left: 2px solid var(--accent-1) !important;
        background: var(--chat-user-bg) !important;
    }

    /* Assistant — violet tint */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-left: 2px solid var(--accent-2) !important;
        background: var(--chat-ai-bg) !important;
    }

    /* ─── Chat Input ─── */
    [data-testid="stChatInput"] {
        background: var(--input-bg) !important;
        border: 1px solid var(--panel-border-m) !important;
        border-radius: var(--radius-lg) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.35s ease !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(79,159,254,0.6) !important;
        box-shadow: 0 0 0 3px rgba(79,159,254,0.1), 0 0 20px rgba(79,159,254,0.12) !important;
    }

    /* ─── Meta Row ─── */
    .meta-row {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        margin-top: 0.5rem;
        background: rgba(79,159,254,0.09);
        border: 1px solid rgba(79,159,254,0.22);
        color: var(--accent-1);
        border-radius: var(--radius-pill);
        padding: 0.2rem 0.6rem;
        font-size: 0.79rem;
        font-family: var(--font-mono);
        letter-spacing: 0.03em;
    }
    .meta-row::before { content: "⬡"; font-size: 0.65rem; opacity: 0.7; }

    /* ─── Divider ─── */
    hr {
        border: none !important;
        border-top: 1px solid var(--panel-border) !important;
        margin: 0.85rem 0 !important;
    }

    /* ─── Streamlit overrides ─── */
    .stSpinner > div {
        border-color: var(--accent-1) transparent transparent transparent !important;
    }

    /* ─── Animations ─── */
    @keyframes chat-enter {
        from { transform: translateY(10px) scale(0.99); opacity: 0; }
        to   { transform: translateY(0)   scale(1);    opacity: 1; }
    }

    @keyframes glow-pulse {
        0%, 100% { box-shadow: 0 0 12px rgba(79,159,254,0.18); }
        50%       { box-shadow: 0 0 28px rgba(79,159,254,0.38); }
    }

    @keyframes fade-up {
        from { transform: translateY(16px); opacity: 0; }
        to   { transform: translateY(0);    opacity: 1; }
    }

    /* ─── Responsive ─── */
    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 1.1rem 1.5rem !important;
            border-radius: var(--radius-lg) !important;
        }
        .hero-shell { padding: 1.1rem 1.2rem 1.2rem; }
        .hero-shell h1 { font-size: clamp(1.3rem, 7vw, 1.8rem); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="custom-navbar">
      <div class="navbar-left">
        <span class="navbar-title">Transformer Intelligence Desk</span>
        <span class="navbar-desc">{DOMAIN_DESCRIPTION}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_agent():
    # Shared backend lives in agent.py for submission deliverable alignment.
    return load_backend_agent(DOCUMENTS, collection_name="capstone_kb")


try:
    agent_app, collection, topics = load_agent()
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]


with st.sidebar:
    st.markdown('<div class="sidebar-top">', unsafe_allow_html=True)
    st.markdown("### About")
    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()
    st.markdown(f'<div class="sidebar-card">{DOMAIN_DESCRIPTION}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="session-pill">Session: {st.session_state.thread_id}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("### Topics Covered")
    for topic in topics:
        st.markdown(f'<div class="topic-chip">{topic}</div>', unsafe_allow_html=True)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


if prompt := st.chat_input("Ask something about the Transformer paper..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "Sorry, I could not generate an answer.")
        st.write(answer)

        faith = result.get("faithfulness", 0.0)
        sources = result.get("sources", [])
        if faith > 0:
            if sources:
                source_preview = ", ".join(sources[:3])
                if len(sources) > 3:
                    source_preview += f" (+{len(sources) - 3} more)"
            else:
                source_preview = "No sources"
            st.markdown(
                f'<div class="meta-row">Faithfulness: {faith:.2f} | Sources: {source_preview}</div>',
                unsafe_allow_html=True,
            )

    st.session_state.messages.append({"role": "assistant", "content": answer})
