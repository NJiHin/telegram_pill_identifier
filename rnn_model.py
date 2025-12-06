import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Bidirectional GRU encoder for processing input sequence"""

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - input sequence with coordinates and features
        Returns:
            outputs: (batch, seq_len, hidden_dim*2) - encoder outputs
            hidden: (2, batch, hidden_dim) - final hidden states (forward and backward)
        """
        outputs, hidden = self.gru(x)
        outputs = self.dropout(outputs)
        return outputs, hidden


class Attention(nn.Module):
    """Additive (Bahdanau) attention mechanism"""

    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.W1 = nn.Linear(encoder_dim, decoder_dim)
        self.W2 = nn.Linear(decoder_dim, decoder_dim)
        self.V = nn.Linear(decoder_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: (batch, decoder_dim) - current decoder hidden state
            encoder_outputs: (batch, src_len, encoder_dim) - encoder outputs
            mask: (batch, src_len) - mask for padded positions (1 for valid, 0 for padding)
        Returns:
            context: (batch, encoder_dim) - weighted context vector
            attention_weights: (batch, src_len) - attention distribution
        """
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        # (batch, src_len, decoder_dim)
        encoder_transform = self.W1(encoder_outputs)

        # (batch, 1, decoder_dim)
        decoder_transform = self.W2(decoder_hidden).unsqueeze(1)

        # (batch, src_len, decoder_dim)
        scores = self.V(torch.tanh(encoder_transform + decoder_transform))

        # (batch, src_len)
        scores = scores.squeeze(2)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)

        # (batch, src_len)
        attention_weights = F.softmax(scores, dim=1)

        # (batch, encoder_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


class Decoder(nn.Module):
    """GRU decoder with attention for generating corrected imprints"""

    def __init__(self, vocab_size, embedding_dim, encoder_dim, decoder_dim, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(encoder_dim, decoder_dim)
        self.gru = nn.GRU(embedding_dim + encoder_dim, decoder_dim, batch_first=True)
        self.fc_out = nn.Linear(encoder_dim + decoder_dim + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, encoder_outputs, mask=None):
        """
        Args:
            input_token: (batch,) - current input token indices
            hidden: (1, batch, decoder_dim) - previous hidden state
            encoder_outputs: (batch, src_len, encoder_dim) - encoder outputs
            mask: (batch, src_len) - mask for padded positions
        Returns:
            output: (batch, vocab_size) - predicted token distribution
            hidden: (1, batch, decoder_dim) - updated hidden state
            attention_weights: (batch, src_len) - attention weights
        """
        # (batch, 1, embedding_dim)
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))

        # Get attention context
        # hidden: (1, batch, decoder_dim) -> (batch, decoder_dim)
        context, attention_weights = self.attention(hidden.squeeze(0), encoder_outputs, mask)

        # (batch, 1, embedding_dim + encoder_dim)
        gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)

        # (batch, 1, decoder_dim), (1, batch, decoder_dim)
        output, hidden = self.gru(gru_input, hidden)

        # Prepare for output projection
        # (batch, embedding_dim + encoder_dim + decoder_dim)
        prediction = torch.cat([
            output.squeeze(1),
            context,
            embedded.squeeze(1)
        ], dim=1)

        # (batch, vocab_size)
        output = self.fc_out(prediction)

        return output, hidden, attention_weights


class Seq2SeqWithAttention(nn.Module):
    """Complete sequence-to-sequence model for pill imprint correction"""

    def __init__(self, input_dim, vocab_size, embedding_dim=45, hidden_dim=256,
                 dropout=0.1, sos_idx=0, eos_idx=1):
        super().__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        # Encoder: bidirectional -> output is 2*hidden_dim
        self.encoder = Encoder(input_dim, hidden_dim, dropout)

        # Decoder
        encoder_dim = hidden_dim * 2  # bidirectional
        self.decoder = Decoder(vocab_size, embedding_dim, encoder_dim, hidden_dim, dropout)

        # Bridge to combine bidirectional encoder hidden states
        self.bridge = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5, src_mask=None):
        """
        Args:
            src: (batch, src_len, input_dim) - input sequence
            trg: (batch, trg_len) - target sequence (token indices)
            teacher_forcing_ratio: probability of using teacher forcing
            src_mask: (batch, src_len) - mask for source sequence
        Returns:
            outputs: (batch, trg_len, vocab_size) - predicted distributions
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)

        # Encode
        encoder_outputs, encoder_hidden = self.encoder(src)

        # Bridge: combine forward and backward hidden states
        # encoder_hidden: (2, batch, hidden_dim) -> (batch, hidden_dim*2)
        encoder_hidden = encoder_hidden.permute(1, 0, 2).contiguous().view(batch_size, -1)
        decoder_hidden = torch.tanh(self.bridge(encoder_hidden)).unsqueeze(0)

        # Prepare output tensor
        outputs = torch.zeros(batch_size, trg_len, self.vocab_size).to(src.device)

        # First input to decoder is SOS token (use first target token)
        input_token = trg[:, 0]

        # Decode
        for t in range(1, trg_len):
            output, decoder_hidden, _ = self.decoder(
                input_token, decoder_hidden, encoder_outputs, src_mask
            )
            outputs[:, t, :] = output

            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if use_teacher_forcing else top1

        return outputs

    def predict(self, src, max_len=50, src_mask=None, temperature=1.0):
        """
        Greedy decoding for inference with proper early stopping

        Args:
            src: (batch, src_len, input_dim) - input sequence
            max_len: maximum output length
            src_mask: (batch, src_len) - mask for source sequence
            temperature: softmax temperature for sampling (1.0 = greedy)
        Returns:
            predictions: list of (batch,) tensors - predicted token indices
            attention_weights: list of attention weights at each step
            lengths: (batch,) - actual lengths of predictions before EOS
        """
        batch_size = src.size(0)
        device = src.device

        # Encode
        encoder_outputs, encoder_hidden = self.encoder(src)

        # Bridge
        encoder_hidden = encoder_hidden.permute(1, 0, 2).contiguous().view(batch_size, -1)
        decoder_hidden = torch.tanh(self.bridge(encoder_hidden)).unsqueeze(0)

        # Start with SOS token
        input_token = torch.full((batch_size,), self.sos_idx, dtype=torch.long, device=device)

        predictions = []
        attention_weights_list = []

        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        lengths = torch.full((batch_size,), max_len, dtype=torch.long, device=device)

        for t in range(max_len):
            output, decoder_hidden, attention_weights = self.decoder(
                input_token, decoder_hidden, encoder_outputs, src_mask
            )

            # Apply temperature
            if temperature != 1.0:
                output = output / temperature

            # Greedy decoding
            top1 = output.argmax(1)

            # Mark sequences that just finished
            just_finished = (top1 == self.eos_idx) & ~finished
            lengths[just_finished] = t + 1
            finished = finished | just_finished

            predictions.append(top1)
            attention_weights_list.append(attention_weights)

            # Next input (replace with PAD for finished sequences)
            input_token = top1
            input_token[finished] = self.eos_idx

            # Stop if all sequences have finished
            if finished.all():
                break

        # Stack predictions: (batch, seq_len)
        predictions = torch.stack(predictions, dim=1)

        return predictions, attention_weights_list, lengths
