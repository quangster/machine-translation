import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        x = x.permute(1, 0)
        # x: (seq_length, N)
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)
        encoder_states, (hidden, cell) = self.rnn(embedding)
        # encoder_states: (seq_length, N, hidden_size * num_directions)
        # hidden: (num_layers * num_directions, N, hidden_size)
        # cell: (num_layers * num_directions, N, hidden_size)
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))
        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Dropout layer
        self.dropout = nn.Dropout(p)

        # Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_size)

        # LSTM layer
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers, dropout=p)

        # Attention mechanism
        self.energy = nn.Linear(hidden_size * 3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

        # Fully connected layer for output predictions
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states, hidden, cell):
        # x shape: (N) but we want (1, N)
        x = x.unsqueeze(0)

        # Embedding and dropout
        embedding = self.dropout(self.embedding(x)) # embeding shape (1, N, embedding_size)

        # Attention mechanism
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        attention = self.softmax(energy) # (seq_length, N, 1)
        attention = attention.permute(1, 2, 0) # (N, 1, seq_length)

        encoder_states = encoder_states.permute(1, 0, 2)  # (N, seq_length, hidden_size * 2)

        # Context vector calculation
        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)

        # Concatenate context vector and embedding
        rnn_input = torch.cat((context_vector, embedding), dim=2)

        # Decoder pass
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))     # outputs shape (1, n, hidden_size)

        predictions = self.fc(outputs) # shape of predictions: (1, N, length_of_vocab)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_size):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_size = target_vocab_size
    
    def forward(self, source, target):
        batch_size = source.shape[0]
        target_len = target.shape[1]

        device = source.device

        outputs = torch.zeros(batch_size, target_len, self.target_vocab_size).to(device)

        # Encoder pass
        encoder_states, hidden, cell = self.encoder(source)

        x = target[:, 0]

        for t in range(0, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[:, t, :] = output
            # best_guess = output.argmax(1)
            x = target[:, t]

        return outputs
