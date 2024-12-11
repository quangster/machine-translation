import numpy as np
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size: int, emb_size: int, padding_idx: int=0):
        """ Token Embedding

        Args:
            vocab_size (int): vocabulary size
            emb_size (int): embedding size
            padding_idx (int): padding index. Defaults to 0.
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """ Token Embedding Forward

        Args:
            tokens (torch.Tensor): token indexes

        Returns:
            torch.Tensor: embedded tensor
        """

        # tokens: (batch_size, seq_len)
        output = self.embedding(tokens.long()) * np.sqrt(self.emb_size)
        # output: (batch_size, seq_len, emb_size)
        return output
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000):
        """ Positional Encoding

        Args:
            d_model (int): embedding size
            dropout (float): dropout value. Defaults to 0.1.
            max_len (int): input lengthsequence. Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Positional Encoding Forward

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        # x: (batch_size, seq_len, d_model)
        output = x + self.pe[:, :x.size(1)]
        # output: (batch_size, seq_len, d_model)
        return self.dropout(output)


class TransformerTorch(nn.Module):

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        """ Transformer Model with PyTorch

        Args:
            num_encoder_layers (int): number of encoder layers
            num_decoder_layers (int): number of decoder layers
            emb_size (int): embedding size
            nhead (int): number of heads
            src_vocab_size (int): source vocabulary size
            tgt_vocab_size (int): target vocabulary size
            dim_feedforward (int, optional): the dimension of the feedforward network model. Defaults to 512.
            dropout (float, optional): dropout value. Defaults to 0.1.
        """
        super(TransformerTorch, self).__init__()
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)


    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """ Transformer Forward

        Args:
            src (torch.Tensor): source tensor
            tgt (torch.Tensor): target tensor
            src_mask (torch.Tensor): the additive mask for the src sequence
            tgt_mask (torch.Tensor): the additive mask for the tgt sequence
            src_key_padding_mask (torch.Tensor): the Tensor mask for src keys per batch
            tgt_key_padding_mask (torch.Tensor): the Tensor mask for tgt keys per batch
            memory_key_padding_mask (torch.Tensor): the Tensor mask for memory keys per batch

        Returns:
            torch.Tensor: output tensor
        """
        # src: (batch_size, src_seq_len)
        # tgt: (batch_size, tgt_seq_len)
        # src_mask: (src_seq_len, src_seq_len)
        # tgt_mask: (tgt_seq_len, tgt_seq_len)
        # src_key_padding_mask: (batch_size, src_seq_len)
        # tgt_key_padding_mask: (batch_size, tgt_seq_len)
        # memory_key_padding_mask: (batch_size, src_seq_len
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        # src_emb: (batch_size, src_seq_len, emb_size)
        # tgt_emb: (batch_size, tgt_seq_len, emb_size)
        outs = self.transformer(
            src_emb, tgt_emb, src_mask, tgt_mask, None,
            src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask
        )
        # outs: (batch_size, tgt_seq_len, emb_size)
        return self.fc(outs)


    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """ Encode

        Args:
            src (torch.Tensor): source tensor
            src_mask (torch.Tensor): the additive mask for the src sequence

        Returns:
            torch.Tensor: output tensor
        """
        # src: (batch_size, src_seq_len)
        # src_mask: (src_seq_len, src_seq_len)
        embedding = self.positional_encoding(self.src_tok_emb(src))
        # embedding: (batch_size, src_seq_len, emb_size)
        output = self.transformer.encoder(embedding, src_mask)
        # output: (batch_size, src_seq_len, emb_size)
        return output


    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """ Decode

        Args:
            tgt (torch.Tensor): target tensor
            memory (torch.Tensor): memory tensor
            tgt_mask (torch.Tensor): the additive mask for the tgt sequence

        Returns:
            torch.Tensor: output tensor
        """
        # tgt: (batch_size, tgt_seq_len)
        # memory: (batch_size, src_seq_len, emb_size)
        # tgt_mask: (tgt_seq_len, tgt_seq_len)

        embedding = self.positional_encoding(self.tgt_tok_emb(tgt))
        # embedding: (batch_size, tgt_seq_len, emb_size)
        output = self.transformer.decoder(embedding, memory, tgt_mask)
        # output: (batch_size, tgt_seq_len, emb_size)
        return output
    

    @staticmethod
    def create_mask(src: torch.Tensor, tgt: torch.Tensor, padding_idx: int=0, device='cpu') -> list[torch.Tensor]:
        """ Create Mask

        Args:
            src (torch.Tensor): source tensor
            tgt (torch.Tensor): target tensor
            padding_idx (int): padding index. Defaults to 0.
            device (str): device. Defaults to 'cpu'.

        Returns:
            list[torch.Tensor]: mask list
        """
        # src: (batch_size, src_seq_len)
        # tgt: (batch_size, tgt_seq_len)
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=device)
        # tgt_mask: (tgt_seq_len, tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)
        # src_mask: (src_seq_len, src_seq_len)
        src_key_padding_mask = (src == padding_idx)
        # src_key_padding_mask: (batch_size, src_seq_len)
        tgt_key_padding_mask = (tgt == padding_idx)
        # tgt_key_padding_mask: (batch_size, tgt_seq_len)
        return src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask