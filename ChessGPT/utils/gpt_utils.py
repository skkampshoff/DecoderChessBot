import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import chess
import chess.pgn
from collections import Counter

class ChessTransformer(nn.Module):
    def __init__(self, board_vocab_size=19, move_vocab_size=500, d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4, max_move_len=400):
        super().__init__()

        # encoder for board encodings
        self.board_embed = nn.Embedding(board_vocab_size, d_model)
        self.board_pos = nn.Embedding(69, d_model)  # 8x8 board + metadata
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # decoder for move prediction
        self.move_embed = nn.Embedding(move_vocab_size, d_model)
        self.move_pos = nn.Embedding(max_move_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(d_model, move_vocab_size)

    def forward(self, board_tokens, move_tokens, legal_moves_mask=None):
        batch_size, seq_len = move_tokens.shape

        move_tokens = move_tokens.clamp(0, self.move_embed.num_embeddings - 1)

        # encoder
        pos_indices = torch.arange(69, device=board_tokens.device).unsqueeze(0).expand_as(board_tokens)
        enc_input = self.board_embed(board_tokens) + self.board_pos(pos_indices)
        memory = self.encoder(enc_input)

        # decoder
        pos_indices = torch.arange(seq_len, device=board_tokens.device).unsqueeze(0).expand_as(move_tokens)
        dec_input = self.move_embed(move_tokens) + self.move_pos(pos_indices)

        # mask
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=move_tokens.device), diagonal=1)
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))

        decoded = self.decoder(dec_input, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(decoded)

        if legal_moves_mask is not None:
            expanded_mask = legal_moves_mask.unsqueeze(0).unsqueeze(0).expand(logits.size(0), logits.size(1), -1)
            logits = logits.masked_fill(~expanded_mask, float('-inf'))
            return logits

        return logits


class ChessDataset(Dataset):
    def __init__(self, df, move_vocab):
        self.samples = []

        for idx, row in df.iterrows():
            move_tokens = row['moves'].split()
            fen_tokens_list = row['fen_tokens']  # use precomputed tokens

            for board_tokens, move_san in zip(fen_tokens_list, move_tokens):
                move_token = move_vocab.get(move_san, move_vocab['<UNK>'])
                self.samples.append((board_tokens, move_token))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b, m = self.samples[idx]
        return b, torch.tensor(m)
    

def fen_to_tokens(fen: str) -> torch.Tensor:
    parts = fen.strip().split()
    board_part, turn_part, castling_part = parts[0], parts[1], parts[2]

    piece_to_id = {
        '.': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
    }

    board_tokens = []
    for row in board_part.split('/'):
        for ch in row:
            if ch.isdigit():
                board_tokens.extend([0]*int(ch))
            else:
                board_tokens.append(piece_to_id[ch])
    assert len(board_tokens) == 64, f'Expected 64 squares, got {len(board_tokens)}'

    turn_token = 13 if turn_part == 'w' else 14
    castling_vec = [token_id if symbol in castling_part else 0 for symbol, token_id in zip('KQkq', range(15, 19))]

    all_tokens = [turn_token] + castling_vec + board_tokens
    return torch.tensor(all_tokens, dtype=torch.long)


def build_move_vocab(df, min_freq=5) -> dict:
    all_moves = []
    for san_str in df['moves']:
        all_moves.extend(san_str.split())
    counter = Counter(all_moves)
    vocab = {move: i + 4 for i, (move, count) in enumerate(counter.items()) if count >= min_freq}
    vocab['<PAD>'] = 0
    vocab['<SOS>'] = 1
    vocab['<EOS>'] = 2
    vocab['<UNK>'] = 3
    return vocab


def moves_to_fens(moves_str) -> list:
    board = chess.Board()
    fens = []
    for move_san in moves_str.split():
        fens.append(board.fen())
        move = board.parse_san(move_san)
        board.push(move)
    return fens