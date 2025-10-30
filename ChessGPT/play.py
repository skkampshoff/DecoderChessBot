import chess
import random
from .interface import Interface

import torch
import torch.nn as nn
import json
import os

this_script_dir = os.path.dirname(os.path.abspath(__file__)) # path to this script's directory
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load move mapping
mti_path = os.path.join(this_script_dir, 'move_to_id.json')
with open(mti_path, "r") as f:
    move_to_id = json.load(f)
id_to_move = {v: k for k, v in move_to_id.items()}

PAD_ID = move_to_id["<PAD>"]
START_ID = move_to_id["<START>"]
END_ID = move_to_id["<END>"]

vocab_size = len(move_to_id)

# Load best hyperparams
bh_path = os.path.join(this_script_dir, 'best_hparams.json')
with open(bh_path, "r") as f:
    best_hparams = json.load(f)
params = best_hparams["best_params"]

MAX_LEN = 200  # same length used in training

# =============== Model Definition ===============
class ChessDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, max_len=200):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x shape: [batch, seq_len] → transformer expects [seq_len, batch]
        x = x.transpose(0, 1)
        seq_len, batch_size = x.size()

        # Add embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
        x = self.embed(x) + self.pos_embed(positions)

        # Decoder masking: prevent attention to future tokens
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        x = self.decoder(x, x, tgt_mask=mask)
        logits = self.fc_out(x)  # [seq_len, batch, vocab_size]
        return logits.transpose(0, 1)  # [batch, seq_len, vocab_size]

model = ChessDecoder(
    vocab_size=vocab_size,
    d_model=params["d_model"],
    nhead=params["nhead"],
    num_layers=params["num_layers"],
    max_len=MAX_LEN
).to(device)

# model.load_state_dict(torch.load("best_model.pt", map_location=device))
model_path = os.path.join(this_script_dir, '500k_model.pt')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# =============== Helpers ===============

def encode_moves(moves):
    """Convert a list of SAN moves into token ids with padding."""
    tokens = [START_ID] + [move_to_id[m] for m in moves] + [END_ID]
    if len(tokens) < MAX_LEN:
        tokens += [PAD_ID] * (MAX_LEN - len(tokens))
    else:
        tokens = tokens[:MAX_LEN]
    return torch.tensor(tokens, dtype=torch.long)

def get_legal_model_move(board, moves, k=10):
    x = encode_moves(moves).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)  # [1, seq_len, vocab]
    next_token_logits = logits[0, len(moves), :]  # take next token prediction
    probs = torch.softmax(next_token_logits, dim=-1)

    topk = torch.topk(probs, k)
    top_ids = topk.indices.tolist()

    legal_sans = {board.san(m): m for m in board.legal_moves}
    for move_id in top_ids:
        move_str = id_to_move.get(move_id)
        if move_str in legal_sans:
            return legal_sans[move_str]  # First legal move found

    return None  # No legal move found


def play(interface: Interface, color = "w"):
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = chess.Board(fen)
    moves = []

    if color == "b":
        move = interface.input()
        board.push_san(move)

    while True:
        bot_move = get_legal_model_move(board, moves, k=25)
        if bot_move and bot_move in [board.san(m) for m in board.legal_moves]:
            interface.output(board.san(bot_move))
            board.push(bot_move)
        else: 
            all_moves = list(board.legal_moves)
            best_move = random.choice(all_moves)
            interface.output(board.san(best_move))
            board.push(best_move)

        move = interface.input()
        board.push_san(move)
