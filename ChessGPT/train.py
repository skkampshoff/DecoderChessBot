import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import chess
import chess.pgn
from collections import Counter

class ChessTransformer(nn.Module):
    def __init__(self, board_vocab_size=19, move_vocab_size=500, d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4, max_move_len=400):
        super().__init__()

        # encoder for board encodings
        self.board_embed = nn.Embedding(board_vocab_size, d_model)
        self.board_pos = nn.Embedding(69, d_model) # 8x8 board
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # decoder for list of moves in san
        self.move_embed = nn.Embedding(move_vocab_size, d_model)
        self.move_pos = nn.Embedding(max_move_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # output layer
        self.fc_out = nn.Linear(d_model, move_vocab_size)

    def forward(self, board_tokens, move_tokens, legal_moves_mask=None):
        # board_tokens: [batch,69] (spatial board encoding plus some metadata)
        # move_tokens:  [batch,squ_len] (move sequence so far)
        batch_size, seq_len = move_tokens.shape

        # encoder
        pos_indices = torch.arange(69, device=board_tokens.device).unsqueeze(0).expand_as(board_tokens)
        enc_input = self.board_embed(board_tokens) + self.board_pos(pos_indices)
        memory = self.encoder(enc_input)

        # decoder
        pos_indices = torch.arange(seq_len, device=board_tokens.device).unsqueeze(0).expand_as(move_tokens)
        dec_input = self.move_embed(move_tokens) + self.move_pos(pos_indices)

        # mask
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=move_tokens.device), diagonal=1).bool()

        decoded = self.decoder(dec_input, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(decoded)

        # Apply legal moves mask if provided
        if legal_moves_mask is not None:
            # Set logits of illegal moves to large negative value
            logits = logits.transpose(0, 1)  # [batch, seq_len, vocab_size]
            # Expand mask to match logits dimensions
            expanded_mask = legal_moves_mask.unsqueeze(0).unsqueeze(0).expand(logits.size(0), logits.size(1), -1)
            logits = logits.masked_fill(~expanded_mask, float('-inf'))
            return logits

        return logits
    

# converts fen string into tensor of 69 tokens
def fen_to_tokens(fen: str) -> torch.Tensor:
    # split fen into fields
    parts = fen.strip().split()
    board_part, turn_part, castling_part = parts[0], parts[1], parts[2]

    # board mapping
    piece_to_id = {
        '.': 0,
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
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

    # turn token (13 for white 14 for black)
    turn_token = 13 if turn_part == 'w' else 14

    # castling tokens (15-18)
    castling_tokens = []
    for symbol, token_id in zip('KQkq', range(15,19)):
        if symbol in castling_part:
            castling_tokens.append(token_id)

    castling_vec = [token_id if symbol in castling_part else 0 for symbol, token_id in zip('KQkq', range(15,19))]

    # combine all tokens (total 1 + 4 + 64 = 69)
    all_tokens = [turn_token] + castling_vec + board_tokens
    return torch.tensor(all_tokens, dtype=torch.long)


# build move vocab from san to integer
def build_move_vocab(df, min_freq=1) -> dict:
    all_moves = []
    for san_str in df['moves']:
        all_moves.extend(san_str.split())

    counter = Counter(all_moves)
    vocab = {move: i + 4 for i, (move, count) in enumerate(counter.items()) if count >= min_freq}

    # special tokens
    vocab['<PAD>'] = 0
    vocab['<SOS>'] = 1
    vocab['<EOS>'] = 2
    vocab['<UNK>'] = 3

    return vocab


# tokenizes games
def tokenize_moves(san_str, vocab, max_len=400) -> list:
    moves = san_str.split()
    tokens = [vocab.get('<SOS>')]
    for move in moves:
        tokens.append(vocab.get(move,vocab['<UNK>']))
    tokens.append(vocab.get('<EOS>'))

    # pad/truncate
    if len(tokens) < max_len:
        tokens.extend([vocab['<PAD>']] * (max_len - len(tokens)))
    else:
        tokens = tokens[:max_len]

    return tokens


# generate list of fens from list of san notations
def moves_to_fens(moves_str) -> list:
    board = chess.Board()
    fens = []
    for move_san in moves_str.split():
        fens.append(board.fen())
        move = board.parse_san(move_san)
        board.push(move)
    return fens


# saves vocab for later use
def save_vocab(vocab, path):
    with open(path, 'w') as f:
        json.dump(vocab, f)


# main train script run by competition moderation / training setup
def train():
    pass

game_df = pd.read_csv('./shared_resources/games.csv')
game_df = game_df[['moves']]

move_vocab = build_move_vocab(game_df)

import optuna
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1. Tiny dataset class
# -----------------------------
class ChessDataset(Dataset):
    def __init__(self, df, move_vocab, max_len=400):
        self.df = df
        self.move_vocab = move_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        san_str = self.df.iloc[idx]['moves']
        fens = moves_to_fens(san_str)
        board_tokens = fen_to_tokens(fens[0])  # first position
        move_tokens = torch.tensor(tokenize_moves(san_str, self.move_vocab, self.max_len), dtype=torch.long)
        return board_tokens, move_tokens

# -----------------------------
# 2. Objective for Optuna
# -----------------------------
def objective(trial):
    # sample hyperparameters
    d_model = trial.suggest_categorical('d_model', [128, 256])
    nhead = trial.suggest_categorical('nhead', [4, 8])
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 3)
    num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 3)
    lr = trial.suggest_float('lr', 1e-4, 1e-3)

    # k-fold split (3 folds for example)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    val_losses = []

    for train_idx, val_idx in kf.split(game_df):
        train_subset = game_df.iloc[train_idx]
        val_subset = game_df.iloc[val_idx]

        train_ds = ChessDataset(train_subset, move_vocab)
        val_ds = ChessDataset(val_subset, move_vocab)

        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)  # tiny batch for tuning
        val_loader = DataLoader(val_ds, batch_size=2)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ChessTransformer(
            board_vocab_size=19,
            move_vocab_size=len(move_vocab),
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=move_vocab['<PAD>'])

        # --- Quick 3-epoch train for tuning ---
        for epoch in range(3):
            model.train()
            for batch in train_loader:
                b_tokens, m_tokens = [x.to(device) for x in batch]
                optimizer.zero_grad()
                logits = model(b_tokens, m_tokens[:, :-1])
                target = m_tokens[:, 1:]
                loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                loss.backward()
                optimizer.step()

        # --- Validate ---
        model.eval()
        val_loss = 0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                b_tokens, m_tokens = [x.to(device) for x in batch]
                logits = model(b_tokens, m_tokens[:, :-1])
                target = m_tokens[:, 1:]
                l = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                val_loss += l.item()
                count += 1
        val_losses.append(val_loss / count)

    return sum(val_losses) / len(val_losses)

# -----------------------------
# 3. Run Optuna study
# -----------------------------
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=10, show_progress_bar=True)

# -----------------------------
# 4. Save best hyperparameters
# -----------------------------
best_params = study.best_trial.params
with open('best_hyperparams.json', 'w') as f:
    json.dump(best_params, f)

print("Best hyperparameters:", best_params)