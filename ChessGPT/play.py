import chess
import random
from .interface import Interface
import pandas as pd

import torch
import torch.nn as nn
import json
import os
from ChessGPT.utils import gpt_utils as gpt

this_script_dir = os.path.dirname(os.path.abspath(__file__)) # path to this script's directory

MAX_LEN = 400  # same length used in training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load move mapping
mti_path = os.path.join(this_script_dir, 'utils/move_vocab.json')
with open(mti_path, "r") as f:
    move_to_id = json.load(f)
id_to_move = {v: k for k, v in move_to_id.items()}

PAD_ID = move_to_id["<PAD>"]
START_ID = move_to_id["<SOS>"]
END_ID = move_to_id["<EOS>"]

vocab_size = len(move_to_id)

# Load best hyperparams
bh_path = os.path.join(this_script_dir, 'utils/best_hyperparams.json')
with open(bh_path, "r") as f:
    params = json.load(f)


# load in model
model_path = os.path.join(this_script_dir, 'utils', 'checkpoints', 'model_epoch_10.pt')
model = gpt.ChessTransformer(
    board_vocab_size=19,
    move_vocab_size=vocab_size,
    d_model=params['d_model'],
    nhead=params['nhead'],
    num_encoder_layers=params['num_encoder_layers'],
    num_decoder_layers=params['num_decoder_layers'],
    max_move_len=MAX_LEN
).to(device)

# load weights
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()
gpt.model = model

# generate move prediction
def get_legal_model_move(board, moves, k=10):
    '''
    Generate a model-predicted legal move for the current board state.

    Args: 
        board (chess.Board): Current board position.
        moves (list[str]): Move history in SAN format.
        k (int): Top-k moves to sample from

    Returns:
        chess.Move or None
    '''

    # encode current board
    board_tokens = gpt.fen_to_tokens(board.fen()).unsqueeze(0).to(device)

    # encode move history
    move_ids = []
    for m in moves[-MAX_LEN:]: # truncate when needed
        move_ids.append(move_to_id.get(m, move_to_id['<UNK>']))
    move_ids = [START_ID] + move_ids[-(MAX_LEN - 2):] + [END_ID]
    move_tensor = torch.tensor(move_ids, dtype=torch.long, device=device).unsqueeze(0)

    # legal move mask
    legal_sans = {board.san(m): m for m in board.legal_moves}
    legal_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)

    unk_id = move_to_id.get("<UNK>", None)

    for move_str in legal_sans.keys():
        move_id = move_to_id.get(move_str, unk_id)
        if move_id is not None:
            legal_mask[move_id] = True
    
    # forward pass through model
    with torch.no_grad():
        logits = gpt.model(board_tokens, move_tensor, legal_moves_mask=legal_mask)
        next_token_logits = logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)

    # restrict to legal moves
    legal_probs = probs * legal_mask
    legal_probs /= legal_probs.sum() + 1e-8 # normalize

    # select top k legal moves
    topk = torch.topk(legal_probs, k=min(k, legal_mask.sum().item()))
    top_indices = topk.indices.tolist()
    top_moves = [id_to_move[i] for i in top_indices if id_to_move[i] in legal_sans]

    # choose from top-k
    for move_str in top_moves:
        try:
            move = legal_sans[move_str]
            return move
        except KeyError:
            continue
    return None


def play(interface: Interface, color = "w"):
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = chess.Board(fen)
    moves = []

    # set up opening db things
    opening_db = pd.read_csv(os.path.join(this_script_dir, '../shared_resources/openings_fen7.csv'))
    opening_db = opening_db[opening_db['winning_percentage'] > 50]
    opening_db = opening_db.sort_values('winning_percentage', ascending=False)
    opening_db = opening_db.drop_duplicates(subset='fen', keep='first')
    openings = {} # openings dictionary
    for _, row in opening_db.iterrows():
        openings[row['fen']] = row['best_move']
    in_opening = True

    if color == "b":
        move = interface.input()
        board.push_san(move)
        moves.append(move)

    # loop for playing openings
    while in_opening:
        move = None
        fen = board.fen()
        move_uci = openings.get(fen)
        # get move if possible
        if move_uci:
            try:
                move = chess.Move.from_uci(move_uci)
            except:
                continue

        if move:
            interface.output(board.san(move))
            board.push(move)
            moves.append(move)
        else:
            in_opening = False
            break

        move = interface.input()
        board.push_san(move)
        moves.append(move)    

    # mid & endgame loop
    while True:
        bot_move = get_legal_model_move(board, moves, k=25)
        if bot_move and bot_move in [board.san(m) for m in board.legal_moves]:
            interface.output(board.san(bot_move))
            board.push(bot_move)
            moves.append(bot_move)
        else: 
            all_moves = list(board.legal_moves)
            best_move = random.choice(all_moves)
            interface.output(board.san(best_move))
            board.push(best_move)
            moves.append(best_move)

        move = interface.input()
        board.push_san(move)
        moves.append(move)
