import pandas as pd
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ChessGPT.utils import gpt_utils as gpt

from tqdm import tqdm

def train():
    # load in data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    game_df = pd.read_csv(os.path.join(BASE_DIR, '../shared_resources/games.csv'))
    game_df = game_df[['moves']]
    game_df['fen_tokens'] = game_df['moves'].apply(lambda s: [gpt.fen_to_tokens(f) for f in gpt.moves_to_fens(s)])

    '''
    # generate and store vocab
    move_vocab = gpt.build_move_vocab(game_df)
    vocab_path = os.path.join(BASE_DIR, 'utils/move_vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(move_vocab, f, indent=2)
    '''
    
    # pull in vocab
    vocab_path = os.path.join(BASE_DIR, 'utils/move_vocab.json')
    with open(vocab_path, 'r') as f:
        move_vocab = json.load(f)

    dataset = gpt.ChessDataset(game_df, move_vocab)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # hard coding params
    params = {
    'd_model': 128,
    'nhead': 8,
    'num_encoder_layers': 1,
    'num_decoder_layers': 1,
    'lr': 0.0004806284511583589
    }

    # set up model, optimizer, & loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model hyperparameters hardcoded from optuna tuning (results in utils.best_hyperparams.json)
    model = gpt.ChessTransformer(
        board_vocab_size=19,
        move_vocab_size=len(move_vocab),
        d_model=params['d_model'],
        nhead=params['nhead'],
        num_encoder_layers=params['num_encoder_layers'],
        num_decoder_layers=params['num_decoder_layers'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004806284511583589)
    criterion = nn.CrossEntropyLoss(ignore_index=move_vocab['<PAD>'])

    # training loop
    num_epochs = 10
    print(f'Starting training on {device} for {num_epochs} epochs...')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for batch_idx, (b_tokens, m_tokens) in enumerate(progress_bar):
            b_tokens, m_tokens = [x.to(device) for x in (b_tokens, m_tokens)]
            optimizer.zero_grad()

            m_tokens = m_tokens.unsqueeze(1)
            logits = model(b_tokens, m_tokens)
            target = m_tokens.squeeze(1).clamp(0, logits.size(-1) - 1)

            loss = criterion(logits.squeeze(1), target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_loss:.4f}')

        # save checkpoint
        ckpt_path = os.path.join(BASE_DIR, f'utils/checkpoints/model_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'params': params
        }, ckpt_path)
    print('Training Complete')

if __name__ == '__main__':
    train()