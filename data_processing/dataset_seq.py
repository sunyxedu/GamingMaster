import json
import os

import torch
from torch.utils.data import Dataset


def collate_variable_length_sequences(batch):
    """
    Custom collate function for variable-length sequences.
    Pads sequences to the same length within a batch.
    """
    # Find max sequence length in this batch
    max_T = max(item['meta']['T'] for item in batch)
    
    # Get dimensions from first item
    H = batch[0]['meta']['H']
    W = batch[0]['meta']['W']
    K = batch[0]['meta']['K']
    C = batch[0]['obs']['grid_seq'].shape[2]
    G = batch[0]['obs']['globals_seq'].shape[2]
    
    B = len(batch)
    
    # Initialize padded tensors
    collated = {
        'obs': {
            'grid_seq': torch.zeros(B, max_T, C, H, W),
            'globals_seq': torch.zeros(B, max_T, G),
            'legal_source_mask_seq': torch.zeros(B, max_T, 1, H, W, dtype=torch.bool)
        },
        'tgt': {
            'count': torch.zeros(B, max_T, dtype=torch.long),
            'src_y': torch.zeros(B, max_T, K, dtype=torch.long),
            'src_x': torch.zeros(B, max_T, K, dtype=torch.long),
            'dir_id': torch.zeros(B, max_T, K, dtype=torch.long),
            'ratio_id': torch.zeros(B, max_T, K, dtype=torch.long),
            'value': torch.zeros(B, max_T),
            'enemy_heatmap': torch.zeros(B, max_T, 1, H, W),
            'next_visibility': torch.zeros(B, max_T, 1, H, W)
        },
        'meta': {
            'lengths': [],
            'files': [],
            # Tick features collated here for convenience (optional use)
            'tick_seq': torch.zeros(B, max_T),
            'tick_bucket_seq': torch.zeros(B, max_T)
        }
    }
    
    # Fill in actual values
    for i, item in enumerate(batch):
        T = item['meta']['T']
        collated['obs']['grid_seq'][i, :T] = item['obs']['grid_seq'][0]
        collated['obs']['globals_seq'][i, :T] = item['obs']['globals_seq'][0]
        collated['obs']['legal_source_mask_seq'][i, :T] = item['obs']['legal_source_mask_seq'][0]
        
        collated['tgt']['count'][i, :T] = item['tgt']['count'][0]
        collated['tgt']['src_y'][i, :T] = item['tgt']['src_y'][0]
        collated['tgt']['src_x'][i, :T] = item['tgt']['src_x'][0]
        collated['tgt']['dir_id'][i, :T] = item['tgt']['dir_id'][0]
        collated['tgt']['ratio_id'][i, :T] = item['tgt']['ratio_id'][0]
        collated['tgt']['value'][i, :T] = item['tgt']['value'][0]
        collated['tgt']['enemy_heatmap'][i, :T] = item['tgt']['enemy_heatmap'][0]
        collated['tgt']['next_visibility'][i, :T] = item['tgt']['next_visibility'][0]
        
        collated['meta']['lengths'].append(T)
        collated['meta']['files'].append(item['meta']['file'])
        # Collate tick sequences if present in item.meta
        if 'tick_seq' in item['meta']:
            collated['meta']['tick_seq'][i, :T] = item['meta']['tick_seq']
        if 'tick_bucket_seq' in item['meta']:
            collated['meta']['tick_bucket_seq'][i, :T] = item['meta']['tick_bucket_seq']
    
    return collated


class SeqReplayDataset(Dataset):
    """
    Dataset for sequence-based replay data.
    
    Loads JSON files in gio format with grid_seq, globals_seq, and action targets.
    """
    
    def __init__(self, root, file_extension='.json', max_sequence_length=None):
        """
        Initialize dataset.
        
        Args:
            root: Directory containing JSON replay files
            file_extension: File extension to filter (default: '.json')
            max_sequence_length: Optional max length to truncate sequences
        """
        self.root = root
        self.max_seq_len = max_sequence_length
        self.paths = [
            os.path.join(root, f) 
            for f in os.listdir(root) 
            if f.endswith(file_extension)
        ]
        self.paths.sort()
        
        if len(self.paths) == 0:
            print(f"Warning: No {file_extension} files found in {root}")
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        try:
            with open(self.paths[idx], 'r') as f:
                obj = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Warning: skipping corrupt file {self.paths[idx]}: {e}")
            # Fall back to a random other sample
            return self[torch.randint(len(self), (1,)).item()]

        # Load grid sequence [T, C, H, W]
        grid = torch.tensor(obj['grid_seq'], dtype=torch.float32)
        T, C, H, W = grid.shape
        
        # Optionally truncate sequence
        if self.max_seq_len and T > self.max_seq_len:
            T = self.max_seq_len
            grid = grid[:T]
        
        # Load global features [T, G]
        globals_ = torch.tensor(
            obj.get('globals_seq', [[0.0] * 4 for _ in range(T)]),
            dtype=torch.float32
        )[:T]
        
        # Load legal source mask [T, 1, H, W]
        lsrc = torch.tensor(
            obj.get('legal_source_mask_seq', [[[[True] * W for _ in range(H)]] for _ in range(T)]),
            dtype=torch.bool
        )[:T]
        
        # Action parameters
        K = obj.get('max_actions_per_tick', 4)
        
        # Load action targets
        count = torch.tensor(
            obj.get('count', [0] * T),
            dtype=torch.long
        )[:T]
        
        src_y = torch.tensor(
            obj.get('src_y', [[0] * K for _ in range(T)]),
            dtype=torch.long
        )[:T]
        
        src_x = torch.tensor(
            obj.get('src_x', [[0] * K for _ in range(T)]),
            dtype=torch.long
        )[:T]
        
        dir_id = torch.tensor(
            obj.get('dir_id', [[0] * K for _ in range(T)]),
            dtype=torch.long
        )[:T]
        
        ratio_id = torch.tensor(
            obj.get('ratio_id', [[0] * K for _ in range(T)]),
            dtype=torch.long
        )[:T]
        
        value = torch.tensor(
            obj.get('value', [0.0] * T),
            dtype=torch.float32
        )[:T]
        
        # Load auxiliary targets [T, 1, H, W]
        enemy = torch.tensor(
            obj.get('enemy_heatmap', [[[[0.0] * W for _ in range(H)]] for _ in range(T)]),
            dtype=torch.float32
        )[:T]  # Already has shape [T, 1, H, W]
        
        nextv = torch.tensor(
            obj.get('next_visibility', [[[[0.0] * W for _ in range(H)]] for _ in range(T)]),
            dtype=torch.float32
        )[:T]  # Already has shape [T, 1, H, W]
        
        # Load tick sequences if present (from converters)
        tick_seq = torch.tensor(
            obj.get('tick_seq', list(range(T))),
            dtype=torch.float32
        )[:T]
        tick_bucket_seq = torch.tensor(
            obj.get('tick_bucket_seq', [int(((t % 50) + 1) // 2) for t in range(T)]),
            dtype=torch.float32
        )[:T]

        return {
            'obs': {
                'grid_seq': grid.unsqueeze(0),              # [1, T, C, H, W]
                'globals_seq': globals_.unsqueeze(0),       # [1, T, G]
                'legal_source_mask_seq': lsrc.unsqueeze(0)  # [1, T, 1, H, W]
            },
            'tgt': {
                'count': count.unsqueeze(0),                # [1, T]
                'src_y': src_y.unsqueeze(0),                # [1, T, K]
                'src_x': src_x.unsqueeze(0),                # [1, T, K]
                'dir_id': dir_id.unsqueeze(0),              # [1, T, K]
                'ratio_id': ratio_id.unsqueeze(0),          # [1, T, K]
                'value': value.unsqueeze(0),                # [1, T]
                'enemy_heatmap': enemy.unsqueeze(0),        # [1, T, 1, H, W]
                'next_visibility': nextv.unsqueeze(0)       # [1, T, 1, H, W]
            },
            'meta': {
                'H': H,
                'W': W,
                'K': K,
                'T': T,
                'file': os.path.basename(self.paths[idx]),
                # Expose tick sequences in meta for optional consumption
                'tick_seq': tick_seq,
                'tick_bucket_seq': tick_bucket_seq
            }
        }
