import os, json, torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


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
    C = batch[0]['obs']['grid_seq'].shape[2]
    G = batch[0]['obs']['globals_seq'].shape[2]
    A = batch[0]['obs']['prev_action_seq'].shape[2]
    
    B = len(batch)
    
    # Initialize padded tensors
    collated = {
        'obs': {
            'grid_seq': torch.zeros(B, max_T, C, H, W),
            'globals_seq': torch.zeros(B, max_T, G),
            'prev_action_seq': torch.zeros(B, max_T, A),
            'legal_source_mask_seq': torch.zeros(B, max_T, 1, H, W, dtype=torch.bool)
        },
        'tgt': {
            'count': torch.zeros(B, max_T, dtype=torch.long),
            'src_y': torch.zeros(B, max_T, dtype=torch.long),
            'src_x': torch.zeros(B, max_T, dtype=torch.long),
            'dir_id': torch.zeros(B, max_T, dtype=torch.long),
            'ratio_id': torch.zeros(B, max_T, dtype=torch.long),
            'value': torch.zeros(B, max_T),
            'next_visibility': torch.zeros(B, max_T, 1, H, W)
        },
        'meta': {
            'lengths': [],
            'files': []
        }
    }
    
    # Fill in actual values
    for i, item in enumerate(batch):
        T = item['meta']['T']
        collated['obs']['grid_seq'][i, :T] = item['obs']['grid_seq'][0]
        collated['obs']['globals_seq'][i, :T] = item['obs']['globals_seq'][0]
        collated['obs']['prev_action_seq'][i, :T] = item['obs']['prev_action_seq'][0]
        collated['obs']['legal_source_mask_seq'][i, :T] = item['obs']['legal_source_mask_seq'][0]
        
        collated['tgt']['count'][i, :T] = item['tgt']['count'][0]
        collated['tgt']['src_y'][i, :T] = item['tgt']['src_y'][0]
        collated['tgt']['src_x'][i, :T] = item['tgt']['src_x'][0]
        collated['tgt']['dir_id'][i, :T] = item['tgt']['dir_id'][0]
        collated['tgt']['ratio_id'][i, :T] = item['tgt']['ratio_id'][0]
        collated['tgt']['value'][i, :T] = item['tgt']['value'][0]
        collated['tgt']['next_visibility'][i, :T] = item['tgt']['next_visibility'][0]
        
        collated['meta']['lengths'].append(T)
        collated['meta']['files'].append(item['meta']['file'])
    
    return collated


class SeqReplayDataset(Dataset):
    """
    Dataset for sequence-based replay data.
    
    Loads JSON files in gio format with grid_seq, globals_seq, and action targets.
    """
    
    def __init__(self, root, file_extension='.json', max_sequence_length=None, recursive=True):
        """
        Initialize dataset.

        Args:
            root: Directory containing JSON replay files
            file_extension: File extension to filter (default: '.json')
            max_sequence_length: Optional max length to truncate sequences
            recursive: If True, search subdirectories recursively (default: True)
        """
        self.root = root
        self.max_seq_len = max_sequence_length

        if recursive:
            # Search recursively in all subdirectories
            self.paths = []
            for dirpath, dirnames, filenames in os.walk(root):
                for f in filenames:
                    if f.endswith(file_extension):
                        self.paths.append(os.path.join(dirpath, f))
        else:
            # Only search in root directory
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
        with open(self.paths[idx], 'r') as f:
            obj = json.load(f)
        
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
        
        # Load previous action [T, A]
        prev_action = torch.tensor(
            obj.get('prev_action_seq', [[0.0] * 8 for _ in range(T)]),
            dtype=torch.float32
        )[:T]
        
        # Load legal source mask [T, 1, H, W]
        lsrc = torch.tensor(
            obj.get('legal_source_mask_seq', [[[[True] * W for _ in range(H)]] for _ in range(T)]),
            dtype=torch.bool
        )[:T]
        
        # Load action targets
        count = torch.tensor(
            obj.get('count', [0] * T),
            dtype=torch.long
        )[:T]

        src_y = torch.tensor(
            obj.get('src_y', [0] * T),
            dtype=torch.long
        )[:T]

        src_x = torch.tensor(
            obj.get('src_x', [0] * T),
            dtype=torch.long
        )[:T]

        dir_id = torch.tensor(
            obj.get('dir_id', [0] * T),
            dtype=torch.long
        )[:T]

        ratio_id = torch.tensor(
            obj.get('ratio_id', [0] * T),
            dtype=torch.long
        )[:T]
        
        value = torch.tensor(
            obj.get('value', [0.0] * T),
            dtype=torch.float32
        )[:T]
        
        # Load auxiliary targets [T, 1, H, W]
        nextv = torch.tensor(
            obj.get('next_visibility', [[[[0.0] * W for _ in range(H)]] for _ in range(T)]),
            dtype=torch.float32
        )[:T]  # Already has shape [T, 1, H, W]
        
        return {
            'obs': {
                'grid_seq': grid.unsqueeze(0),              # [1, T, C, H, W]
                'globals_seq': globals_.unsqueeze(0),       # [1, T, G]
                'prev_action_seq': prev_action.unsqueeze(0), # [1, T, A]
                'legal_source_mask_seq': lsrc.unsqueeze(0)  # [1, T, 1, H, W]
            },
            'tgt': {
                'count': count.unsqueeze(0),                # [1, T]
                'src_y': src_y.unsqueeze(0),                # [1, T]
                'src_x': src_x.unsqueeze(0),                # [1, T]
                'dir_id': dir_id.unsqueeze(0),              # [1, T]
                'ratio_id': ratio_id.unsqueeze(0),          # [1, T]
                'value': value.unsqueeze(0),                # [1, T]
                'next_visibility': nextv.unsqueeze(0)       # [1, T, 1, H, W]
            },
            'meta': {
                'H': H,
                'W': W,
                'T': T,
                'file': os.path.basename(self.paths[idx])
            }
        }
