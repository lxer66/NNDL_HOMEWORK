import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 确保 output 目录存在
os.makedirs('output', exist_ok=True)

# 词汇表类
class Vocabulary:
    def __init__(self):
        # 特殊标记
        self.char2index = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<MASK>':3}
        for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
            self.char2index[c] = i + 4
        self.index2char = {v:k for k,v in self.char2index.items()}
        self.vocab_size = len(self.char2index)
    def encode(self, text):
        return [self.char2index[c] for c in text.lower() if c in self.char2index]
    def decode(self, idxs):
        return ''.join([self.index2char.get(i, '?') for i in idxs if i in self.index2char and i>=4])

# 字符级GRU模型
class CharGRU(nn.Module):
    def __init__(self, vocab_size, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.gru = nn.GRU(32, 128, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(128*2 if bidirectional else 128, vocab_size)
    def forward(self, x, h=None):
        x = self.embedding(x)
        out, h = self.gru(x, h)
        out = self.fc(out)
        return out, h

# 统一可视化函数
# prob_matrix: (seq_len, vocab_size)
def plot_rank_heatmap(prob_matrix, filename, input_seq=None):
    """
    prob_matrix: (seq_len, vocab_size)
    input_seq: list of chars or None. 
               If a char is present, it's a known input. 
               If None, it's an inferred step.
    """
    seq_len, vocab_size = prob_matrix.shape
    top5_probs = np.zeros((5, seq_len))
    top5_chars = np.empty((5, seq_len), dtype=object)
    
    x_labels = []
    step_count = 1
    
    for t in range(seq_len):
        # Check if this position is a known input
        is_known_input = input_seq is not None and t < len(input_seq) and input_seq[t] is not None
        
        if is_known_input:
            # Known input: Show the character, prob 1.0
            char = input_seq[t].upper()
            top5_chars[:, t] = char
            top5_probs[:, t] = 1.0 
            x_labels.append(char)
        else:
            # Inferred: Show Top 5 predictions
            idx = np.argsort(prob_matrix[t])[::-1][:5]
            top5_probs[:, t] = prob_matrix[t][idx]
            # Map indices to chars
            chars = []
            for i in idx:
                if i >= 4:
                    chars.append(chr(i + 93)) # 4->a (97)
                else:
                    chars.append(['<PAD>','<SOS>','<EOS>','<MASK>'][i])
            top5_chars[:, t] = chars
            x_labels.append(f'Step {step_count}')
            step_count += 1

    plt.figure(figsize=(max(10, 1+seq_len*0.8), 5))
    ax = sns.heatmap(top5_probs, annot=top5_chars, fmt='', cmap='YlGnBu', cbar=True,
                     xticklabels=x_labels,
                     yticklabels=[f'Rank {i+1}' for i in range(5)])
    plt.title('Top-5 Char Probabilities')
    plt.xlabel('Sequence')
    plt.ylabel('Rank')
    plt.tight_layout()
    plt.savefig(f'output/{filename}.png', dpi=200)
    plt.close()
