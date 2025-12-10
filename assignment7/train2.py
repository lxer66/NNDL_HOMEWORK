import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from common import Vocabulary, CharGRU
import os
import random

# 补全任务数据集
class MaskedDataset(Dataset):
    def __init__(self, names_file, vocab, maxlen=16):
        with open(names_file, encoding='utf-8') as f:
            self.names = [line.strip().lower() for line in f if line.strip()]
        self.vocab = vocab
        self.maxlen = maxlen
    def __len__(self):
        return len(self.names)
    def __getitem__(self, idx):
        name = self.names[idx]
        x = self.vocab.encode(name)
        # 随机掩码1~N个字符
        n_mask = max(1, int(len(x)*0.3))
        mask_pos = random.sample(range(len(x)), n_mask)
        x_masked = x.copy()
        for p in mask_pos:
            x_masked[p] = self.vocab.char2index['<MASK>']
        x_masked = x_masked[:self.maxlen] + [0]*(self.maxlen-len(x_masked))
        y = x[:self.maxlen] + [0]*(self.maxlen-len(x))
        return torch.tensor(x_masked), torch.tensor(y)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = Vocabulary()
    ds = MaskedDataset('data/names.txt', vocab)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    model = CharGRU(vocab.vocab_size, bidirectional=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
    print("Start Training Task 2...")
    for epoch in range(100):
        model.train()
        total = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            loss = loss_fn(out.view(-1, vocab.vocab_size), y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if (epoch+1)%10==0: print(f'Epoch {epoch+1}, Loss: {total/len(dl):.4f}')
    
    torch.save(model.state_dict(), 'output/task2.pth')
    print("Task 2 Model Saved to output/task2.pth")

if __name__=='__main__':
    train()
