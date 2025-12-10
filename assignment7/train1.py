import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from common import Vocabulary, CharGRU
import os

# 生成任务数据集
class GenerationDataset(Dataset):
    def __init__(self, names_file, vocab, maxlen=16):
        with open(names_file, encoding='utf-8') as f:
            self.names = [line.strip().lower() for line in f if line.strip()]
        self.vocab = vocab
        self.maxlen = maxlen
    def __len__(self):
        return len(self.names)
    def __getitem__(self, idx):
        name = self.names[idx]
        x = [self.vocab.char2index['<SOS>']] + self.vocab.encode(name)
        y = self.vocab.encode(name) + [self.vocab.char2index['<EOS>']]
        x = x[:self.maxlen] + [0]*(self.maxlen-len(x))
        y = y[:self.maxlen] + [0]*(self.maxlen-len(y))
        return torch.tensor(x), torch.tensor(y)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = Vocabulary()
    ds = GenerationDataset('data/names.txt', vocab)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    model = CharGRU(vocab.vocab_size, bidirectional=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
    print("Start Training Task 1...")
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
    
    torch.save(model.state_dict(), 'output/task1.pth')
    print("Task 1 Model Saved to output/task1.pth")

if __name__=='__main__':
    train()
