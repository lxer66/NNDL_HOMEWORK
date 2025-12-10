import torch
import torch.nn.functional as F
from common import Vocabulary, CharGRU, plot_rank_heatmap
import os
import numpy as np

def interactive():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = Vocabulary()
    model = CharGRU(vocab.vocab_size, bidirectional=False).to(device)
    
    if not os.path.exists('output/task1.pth'):
        print("Error: output/task1.pth not found. Please run train1.py first.")
        return
        
    model.load_state_dict(torch.load('output/task1.pth', map_location=device))
    model.eval()
    print("Task 1 Model Loaded.")
    
    # 设置采样温度，越高越随机 (0.1 ~ 1.0+)
    temperature = 1.0
    
    while True:
        prefix = input(f'Enter prefix (or quit) [Temp={temperature}]: ').strip().lower()
        if prefix=='quit': break
        
        # Encode prefix
        idxs = [vocab.char2index['<SOS>']] + vocab.encode(prefix)
        
        # We need to collect probabilities for the generated part
        generated_probs = []
        
        with torch.no_grad():
            for _ in range(16):
                x = torch.tensor([idxs], dtype=torch.long).to(device)
                out, _ = model(x)
                
                # 获取最后一步的 logits 并应用温度
                logits = out[0, -1] / temperature
                prob = F.softmax(logits, dim=-1)
                
                # 保存用于可视化的原始概率 (不带温度，或者带温度看需求，这里存带温度的反映真实采样分布)
                prob_np = prob.cpu().numpy()
                generated_probs.append(prob_np)
                
                # 采样而不是 argmax
                next_idx = torch.multinomial(prob, 1).item()
                
                if next_idx==vocab.char2index['<EOS>']: break
                if next_idx<4: continue # Skip special tokens if generated
                idxs.append(next_idx)
        
        name = vocab.decode(idxs[1:]) # Skip SOS
        print('Generated:', name.capitalize())
        
        # Prepare data for visualization
        prefix_len = len(prefix)
        gen_len = len(generated_probs)
        vocab_size = vocab.vocab_size
        
        # Full matrix: [prefix_len + gen_len, vocab_size]
        full_prob_matrix = np.zeros((prefix_len + gen_len, vocab_size))
        
        # Fill generated part
        if gen_len > 0:
            full_prob_matrix[prefix_len:] = np.stack(generated_probs, axis=0)
            
        # Construct input_seq
        # Prefix chars are known. Generated chars are None (so they show up as Step X)
        input_seq = list(prefix) + [None] * gen_len
        
        plot_rank_heatmap(full_prob_matrix, f'task1_{name}', input_seq=input_seq)
        print(f'Figure saved: output/task1_{name}.png')

if __name__=='__main__':
    interactive()
