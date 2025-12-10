import torch
import torch.nn.functional as F
from common import Vocabulary, CharGRU, plot_rank_heatmap
import os
import numpy as np

def interactive():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = Vocabulary()
    model = CharGRU(vocab.vocab_size, bidirectional=True).to(device)
    
    if not os.path.exists('output/task2.pth'):
        print("Error: output/task2.pth not found. Please run train2.py first.")
        return
        
    model.load_state_dict(torch.load('output/task2.pth', map_location=device))
    model.eval()
    print("Task 2 Model Loaded.")
    
    # 设置采样温度
    temperature = 1.0

    while True:
        s = input(f'Enter masked name (use _ for mask, or quit) [Temp={temperature}]: ').strip().lower()
        if s=='quit': break
        
        # Prepare input sequence for visualization
        # Known chars are strings, masks are None
        input_seq_viz = []
        for c in s:
            if c == '_':
                input_seq_viz.append(None)
            elif c in vocab.char2index:
                input_seq_viz.append(c)
            else:
                pass
        
        # Encode for model
        idxs = [vocab.char2index[c] if c!='_' else vocab.char2index['<MASK>'] for c in s if c=='_' or c in vocab.char2index]
        if not idxs: continue
        idxs = idxs[:16]
        input_seq_viz = input_seq_viz[:16]
        
        # Iterative filling
        filled = idxs.copy()
        seq_len = len(filled)
        vocab_size = vocab.vocab_size
        # Matrix to store probabilities used for filling
        fill_probs = np.zeros((seq_len, vocab_size))

        with torch.no_grad():
            for _ in range(16):
                if vocab.char2index['<MASK>'] not in filled: break
                x = torch.tensor([filled], dtype=torch.long).to(device)
                out, _ = model(x)
                
                # 应用温度
                logits = out[0] / temperature
                probs = F.softmax(logits, dim=-1)
                probs_np = probs.cpu().numpy()
                
                # Find best fill position (still based on max confidence to be robust)
                mask_pos = [i for i,v in enumerate(filled) if v==vocab.char2index['<MASK>']]
                if not mask_pos: break
                
                # 策略：找到置信度最高的位置，但对该位置的字符进行采样
                # 这里我们用 max probability 来决定填哪个坑，这步通常不需要随机性，
                # 因为我们想先填最有把握的坑。
                best_pos = max(mask_pos, key=lambda i: probs_np[i,4:].max())
                
                # 对选定位置的字符进行采样
                # 只在有效字符范围内采样 (index >= 4)
                valid_probs = probs[best_pos, 4:]
                # 重新归一化以便 multinomial 使用
                valid_probs = valid_probs / valid_probs.sum()
                
                char_offset = torch.multinomial(valid_probs, 1).item()
                best_idx = char_offset + 4
                
                # Record probabilities for the chosen position
                fill_probs[best_pos] = probs_np[best_pos]
                
                filled[best_pos] = best_idx
        
        name = vocab.decode(filled)
        print('Completed:', name.capitalize())
        
        # Final pass to get probabilities for known chars
        with torch.no_grad():
            x = torch.tensor([filled], dtype=torch.long).to(device)
            out, _ = model(x)
            final_probs = F.softmax(out[0], dim=-1).cpu().numpy()
            
        # Merge: use fill_probs for inferred positions, final_probs for known positions
        combined_probs = fill_probs.copy()
        for i in range(seq_len):
            # If it was a known input (not None in viz), use final_probs
            if i < len(input_seq_viz) and input_seq_viz[i] is not None:
                combined_probs[i] = final_probs[i]
            # If fill_probs is still zero (safety fallback), use final_probs
            elif np.sum(combined_probs[i]) == 0:
                 combined_probs[i] = final_probs[i]

        # We only want to visualize the length of the name
        combined_probs = combined_probs[:seq_len]
        
        plot_rank_heatmap(combined_probs, f'task2_{name}', input_seq=input_seq_viz)
        print(f'Figure saved: output/task2_{name}.png')

if __name__=='__main__':
    interactive()
