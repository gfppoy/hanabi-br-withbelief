import torch
import torch.nn.functional as F

def compute_belief(j, src, nopeak_mask, seq_len, priv_s):
    games_considered = [f(i) for i in range(len(seq_len)) for f in (lambda i: 2*i, lambda i: 2*i+1) if seq_len[i] > j]
    if not games_considered: #empty
        return

    temp = torch.zeros(([len(games_considered), src.size(1), 15]), dtype=torch.long, device="cuda:0") # bs x seq_len x 15

    temp[:] = src[games_considered, :, :]
    temp[:, j+1:, :] = 205
    preds = torch.ones((trg[games_considered, j, :-1].shape))
    # preds = belief_module(temp, trg[games_considered, j, :-1].long(), None, nopeak_mask)
    priv_s[j,games_considered,433:563] = F.softmax(preds[:, :-1, 0:26], dim=-1).reshape(preds.size(0), 5 * 26)