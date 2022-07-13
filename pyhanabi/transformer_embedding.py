# Parts of implementation based off https://github.com/SamLynnEvans/Transformer

import random
import torch
import numpy as np
from typing import Optional
import math
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:,:seq_len]#.cuda()
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(
        self, q : torch.Tensor, 
        k : torch.Tensor, 
        v : torch.Tensor, 
        mask : Optional[torch.Tensor]=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            scores = self.dropout(scores)

        scores = torch.matmul(scores, v)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

import copy

# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.d_model = d_model
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, 
    x : torch.Tensor, 
    mask : Optional[torch.Tensor]=None):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)#.cuda()
    def forward(self, 
    x : torch.Tensor, 
    e_outputs : torch.Tensor, 
    src_mask : Optional[torch.Tensor]=None, 
    trg_mask : Optional[torch.Tensor]=None):
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
            src_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ff(x2))
            return x

# We can then build a convenient cloning function that can generate multiple layers:
#def get_clones(module, N):
#    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout = 0.1):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.embed = Embedder(vocab_size, d_model//2)
        self.features_to_latent_turns = nn.Linear(15*d_model//2, d_model)
        self.fully_conn_on_raw = nn.Linear(838, d_model)
        # self.latent_turns_to_turns = nn.Linear(4*d_model, d_model)

        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(N)])
        self.norm = Norm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, 
    src : torch.Tensor, 
    mask : Optional[torch.Tensor]=None):
        x_state_feat = self.embed(src)
        x = self.dropout(self.features_to_latent_turns(x_state_feat.reshape(src.size(0), src.size(1), 15*self.d_model//2)))

        # x = self.dropout(self.fully_conn_on_raw(src))

        x = self.norm(x)
        x = self.pe(x)

        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(N)])
        self.norm = Norm(d_model)
    def forward(self, 
    trg : torch.Tensor, 
    e_outputs : torch.Tensor, 
    src_mask : Optional[torch.Tensor]=None, 
    trg_mask : Optional[torch.Tensor]=None):
        x = self.embed(trg)
        x = self.pe(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)
        self.use = False

    def get_samples(self, 
    obs : torch.Tensor, 
    own_hand : torch.Tensor,
    seq_len : torch.Tensor,
    device : str):

        batch_size = obs.size(1)

        target_shape = list(own_hand.shape[:-1]) + [5,25]

        target = own_hand.view(target_shape)  # have to add empty card dimension to target!
        target_empty_mask = (target.sum(-1, keepdim=True) == 0.0)
        target = torch.cat([target, target_empty_mask.float()], -1)
        # for j in range(len(batch.seq_len)):
        #     target[batch.seq_len[j].long().item():, j, 0, :] *= 0
        #     target[batch.seq_len[j].long().item():, j, 1, :] *= 0
        # target = target.view(80, 2*batchsize, 5, -1)

        trgs = torch.zeros((own_hand.size(0), batch_size, 2, 7))

        # 0-27 is active agent's cards (25 is empty card, 26 is start of seq token, 27 is end of seq token)
        trgs[:,:,:,0] = 26
        trgs[:,:,:,6] = 27
        trgs[:,:,:,1:6] = target.argmax(dim=-1).reshape(target_shape[:-1])
        # for ex in range(batch_size):
        #     for card in range(1,6):
        #         if 1 in target[gamesteps[ex], ex, card-1, :]:
        #             trgs[ex,card] = int((target[gamesteps[ex], ex, card-1, :] == 1).nonzero(as_tuple=True)[0])
        #         else:
        #             trgs[ex,card] = 25

        # 0-25 is partner's cards
                #   0-25 is first card
                #   26-51 is second card
                #   52-77 is third card
                #   78-103 is fourth card
                #   104-129 is fifth card
        
        partner_cards = obs[:,:,:,125:250].reshape(obs.size(0),batch_size,2,5,25)
        partner_cards_empty_mask = (partner_cards.sum(-1, keepdim=True) == 0.0)
        partner_cards = torch.cat([partner_cards, partner_cards_empty_mask.float()], -1)
        partner_cards = partner_cards.argmax(dim=-1)

        # partner_cards[:,:,1] += 26
        # partner_cards[:,:,2] += 52
        # partner_cards[:,:,3] += 78
        # partner_cards[:,:,4] += 104

        # 26-66 is remaining deck size

        decksizes = 26 + torch.sum(obs[:,:,:,252:292], -1, dtype = torch.long)

        # 67+5*c-72+5*c is fireworks of colour c

        fireworks = obs[:,:,:,292:317].reshape(obs.size(0),batch_size,2,5,5)
        fireworks_empty_mask = (fireworks.sum(-1, keepdim=True) == 0.0)
        fireworks = torch.cat([fireworks, fireworks_empty_mask.float()], -1)
        fireworks = fireworks.argmax(dim = -1)

        for c in range(5):
            fireworks[:,:,:,c] = 67+5*c+fireworks[:,:,:,c]

        # 93-101 is info tokens

        info_tokens = 93 + torch.sum(obs[:,:,:,317:325], -1, dtype = torch.long)

        # 102-105 is life tokens

        life_tokens = 102 + torch.sum(obs[:,:,:,325:328], -1, dtype = torch.long)

        if torch.sum(obs[1:,:,:,378:431]).item() == 0:
            move_type = torch.ones(obs.size(0), obs.size(1), obs.size(2), dtype = torch.long, device=device) * 203
            move_affect = torch.ones(obs.size(0), obs.size(1), obs.size(2), dtype = torch.long, device=device) * 204

        else:
            move_type = obs[1:,:,:,380:384]
            move_type_empty_mask = (move_type.sum(-1, keepdim=True) == 0.0)
            move_type = torch.cat([move_type, move_type_empty_mask.float()], -1)
            move_type = move_type.argmax(dim = -1)
            move_type = 5*move_type + 106

            which_colour = obs[1:,:,:,386:391]
            which_rank = obs[1:,:,:,391:396]
            which_play_disc = obs[1:,:,:,401:406]

            which_colour_empty_mask = (which_colour.sum(-1, keepdim=True) == 0.0)
            which_colour = torch.cat([which_colour_empty_mask.float(), which_colour], -1)

            which_rank_empty_mask = (which_rank.sum(-1, keepdim=True) == 0.0)
            which_rank = torch.cat([which_rank_empty_mask.float(), which_rank], -1)

            which_play_disc_empty_mask = (which_play_disc.sum(-1, keepdim=True) == 0.0)
            which_play_disc = torch.cat([which_play_disc_empty_mask.float(), which_play_disc], -1)

            which_colour = which_colour.argmax(dim = -1)
            which_rank = which_rank.argmax(dim = -1)
            which_play_disc = which_play_disc.argmax(dim = -1)

            move_type += (which_colour + which_rank + which_play_disc - 1)

            which_player = obs[1:,:,:,378:380]
            which_player_empty_mask = (which_player.sum(-1, keepdim=True) == 0.0)
            which_player = torch.cat([which_player, which_player_empty_mask.float()], -1)
            which_player = which_player.argmax(dim = -1)

            move_type += (20*which_player)

            move_affect = obs[1:,:,:,406:431]
            move_affect_empty_mask = (move_affect.sum(-1, keepdim=True) == 0.0)
            move_affect = torch.cat([move_affect, move_affect_empty_mask.float()], -1)
            move_affect = move_affect.argmax(dim = -1)

            move_affect += 146

            move_affect += (obs[1:,:,:,396:401].matmul(2**torch.arange(5, dtype=torch.float, device=device).flip(0).view(5,1))).reshape(-1, batch_size, 2).to(torch.long)

            move_type = torch.cat([torch.tensor([203 for _ in range(batch_size*2)], device=device, dtype=torch.long).reshape(1,batch_size,2), move_type], 0)
            move_affect = torch.cat([torch.tensor([204 for _ in range(batch_size*2)], device=device, dtype=torch.long).reshape(1,batch_size,2), move_affect.to(torch.long)], 0)

        stacked = torch.stack([partner_cards[:,:,:,0], partner_cards[:,:,:,1], partner_cards[:,:,:,2], 
                                partner_cards[:,:,:,3], partner_cards[:,:,:,4], decksizes, fireworks[:,:,:,0],
                                fireworks[:,:,:,1], fireworks[:,:,:,2], fireworks[:,:,:,3], fireworks[:,:,:,4], 
                                info_tokens, life_tokens, move_type, move_affect] , dim=-1)

        stacked = stacked.transpose(0, 1)
        trgs = trgs.transpose(0, 1)

        # ck_tokens = ck_tokens.transpose(0,1).reshape(batch_size, -1)

        # pad with end_of_seq token 5606

        # interleaved = torch.cat([interleaved, 5606+torch.zeros((batch_size, 300), device=device, dtype=torch.long)], 1)

        for j in range(batch_size):
            stacked[j, seq_len[j]:, :, :] = 205
            # interleaved[j, (gamesteps[j]+1)*15:(gamesteps[j]+1)*15+25] = ck_tokens[j, gamesteps[j]*25:(gamesteps[j]+1)*25]
            # interleaved[j, (gamesteps[j]+1)*15+25:(gamesteps[j]+1)*15+30] = 5605
            # ck_tokens[j, (gamesteps[j]+1)*25:] = 5606

        return stacked.detach(), trgs.to(device).detach() # dims are bs x seq_len x 2 x 15, bs x seq_len x 2 x 7
    
    def get_samples_one_player(self, 
    obs : torch.Tensor, 
    own_hand : torch.Tensor,
    seq_len : torch.Tensor,
    device : str):

        batch_size = obs.size(1)

        assert(not torch.any(torch.round(torch.sum(obs[seq_len-1, torch.arange(start=0,end=batch_size,dtype=torch.long), :], -1)) == 0))
        if (torch.all(seq_len < obs.size(0))):
            assert(torch.any(torch.round(torch.sum(obs[seq_len, torch.arange(start=0,end=batch_size,dtype=torch.long), :], -1)) == 0))

        #target_shape = list(own_hand.shape[:-1]) + [5, 25]

        #target = own_hand.view(target_shape)  # have to add empty card dimension to target!
        #target_empty_mask = (target.sum(-1, keepdim=True) == 0.0)
        #target = torch.cat([target, target_empty_mask.float()], -1)
        # for j in range(len(batch.seq_len)):
        #     target[batch.seq_len[j].long().item():, j, 0, :] *= 0
        #     target[batch.seq_len[j].long().item():, j, 1, :] *= 0
        # target = target.view(80, 2*batchsize, 5, -1)

        #trgs = torch.zeros((own_hand.size(0), batch_size, 7))

        # 0-27 is active agent's cards (25 is empty card, 26 is start of seq token, 27 is end of seq token)
        #trgs[:,:,0] = 26
        #trgs[:,:,6] = 27
        #trgs[:,:,1:6] = target.argmax(dim=-1).reshape(target_shape[:-1])
        # for ex in range(batch_size):
        #     for card in range(1,6):
        #         if 1 in target[gamesteps[ex], ex, card-1, :]:
        #             trgs[ex,card] = int((target[gamesteps[ex], ex, card-1, :] == 1).nonzero(as_tuple=True)[0])
        #         else:
        #             trgs[ex,card] = 25

        # 0-25 is partner's cards
                #   0-25 is first card
                #   26-51 is second card
                #   52-77 is third card
                #   78-103 is fourth card
                #   104-129 is fifth card
        
        partner_cards = obs[:,:,125:250].reshape(obs.size(0),batch_size,5,25)
        partner_cards_empty_mask = (partner_cards.sum(-1, keepdim=True) == 0.0)
        partner_cards = torch.cat([partner_cards, partner_cards_empty_mask.float()], -1)
        partner_cards = partner_cards.argmax(dim=-1)

        # partner_cards[:,:,1] += 26
        # partner_cards[:,:,2] += 52
        # partner_cards[:,:,3] += 78
        # partner_cards[:,:,4] += 104

        # 26-66 is remaining deck size

        decksizes = 26 + torch.sum(obs[:,:,252:292], -1, dtype = torch.long)

        # 67+5*c-72+5*c is fireworks of colour c

        fireworks = obs[:,:,292:317].reshape(obs.size(0),batch_size,5,5)
        fireworks_empty_mask = (fireworks.sum(-1, keepdim=True) == 0.0)
        fireworks = torch.cat([fireworks, fireworks_empty_mask.float()], -1)
        fireworks = fireworks.argmax(dim = -1)

        for c in range(5):
            fireworks[:,:,c] = 67+5*c+fireworks[:,:,c]

        # 93-101 is info tokens

        info_tokens = 93 + torch.sum(obs[:,:,317:325], -1, dtype = torch.long)

        # 102-105 is life tokens

        life_tokens = 102 + torch.sum(obs[:,:,325:328], -1, dtype = torch.long)

        if torch.sum(obs[1:,:,378:431]).item() == 0:
            move_type = torch.ones(obs.size(0), obs.size(1), dtype = torch.long, device=device) * 203
            move_affect = torch.ones(obs.size(0), obs.size(1), dtype = torch.long, device=device) * 204

        else:
            move_type = obs[1:,:,380:384]
            move_type_empty_mask = (move_type.sum(-1, keepdim=True) == 0.0)
            move_type = torch.cat([move_type, move_type_empty_mask.float()], -1)
            move_type = move_type.argmax(dim = -1)
            move_type = 5*move_type + 106

            which_colour = obs[1:,:,386:391]
            which_rank = obs[1:,:,391:396]
            which_play_disc = obs[1:,:,401:406]

            which_colour_empty_mask = (which_colour.sum(-1, keepdim=True) == 0.0)
            which_colour = torch.cat([which_colour_empty_mask.float(), which_colour], -1)

            which_rank_empty_mask = (which_rank.sum(-1, keepdim=True) == 0.0)
            which_rank = torch.cat([which_rank_empty_mask.float(), which_rank], -1)

            which_play_disc_empty_mask = (which_play_disc.sum(-1, keepdim=True) == 0.0)
            which_play_disc = torch.cat([which_play_disc_empty_mask.float(), which_play_disc], -1)

            which_colour = which_colour.argmax(dim = -1)
            which_rank = which_rank.argmax(dim = -1)
            which_play_disc = which_play_disc.argmax(dim = -1)

            move_type += (which_colour + which_rank + which_play_disc - 1)

            which_player = obs[1:,:,378:380]
            which_player_empty_mask = (which_player.sum(-1, keepdim=True) == 0.0)
            which_player = torch.cat([which_player, which_player_empty_mask.float()], -1)
            which_player = which_player.argmax(dim = -1)

            move_type += (20*which_player)

            move_affect = obs[1:,:,406:431]
            move_affect_empty_mask = (move_affect.sum(-1, keepdim=True) == 0.0)
            move_affect = torch.cat([move_affect, move_affect_empty_mask.float()], -1)
            move_affect = move_affect.argmax(dim = -1)

            move_affect += 146

            move_affect += (obs[1:,:,396:401].matmul(2**torch.arange(5, dtype=torch.float, device=device).flip(0).view(5,1))).reshape(-1, batch_size).to(torch.long)

            move_type = torch.cat([torch.tensor([203 for _ in range(batch_size)], device=device, dtype=torch.long).reshape(1,batch_size), move_type], 0)
            move_affect = torch.cat([torch.tensor([204 for _ in range(batch_size)], device=device, dtype=torch.long).reshape(1,batch_size), move_affect.to(torch.long)], 0)

        stacked = torch.stack([partner_cards[:,:,0], partner_cards[:,:,1], partner_cards[:,:,2], 
                                partner_cards[:,:,3], partner_cards[:,:,4], decksizes, fireworks[:,:,0],
                                fireworks[:,:,1], fireworks[:,:,2], fireworks[:,:,3], fireworks[:,:,4], 
                                info_tokens, life_tokens, move_type, move_affect] , dim=-1)

        stacked = stacked.transpose(0, 1)
        #trgs = trgs.transpose(0, 1)

        # ck_tokens = ck_tokens.transpose(0,1).reshape(batch_size, -1)

        # pad with end_of_seq token 5606

        # interleaved = torch.cat([interleaved, 5606+torch.zeros((batch_size, 300), device=device, dtype=torch.long)], 1)

        for j in range(batch_size):
            stacked[j, seq_len[j]:, :] = 205
            # interleaved[j, (gamesteps[j]+1)*15:(gamesteps[j]+1)*15+25] = ck_tokens[j, gamesteps[j]*25:(gamesteps[j]+1)*25]
            # interleaved[j, (gamesteps[j]+1)*15+25:(gamesteps[j]+1)*15+30] = 5605
            # ck_tokens[j, (gamesteps[j]+1)*25:] = 5606

        return stacked.detach()#, trgs.to(device).detach() # dims are bs x seq_len x 15, bs x seq_len x 7

    def forward(self, 
    src : torch.Tensor, 
    trg : torch.Tensor, 
    src_mask : Optional[torch.Tensor]=None, 
    trg_mask : Optional[torch.Tensor]=None):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

def get_model(src_vocab, trg_vocab, d_model, N, heads):
    model = Transformer(src_vocab, trg_vocab, d_model, N, heads) 
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) 
    return model
