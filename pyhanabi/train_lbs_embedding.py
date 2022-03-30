# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import matplotlib as mlp
mlp.use("Agg")

import time
import os
import sys
import argparse
import pprint

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import torch.nn.functional as F
import random
import math

from create import create_envs, create_threads, ActGroup
from eval import evaluate
import common_utils
import rela
import r2d2_beliefmodule as r2d2
import utils

from transformer_embedding import get_model, Transformer

def get_samples(batch, args):
    # obs = batch.obs["priv_s"].to("cpu")
    # obs = torch.cat([obs, batch.action["a"].to("cpu").unsqueeze(-2).repeat(1,1,args.num_player,1).float()], -1)
    # obs = torch.cat([obs, torch.zeros((80, batchsize, args.num_player, 18))], dim=3)
    # obs = obs[:,:,0,:]
    # # obs = obs.view(80, 2*batchsize, -1)

    batch_size = len(batch.seq_len)

    target = batch.obs["own_hand"].view(*batch.obs["own_hand"].shape[:3], args.hand_size,
                                        25)  # have to add empty card dimension to target!
    target_empty_mask = (target.sum(-1, keepdim=True) == 0.0)
    target = torch.cat([target, target_empty_mask.float()], -1)
    # for j in range(len(batch.seq_len)):
    #     target[batch.seq_len[j].long().item():, j, 0, :] *= 0
    #     target[batch.seq_len[j].long().item():, j, 1, :] *= 0
    # target = target.view(80, 2*batchsize, 5, -1)
    target = target[:,:,0,:,:]

    srcs = torch.zeros((batch_size, 3200), device="cpu")
    trgs = torch.zeros((batch_size, 7), device="cpu")

    gamesteps = np.zeros(len(batch.seq_len))
    for j in range(len(batch.seq_len)):
        gamesteps[j] = random.randint(0, batch.seq_len[j]-1)

    gamesteps = gamesteps.astype(int)

    # 0-27 is active agent's cards (25 is empty card, 26 is start of seq token, 27 is end of seq token)
    trgs[:,0] = 26
    trgs[:,6] = 27
    trgs[:,1:6] = (target[gamesteps, range(64), :] == 1).nonzero(as_tuple=True)[2].reshape(-1,5)
    # for ex in range(batch_size):
    #     for card in range(1,6):
    #         if 1 in target[gamesteps[ex], ex, card-1, :]:
    #             trgs[ex,card] = int((target[gamesteps[ex], ex, card-1, :] == 1).nonzero(as_tuple=True)[0])
    #         else:
    #             trgs[ex,card] = 25

    obs = batch.obs["priv_s"]
    obs = obs[:, :, 0, :]
    start = time.time()

    # 0-129 is partner's cards
    #   0-25 is first card
    #   26-51 is second card
    #   52-77 is third card
    #   78-103 is fourth card
    #   104-129 is fifth card
    
    partner_cards = obs[:,:,125:250].reshape(80,batch_size,5,25)
    partner_cards_empty_mask = (partner_cards.sum(-1, keepdim=True) == 0.0)
    partner_cards = torch.cat([partner_cards, partner_cards_empty_mask.float()], -1)
    partner_cards = (partner_cards == 1).nonzero(as_tuple=True)[3].reshape(80,batch_size,5)

    partner_cards[:,:,1] += 26
    partner_cards[:,:,2] += 52
    partner_cards[:,:,3] += 78
    partner_cards[:,:,4] += 104

    # 130-170 is remaining deck size

    decksizes = 130 + torch.sum(obs[:,:,252:292], -1, dtype = torch.long)

    # 171+5*c-176+5*c is fireworks of colour c

    fireworks = obs[:,:,292:317].reshape(80,batch_size,5,5)
    fireworks_empty_mask = (fireworks.sum(-1, keepdim=True) == 0.0)
    fireworks = torch.cat([fireworks, fireworks_empty_mask.float()], -1)
    fireworks = (fireworks == 1).nonzero(as_tuple=True)[3].reshape(80,batch_size,5)
    for c in range(5):
        fireworks[:,:,c] = 171+5*c+fireworks[:,:,c]

    # 197-205 is info tokens

    info_tokens = 197 + torch.sum(obs[:,:,317:325], -1, dtype = torch.long)

    # 206-209 is life tokens

    life_tokens = 206 + torch.sum(obs[:,:,325:328], -1, dtype = torch.long)

    # 210-249 is last action
        #   210-214 is play
        #   215-219 is discard
        #   220-224 is colour hint
        #   225-229 is rank hint
        #       + 20 for player 2
    # 250-274 is card played/discarded
    # 275-306 is cards affected by hint (2*2*2*2*2 possible hints)
    # 1057,1058 no-op action

    move_type = obs[1:,:,380:384]
    move_type_empty_mask = (move_type.sum(-1, keepdim=True) == 0.0)
    move_type = torch.cat([move_type, move_type_empty_mask.float()], -1)
    move_type = (move_type == 1).nonzero(as_tuple=True)[2].reshape(-1, batch_size)
    move_type = 5*move_type + 210

    which_colour = obs[1:,:,386:391]
    which_rank = obs[1:,:,391:396]
    which_play_disc = obs[1:,:,401:406]
    which_colour_empty_mask = (which_colour.sum(-1, keepdim=True) == 0.0)
    which_colour = torch.cat([which_colour_empty_mask.float(), which_colour], -1)
    which_rank_empty_mask = (which_rank.sum(-1, keepdim=True) == 0.0)
    which_rank = torch.cat([which_rank_empty_mask.float(), which_rank], -1)
    which_play_disc_empty_mask = (which_play_disc.sum(-1, keepdim=True) == 0.0)
    which_play_disc = torch.cat([which_play_disc_empty_mask.float(), which_play_disc], -1)
    which_colour = (which_colour == 1).nonzero(as_tuple=True)[2].reshape(-1, batch_size)
    which_rank = (which_rank == 1).nonzero(as_tuple=True)[2].reshape(-1, batch_size)
    which_play_disc = (which_play_disc == 1).nonzero(as_tuple=True)[2].reshape(-1, batch_size)

    move_type += (which_colour + which_rank + which_play_disc - 1)

    which_player = obs[1:,:,378:380]
    which_player_empty_mask = (which_player.sum(-1, keepdim=True) == 0.0)
    which_player = torch.cat([which_player, which_player_empty_mask.float()], -1)
    which_player = (which_player == 1).nonzero(as_tuple=True)[2].reshape(-1, batch_size)
    move_type += (20*which_player)

    move_affect = obs[1:,:,406:431]
    move_affect_empty_mask = (move_affect.sum(-1, keepdim=True) == 0.0)
    move_affect = torch.cat([move_affect, move_affect_empty_mask.float()], -1)
    move_affect = (move_affect == 1).nonzero(as_tuple=True)[2].reshape(-1, batch_size)
    move_affect += 250

    move_affect += (obs[1:,:,396:401].matmul(2**torch.arange(5, device=device, dtype=torch.float).flip(0).view(5,1))).reshape(-1, batch_size).type(torch.long)

    move_type = torch.cat([torch.tensor([1057 for _ in range(batch_size)], device=device).reshape(1,batch_size), move_type], 0)
    move_affect = torch.cat([torch.tensor([1058 for _ in range(batch_size)], device=device).reshape(1,batch_size), move_affect], 0)

    # 307-681 is common knowledge wrt player 1
    #   307-307+(4+3+3+3+2)-1 is card 0, colour 0
    #   . . .
    #   307+4*(4+3+3+3+2)-307+5*(4+3+3+3+2)-1 is card 0, colour 4
    #   . . .
    #   307+24*(4+3+3+3+2)-307+25*(4+3+3+3+2)-1

    # 682-1056 is common knowledge wrt player 1
        #   682-682+(4+3+3+3+2)-1 is card 0, colour 0
        #   . . .
        #   682+4*(4+3+3+3+2)-682+5*(4+3+3+3+2)-1 is card 0, colour 4
        #   . . .
        #   682+24*(4+3+3+3+2)-682+25*(4+3+3+3+2)-1

    ck = obs[:,:,433:783]
    first_rank_indices = np.array([0,5,10,15,20,35,40,45,50,55,70,75,80,85,90,105,110,115,120,125,140,145,150,155,160,
                                    175,180,185,190,195,210,215,220,225,230,245,250,255,260,265,280,285,290,295,300,315,320,325,330,335])
    second_rank_indices = first_rank_indices + 1
    third_rank_indices = first_rank_indices + 2
    fourth_rank_indices = first_rank_indices + 3
    fifth_rank_indices = first_rank_indices + 4

    first_rank_indices_tokens = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,
                                        125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245])
    second_rank_indices_tokens = first_rank_indices_tokens + 1
    third_rank_indices_tokens = first_rank_indices_tokens + 2
    fourth_rank_indices_tokens = first_rank_indices_tokens + 3
    fifth_rank_indices_tokens = first_rank_indices_tokens + 4

    next_token = torch.tensor([i*15 for i in range(50)], device=device)

    ck_tokens = torch.zeros(80,batch_size,250, device=device)
    ck_tokens[:] = 307
    ck_tokens[:,:,first_rank_indices_tokens] += (ck[:,:,first_rank_indices] + next_token)
    ck_tokens[:,:,second_rank_indices_tokens] += (ck[:,:,second_rank_indices] + next_token + 4)
    ck_tokens[:,:,third_rank_indices_tokens] += (ck[:,:,third_rank_indices] + next_token + 4 + 3)
    ck_tokens[:,:,fourth_rank_indices_tokens] += (ck[:,:,fourth_rank_indices] + next_token + 4 + 3 + 3)
    ck_tokens[:,:,fifth_rank_indices_tokens] += (ck[:,:,fifth_rank_indices] + next_token + 4 + 3 + 3 + 3)

    stacked = torch.stack([partner_cards[:,:,0], partner_cards[:,:,1], partner_cards[:,:,2], 
                            partner_cards[:,:,3], partner_cards[:,:,4], decksizes, fireworks[:,:,0],
                            fireworks[:,:,1], fireworks[:,:,2], fireworks[:,:,3], fireworks[:,:,4], 
                            info_tokens, life_tokens, move_type, move_affect], dim=1)

    interleaved = torch.flatten(stacked, start_dim = 0, end_dim = 1).transpose(0,1)

    ck_tokens = ck_tokens.transpose(0,1).reshape(batch_size, -1)

    # pad with end_of_seq token 1059
    for j in range(batch_size):
        interleaved[j, (gamesteps[j]+1)*15:] = 1059
        ck_tokens[j, (gamesteps[j]+1)*250:] = 1059

    # for ex in range(batch_size):
    #     src = torch.tensor([], dtype = torch.long, device="cpu")
    #     for t in range(gamesteps[ex]+1):

            # 0-25 is partner's cards

            # for card in range(5):
            #     if 1 in obs[t,ex,125+card*25:150+card*25]:
            #         src = torch.cat([src, (obs[t,ex,125+card*25:150+card*25] == 1).nonzero(as_tuple=True)[0]], 0)
            #     else:
            #         src = torch.cat([src, torch.tensor([25])])

            # 26-66 is remaining deck size

            # src = torch.cat([src, torch.tensor([26+int(sum(obs[t,ex,252:292]))])])

            # 67+5*c-72+5*c is fireworks of colour c

            # for c in range(5):
            #     if 1 in obs[t,ex,292+5*c:297+5*c]:
            #         src = torch.cat([src, torch.tensor([67+5*c+int((obs[t,ex,292+5*c:297+5*c] == 1).nonzero(as_tuple=True)[0])])], 0)
            #     else:
            #         src = torch.cat([src, torch.tensor([72+5*c])])

            # 93-101 is info tokens

            # src = torch.cat([src, torch.tensor([93+int(sum(obs[t,ex,317:325]))])])

            # 102-105 is life tokens

            # src = torch.cat([src, torch.tensor([102+int(sum(obs[t,ex,325:328]))])])

            # 106-145 is last action
            #   106-110 is play
            #   111-115 is discard
            #   116-120 is colour hint
            #   121-125 is rank hint
            #       + 20 for player 2
            # 146-170 is card played/discarded
            # 171-202 is cards affected by hint (2*2*2*2*2 possible hints)

            # if t > 0:

            #     move = -1
            #     play_or_discard = False

            #     if 1 in obs[t,ex,380]: # play
            #         move = 106+int((obs[t,ex,401:406] == 1).nonzero(as_tuple=True)[0])
            #         play_or_discard = True
        
            #     elif 1 in obs[t,ex,381]: # discard
            #         move = 111+int((obs[t,ex,401:406] == 1).nonzero(as_tuple=True)[0])
            #         play_or_discard = True

            #     elif 1 in obs[t,ex,382]: # colour hint
            #         move = 116+int((obs[t,ex,386:391] == 1).nonzero(as_tuple=True)[0])

            #     else: # rank hint
            #         move = 121+int((obs[t,ex,391:396] == 1).nonzero(as_tuple=True)[0])

            #     if 1 in obs[t,ex,379]: # player 2
            #         move += 20

            #     src = torch.cat([src, torch.tensor([move])])

            #     if play_or_discard:
            #         src = torch.cat([src, torch.tensor([146+int((obs[t,ex,406:431] == 1).nonzero(as_tuple=True)[0])])])

            #     else: # hint
            #         hints = np.array([0,0,0,0,0])
            #         for j in range(5):
            #             if 1 in obs[t,ex,396+j]:
            #                 hints[j] = 1
            #         src = torch.cat([src, torch.tensor([171+hint_enumeration[(hints[0], hints[1], hints[2], hints[3], hints[4])]])])
            # else: # no-op action
            #     src = torch.cat([src, torch.tensor([5603,5604])])

    # start = time.time()

    # v0 = rainman(obs[0:gamesteps[ex]+1,ex,378:433], obs[0:gamesteps[ex]+1,ex,125:250])

    # print("Rainman:")
    # print(time.time() - start)

    # 203-5602 is v0
    #   203-203+216-1 is colour 0, card 0
    #   ...
    #   203+216*4-203+216*5-1 is colour 4, card 0
    #   ...
    #   203+216*24-203+216*25-1 is colour 4, card 4

    # for card in range(5):
    #     for colour in range(5):
    #         possible_ranks = v0[colour*5:(colour+1)*5,card]
    #         possible_ranks = np.where(possible_ranks<0,0,possible_ranks)
    #         token = v0_enumeration[(possible_ranks[0], possible_ranks[1], possible_ranks[2], possible_ranks[3], possible_ranks[4])]
    #         src = torch.cat([src, torch.tensor([203+216*5*card+216*colour+token])])
    
    # pad v0 with fifteen 5605's to make all things divisible by 40
    # for _ in range(15):
    #     src = torch.cat([src, torch.tensor([5605])])

    # # vocab size = 5607, (end of seq token 5606)
    # srcs[ex,:] = torch.cat([src, torch.tensor([5606 for _ in range(3200-len(src))])], 0)

    return interleaved, ck_tokens, trgs

def belief_run(model,obs,ck,target,total_losses,args,optim,step_num,stopwatch,device,train_or_eval):
    if train_or_eval == 'train':
        model.train()
        total_losses, losses = belief_run_loop(model,obs,ck,target,total_losses,args,optim,stopwatch,device,train_or_eval)
        loss = sum(losses)
        optim.zero_grad()
        loss.backward()
        stopwatch.time("forward & backward")
        # if step_num < args.warm_up_period:
        #     for g in optim.param_groups:
        #         g['lr'] = step_num * args.lr / args.warm_up_period
        # elif step_num == args.warm_up_period:
        #     for g in optim.param_groups:
        #         g['lr'] = args.lr
        # else: # decay by inverse root schedule
        #     for g in optim.param_groups:
        #         # g['lr'] = args.lr * min(1.0/math.sqrt(step_num - args.warm_up_period), (step_num - args.warm_up_period) * (args.warm_up_period ** (-1.5)))
        #         g['lr'] = args.lr * 1.0/math.sqrt(step_num - args.warm_up_period)
        g_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
        )
        optim.step()
        step_num += 1
        stopwatch.time("update model")
        return total_losses, step_num
    else:
        with torch.no_grad():
            model.eval()
            total_losses, _ = belief_run_loop(model,obs,ck,target,total_losses,args,optim,stopwatch,device,train_or_eval)
            return total_losses
    
def belief_run_loop(model,obs,ck,target,total_losses,args,optim,stopwatch,device,train_or_eval):
    # gamesteps = np.zeros(len(batch.seq_len))

    # for j in range(len(batch.seq_len)):
    #     seq_len = batch.seq_len[j]

    #     gamestep = random.randint(0, seq_len-1)
    #     gamesteps[j] = gamestep
    #     obs[gamestep+1:, j, :] *= 0

        # gamestep = random.randint(0, seq_len-1)
        # gamesteps[j+args.batchsize] = gamestep
        # obs[gamestep+1:, j+args.batchsize, :] *= 0

    # gamesteps = gamesteps.astype(int)

    # rand_perm = torch.randperm(2*args.batchsize)
    # obs       = obs[:,rand_perm,:]
    # target    = target[:,rand_perm,:]
    # gamesteps = gamesteps[rand_perm]

    _, trg_mask = create_masks(obs, target[:, :-1])

    losses = []
    target = target.to(device)
    obs = obs.to(device)
    # ck = ck.to(device)
    trg_mask = trg_mask.to(device)

    preds = model(obs, target[:, :-1], None, trg_mask)#.to("cpu")

    for j in range(5):
        loss = F.cross_entropy(preds[:,j,:].view(-1, preds.size(-1)), target[:,j+1].contiguous().view(-1), ignore_index = 5606)
        total_losses[j] += loss.item()
        losses.append(loss)

    # for j in range(5):
    #     trg = target
    #     trg[:, j:] *= 0
    #     print(torch.cuda.memory_allocated())
    #     preds = model(obs, trg, None, None)[:,j,:]#.to("cpu")
    #     loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target[:,j].contiguous().view(-1), ignore_index = 5606)
    #     losses.append(loss)
    #     total_losses[j] += loss.item()

        # trg = torch.zeros(args.batchsize, 5, 26)
        # for k in range(args.batchsize):
        #     trg[k,:,:] = target[gamesteps[k],k,:,:]

    # loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target.contiguous().view(-1), ignore_index = -1)

    # loss = get_loss_online(args.batchsize, preds, trg, device, j)

        # losses.append(loss)
        
        # total_losses[j] += loss.item()

        # preds_so_far[:,j,:] = F.softmax(preds.view(args.batchsize,26), dim=-1)

    return total_losses, losses


def create_masks(input_seq, target_seq):
    # creates mask with 0s wherever there is padding in the input
    input_pad = 5606
    input_msk = (input_seq != input_pad).unsqueeze(1)

    size = target_seq.size(1) # get seq_len for matrix
    nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype(np.uint8)
    nopeak_mask = torch.autograd.Variable(torch.from_numpy(nopeak_mask) == 0)

    return input_msk, nopeak_mask

# def free_memory(model, obs, preds, preds_so_far_):
#     torch.cuda.empty_cache()
#     for j in range(5):
#         model[j] = model[j].to(device)
#     obs = obs.to(device)
#     preds = preds.to(device)
#     preds_so_far_ = preds_so_far_.to(device)

def rainman(last_actions, other_player_hands):
    # last_actions is 2d array, seq_length x 55, where 55 is encoding dim for 2 player last action
    # other_player_hands is 2d array, seq_length x 125, where 125 is the 5 x 25, handsize and card embedding dim respectively
    # card to predict is which card in the hand we are tracking the belief of (0 to 4)
    
    seq_length = len(last_actions)
    
    cards_left = np.array([3,2,2,2,1,3,2,2,2,1,3,2,2,2,1,3,2,2,2,1,3,2,2,2,1]) # colour-major-ordering
    
    # own_hand_tracking is handsize x 2 x 5, where 2 is [colour, rank] and 5 is the colour/rank dim
    # 1 at index denotes the given colour/rank for the index is possible
    own_hand_tracking = np.ones((5, 2, 5))
    
    # flag on cards we exactly know, so as to avoid double subtracting over iterations
    exactly_know = np.zeros(5)
    
    # account for other player's initial hand
    other_player_init_hand = other_player_hands[0]
    for j in range(5):
        card = list(other_player_init_hand[25 * j : 25 * (j+1)]).index(1)
        cards_left[card] -= 1
    
    for i in range(1, seq_length):
        last_action = last_actions[i]
        
        if last_action[0] == 1: # active player's turn
            if last_action[2] == 1 or last_action[3] == 1: # active player plays or discards
                which_card_in_hand = list(last_action[23:28]).index(1)
                if exactly_know[which_card_in_hand] == 0: 
                    # card is not yet accounted for in cards_left, so we thus account for it
                    which_card = list(last_action[28:53]).index(1)
                    cards_left[which_card] -= 1
                # reset what we know about the card slot
                own_hand_tracking[which_card_in_hand, :, :] = np.ones((2,5))
                exactly_know[which_card_in_hand] = 0
                # shift cards
                own_hand_tracking[which_card_in_hand:, :, :] = np.roll(own_hand_tracking[which_card_in_hand:, :, :], -1, 0)
                exactly_know[which_card_in_hand:] = np.roll(exactly_know[which_card_in_hand:], -1, 0)
                
        else: # other player's turn
            if last_action[2] == 1 or last_action[3] == 1: # other player plays or discards      
                # account for other player's new card
                try:
                    new_card = list(other_player_hands[i, 100:125]).index(1)
                    cards_left[new_card] -= 1
                except:
                    try:
                        new_card = list(other_player_hands[i, 75:100]).index(1)
                        cards_left[new_card] -= 1
                    except:
                        pass
                            
            elif last_action[4] == 1: # other player gives colour hint
                which_colour = list(last_action[8:13]).index(1)
                for k in range(5):
                    if last_action[k + 18] == 1: # set all colours besides which_colour to 0 for kth card
                        indices = list([0,1,2,3,4])
                        del indices[which_colour]
                        own_hand_tracking[k, 0, indices] = 0
                    else: # set which_colour to 0 for kth card
                        own_hand_tracking[k, 0, which_colour] = 0
                    if exactly_know[k] == 0 and list(own_hand_tracking[k, 0, :]).count(1) == 1 and list(own_hand_tracking[k, 1, :]).count(1) == 1:
                        # exactly know kth card and haven't accounted for it yet, so we thus account for it
                        exactly_know[k] = 1
                        cards_left[list(own_hand_tracking[k, 0, :]).index(1) * 5 
                                   + list(own_hand_tracking[k, 1, :]).index(1)] -= 1
                        
            else: # other player gives rank hint
                which_rank = list(last_action[13:18]).index(1)
                for k in range(5):
                    if last_action[k + 18] == 1: # set all ranks besides which_rank to 0 for kth card
                        indices = list([0,1,2,3,4])
                        del indices[which_rank]
                        own_hand_tracking[k, 1, indices] = 0
                    else: # set which_rank to 0 for kth card
                        own_hand_tracking[k, 1, which_rank] = 0
                    if exactly_know[k] == 0 and list(own_hand_tracking[k, 0, :]).count(1) == 1 and list(own_hand_tracking[k, 1, :]).count(1) == 1:
                        # exactly know kth card and haven't accounted for it yet, so we now account for it
                        exactly_know[k] = 1
                        cards_left[list(own_hand_tracking[k, 0, :]).index(1) * 5 
                                   + list(own_hand_tracking[k, 1, :]).index(1)] -= 1 
                        
    # having processed history, narrow down the cards:

    rainman_prob = np.zeros((25,5))
    for j in range(5):
        rainman_prob[:,j] = cards_left

    for card_to_predict in range(5):
        for j in range(5): # narrow down the colours
            if own_hand_tracking[card_to_predict, 0, j] == 0:
                for k in range(5): # set all ranks k of colour j to 0 in cards_left
                    rainman_prob[j * 5 + k, card_to_predict] = 0
    
    for card_to_predict in range(5):
        for j in range(5): # narrow down the ranks
            if own_hand_tracking[card_to_predict, 1, j] == 0:
                for k in range(5): # set all colours k of rank j to 0 in cards_left
                    rainman_prob[k * 5 + j, card_to_predict] = 0
                
    return rainman_prob

# enumeration of 1 colour (size 216 (=4*3*3*3*2) dictionary)
def enumerate_v0():
    enumeration = {}
    count = 0
    for a in range(4):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    for e in range(2):
                        enumeration[(a,b,c,d,e)] = count
                        count += 1
    return enumeration

# enumeration of hints that got coloured
def enumerate_hints():
    enumeration = {}
    count = 0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    for e in range(2):
                        enumeration[(a,b,c,d,e)] = count
                        count += 1
    return enumeration
        

def load_op_model(method, idx1, idx2, device):
    """load op models, op models was trained only for 2 player
    """
    #root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # assume model saved in root/models/op
    #folder = os.path.join(root, "models", "op", method)
    folder = os.path.join("models", "op", "sad")
    agents = []
    for idx in [idx1, idx2]:
        if idx >= 0 and idx < 3:
            num_fc = 1
            skip_connect = False
        elif idx >= 3 and idx < 6:
            num_fc = 1
            skip_connect = True
        elif idx >= 6 and idx < 9:
            num_fc = 2
            skip_connect = False
        else:
            num_fc = 2
            skip_connect = True
        weight_file = os.path.join(folder, f"M{idx}.pthw")
        if not os.path.exists(weight_file):
            print(f"Cannot find weight at: {weight_file}")
            assert False

        state_dict = torch.load(weight_file)
        input_dim = state_dict["net.0.weight"].size()[1]
        hid_dim = 512
        output_dim = state_dict["fc_a.weight"].size()[0]
        agent = r2d2.R2D2Agent(
            True, # False easier to use as VDN! if using False set method=iql!
            3,
            0.999,
            0.9,
            device,
            input_dim,
            hid_dim,
            output_dim,
            2,
            5,
            False,
            num_fc_layer=num_fc,
            skip_connect=skip_connect,
        ).to(device)
        utils.load_weight(agent.online_net, weight_file, device)
        agents.append(agent)
    return agents



def parse_args():
    parser = argparse.ArgumentParser(description="train belief model")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--method", type=str, default="vdn")
    parser.add_argument("--shuffle_obs", type=int, default=0)
    parser.add_argument("--shuffle_color", type=int, default=0)
    parser.add_argument("--pred_weight", type=float, default=0)
    parser.add_argument("--num_eps", type=int, default=80)

    parser.add_argument("--load_model", type=str, default="")

    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--eta", type=float, default=0.9, help="eta for aggregate priority")
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--sad", type=int, default=1)
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--hand_size", type=int, default=5)
    parser.add_argument("--encoder_dim", type=int, default=858)
    parser.add_argument("--decoder_dim", type=int, default=26)
    parser.add_argument("--num_heads", type=int, default=4) #must have num_head | encoder_dim, decoder_dim
    parser.add_argument("--N", type=int, default=4)

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=0.00008, help="Learning rate")
    parser.add_argument("--warm_up_period", type=float, default=100000, help="Warm Up Period")
    parser.add_argument("--eps", type=float, default=1e-9, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")
    parser.add_argument("--num_lstm_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)

    parser.add_argument("--train_device", type=str, default="cuda:2")
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--num_update_between_sync", type=int, default=3500)
    parser.add_argument("--eval_epochs", type=int, default=1)

    # DQN settings
    parser.add_argument("--multi_step", type=int, default=3)

    # replay buffer settingsœ
    parser.add_argument("--burn_in_frames", type=int, default=2500) #2500
    parser.add_argument("--replay_buffer_size", type=int, default=5000) #5000
    parser.add_argument(
        "--priority_exponent", type=float, default=0.6, help="prioritized replay alpha",
    )
    parser.add_argument(
        "--priority_weight", type=float, default=0.4, help="prioritized replay beta",
    )
    #MOD 3: flag that switches off replay buffer priority by default
    parser.add_argument(
        "--no_replay_buffer_priority", type=bool, default=True, help="switch replay buffer priority off"
    )
    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=15, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=20)

    # actor setting
    parser.add_argument("--act_base_eps", type=float, default=0.4)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--act_device", type=str, default="cuda:1")
    parser.add_argument("--actor_sync_freq", type=int, default=10)

    # special modes
    parser.add_argument("--obs_n_most_recent_own_cards", type=int, default=0)
    parser.add_argument("--use_softmax_policy", type=int, default=0)
    parser.add_argument("--log_beta_range", type=str, default="0.5,10")
    parser.add_argument("--eval_log_betas", type=str, default="1,2,3,5,7,10")
    parser.add_argument("--q_variant", type=str, default="doubleq")
    # Feature 26: empty card
    parser.add_argument("--card-feature-dim", type=int, default=26, help="dimensionality of a single card feature")
    parser.add_argument("--use_autoregressive_belief", type=int, help="if True use autoregressive belief")
    parser.add_argument("--belief_type", type=str, default="own_hand", help="if True use autoregressive belief") # can be own_hand or public
    parser.add_argument("--model_is_op", type=bool, default=False,
                        help="whether OP model is used")  # can be own_hand or public
    parser.add_argument("--idx", default=1, type=int, help="which model to use (for OP)?")

    args = parser.parse_args()
    assert args.method in ["vdn", "iql"]
    # assert args.load_model != "", "You need to load a model in train LBS mode!"
    return args

if __name__ == "__main__":
    print("Waiting for reflection phase...")
    start_time = time.time()
    while torch.load("reflection_phase.pt").item() == 0:
        time.sleep(10)
        if ((time.time()-start_time)//60 > 0 and (time.time()-start_time)//60 % 5 == 0 and (time.time()-start_time)%60 < 15):
            print("Waiting for reflection phase...")
    print("Commence reflection phase.")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    args.load_model = "current_belief.pth"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(args.save_dir, 5)

    common_utils.set_all_seeds(args.seed)
    # pprint.pprint(vars(args))

    if args.method == "vdn":
        args.batchsize = int(np.round(args.batchsize / args.num_player))
        args.replay_buffer_size //= args.num_player
        args.burn_in_frames //= args.num_player

    # explore_eps = utils.generate_explore_eps(
    #     args.act_base_eps, args.act_eps_alpha, args.num_eps
    # )
    # expected_eps = np.mean(explore_eps)
    # print("explore eps:", explore_eps)
    # print("avg explore eps:", np.mean(explore_eps))

    # if not not args.model_is_op:

    # COMMENTED FROM HERE FOR SINGLE REPLAY v v v

    # games = create_envs(
    #     args.num_thread * args.num_game_per_thread,
    #     args.seed,
    #     args.num_player,
    #     args.hand_size,
    #     args.train_bomb,
    #     explore_eps,
    #     args.max_len,
    #     args.sad,
    #     args.shuffle_obs,
    #     args.shuffle_color,
    #     #args.obs_n_most_recent_own_cards # modification by CHRISTIAN
    # )

    # # full obs modification (CHRISTIAN)
    # feature_size = games[0].feature_size()
    # if args.belief_type == "public":
    #     # remove all card observations from input
    #     feature_size = feature_size - (args.hand_size*args.num_player*args.card_feature_dim + 5)
    # print("FEATURE SIZE: ", feature_size)

    # # MOD 5: replace weight loading with agent loading!
    # if args.load_model != "" and not args.model_is_op:
    #     print("*****loading pretrained model*****")
    #     # utils.load_weight(agent.online_net, args.load_model, args.train_device)
    #     overwrite = {}
    #     overwrite["vdn"] = (args.method == "vdn")
    #     overwrite["device"] = "cuda:0"
    #     overwrite["boltzmann_act"] = False
    #     agent, cfg = utils.load_agent(
    #         args.load_model,
    #         overwrite,
    #     )
    #     agent.log_trajectories = True
    #     print("CFG: ", cfg)
    #     assert cfg["num_player"] == args.num_player, "Model num players does not coincide with config num players!"
    #     print("*****done*****")
    # # elif args.load_model != "" and args.model_is_op:
    # # agent = load_op_model(args.method, args.idx, args.idx, args.train_device)[0]

    # # else:
    # # agent = r2d2.R2D2Agent(
    # #     (args.method == "vdn"),
    # #     args.multi_step,
    # #     args.gamma,
    # #     args.eta,
    # #     args.train_device,
    # #     feature_size, # if not args.use_softmax_policy else feature_size + 1,
    # #     args.rnn_hid_dim,
    # #     games[0].num_action(),
    # #     args.num_lstm_layer,
    # #     args.hand_size,
    # #     args.no_replay_buffer_priority  # uniform priority
    # #     # args.use_softmax_policy,
    # #     # args.q_variant
    # # )

    # agent.sync_target_with_online()

    # if args.load_model and not args.model_is_op:
    #     print("*****loading pretrained model*****")
    #     # utils.load_weight(agent.online_net, args.load_model, args.train_device)
    #     utils.load_weight(agent.online_net, args.load_model, "cpu")
    #     print("*****done*****")

    # agent = agent.to(args.train_device)
    # # agent = agent.to("cpu")
    # # optim = torch.optim.Adam(agent.online_net.parameters(), lr=args.lr, eps=args.eps)
    # eval_agent = agent.clone(args.train_device, {"vdn": False})
    # # eval_agent = agent.clone("cpu", {"vdn": False})

    # replay_buffer = rela.RNNPrioritizedReplay(
    #     args.replay_buffer_size,
    #     args.seed,
    #     args.priority_exponent if not args.no_replay_buffer_priority else 0.0,
    #     args.priority_weight if not args.no_replay_buffer_priority else 1.0,
    #     args.prefetch,
    # )

    # act_group = ActGroup(
    #     args.method,
    #     args.act_device,
    #     agent,
    #     args.num_thread,
    #     args.num_game_per_thread,
    #     args.multi_step,
    #     args.gamma,
    #     args.eta,
    #     args.max_len,
    #     args.num_player,
    #     replay_buffer,
    # )

    # assert args.shuffle_obs == False, 'not working with 2nd order aux'
    # context, threads = create_threads(
    #     args.num_thread, args.num_game_per_thread, act_group.actors, games,
    #     #use_softmax_policy=args.use_softmax_policy,
    #     #betas_range=torch.Tensor([float(x) for x in args.log_beta_range.split(",")])
    # )
    # act_group.start()
    # # else: # OP MODEL!
    # #     agents = load_op_model(args.method, args.idx, args.idx, args.train_device)
    # #     if agents is not None:
    # #         runners = [rela.BatchRunner(agent, args.train_device, 1000, ["act"]) for agent in agents]
    # #     num_player = len(runners)
    # #
    # #     context = rela.Context()
    # #     games = create_envs(
    # #         args.num_thread * args.num_game_per_thread,
    # #         args.seed,
    # #         args.num_player,
    # #         args.hand_size,
    # #         args.train_bomb,
    # #         explore_eps,
    # #         -1,
    # #         True, # sad flag
    # #         False,
    # #         False,
    # #     )
    # #
    # #     for g in games:
    # #         env = hanalearn.HanabiVecEnv()
    # #         env.append(g)
    # #         actors = []
    # #         for i in range(num_player):
    # #             actors.append(rela.R2D2Actor(runners[i], 1))
    # #         thread = hanalearn.HanabiThreadLoop(actors, env, True)
    # #         context.push_env_thread(thread)
    # #
    # #     for runner in runners:
    # #         runner.start()

    # context.start()
    
    # while replay_buffer.size() < args.burn_in_frames:
    #     print("warming up replay buffer:", replay_buffer.size())
    #     time.sleep(1)


    # COMMENTED TILL HERE FOR SINGLE REPLAY ^ ^ ^

    # time.sleep(5)
    # while True:
    #     batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
    #     print(batch.seq_len)
    #     priority = torch.zeros(batch.obs["priv_s"].shape[:2], device=batch.obs["priv_s"].device) + 1.0
    #     priority = rela.aggregate_priority(
    #         priority.cpu(), batch.seq_len.cpu(), args.eta
    #     )
    #     replay_buffer.update_priority(priority)
    #     time.sleep(2)

    print("Success, Done")
    print("=======================")

    frame_stat = dict()
    frame_stat["num_acts"] = 0
    frame_stat["num_buffer"] = 0

    # stat = common_utils.MultiCounter(args.save_dir)
    # tachometer = utils.Tachometer()
    stopwatch = common_utils.Stopwatch()

    # count_duplicates = 0
    # samples_seen = []
    # for j in range(100000//32):
    #     print(str(j+1)+"/"+str(100000//32))
    #     print(count_duplicates)
    #     batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
    #     obs = batch.obs["priv_s"].to("cpu")
    #     obs = torch.cat([obs, batch.action["a"].to("cpu").unsqueeze(-2).repeat(1,1,args.num_player,1).float()], -1)
    #     obs = torch.cat([obs, torch.zeros((80, args.batchsize, args.num_player, 18))], dim=3)
    #     obs = obs[:,:,0,:]
    #     for k in range(32):
    #         if list(obs[0,k,:]) not in samples_seen:
    #             samples_seen.append(obs[0,k,:])
    #         else:
    #             count_duplicates += 1

    #     priority = torch.zeros(batch.obs["priv_s"].shape[:2], device=batch.obs["priv_s"].device) + 1.0
    #     priority = rela.aggregate_priority(
    #         priority.cpu(), batch.seq_len.cpu(), args.eta
    #     )
    #     replay_buffer.update_priority(priority)
    #     torch.cuda.synchronize()
    # print(colours)

    ##############
    # Create LBS model
    #############

    # model = [get_model(args.encoder_dim, args.decoder_dim, args.N, args.num_heads).to(device) for _ in range(5)]
    # model = get_model(206, 28, 256, args.N, args.num_heads).to(device)
    model = Transformer(206, 28, 256, args.N, args.num_heads)
    model.load_state_dict(torch.load("current_belief.pth"))
    model = model.to(device)
    model.use = True
    # model = Transformer(5607, 28, 256, args.N, args.num_heads)
    # model.load_state_dict(torch.load("saves_while_training/model$_2player_belief_online_w_v0_pr.pth"))
    # model = model.to(device)
    # torch.save(model.encoder.embed.state_dict(), "saves_while_training/model_trainedsad6_embed.pth")
    # assert 1 == 2

    # # optim = [torch.optim.Adam(model[j].parameters(), lr=args.lr, betas=(0.9, 0.98), eps=args.eps) for j in range(5)]
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.998), eps=args.eps)

    ##############
    # END Create LBS model
    #############
    # print("Doing per card scores...")

    # batch, weight = replay_buffer.sample(1000, args.train_device)
    # x_test, y_test = get_samples(batch, 1000)
    # asdf = per_card_scores_online(x_test, y_test, model, batch)

    # print("Done per card scores!")

    # v0_enumeration = enumerate_v0()
    # hint_enumeration = enumerate_hints()

    # preds = torch.zeros((args.batchsize,1,26), device=device)
    # preds_so_far = torch.zeros((args.batchsize,5,26), device=device)

    # preds_test = torch.zeros((args.batchsize,1,26), device=device)
    # preds_so_far_test = torch.zeros((args.batchsize,5,26))
    # preds_so_far_test_ = torch.zeros((5,args.batchsize,5,26), device=device)

    obs = torch.zeros((args.batchsize, 80, 15)).type(torch.LongTensor)
    # ck = torch.zeros((args.batchsize, 2000)).type(torch.LongTensor)
    target = torch.zeros((args.batchsize, args.hand_size+2)).type(torch.LongTensor)

    obses = torch.zeros((args.batchsize*2, 80, 15)).type(torch.LongTensor)
    # cks = torch.zeros((args.batchsize*4, 2000)).type(torch.LongTensor)
    targets = torch.zeros((args.batchsize*2, args.hand_size+2)).type(torch.LongTensor)

    total_losses = np.zeros(5)
    # total_tests  = np.zeros(5)

    print_every = 50
    save_loss_every = 250
    save_entr_vs_timestep_every = 3000
    running_losses = [[] for _ in range(5)]
    # running_tests  = [[] for _ in range(5)]

    # best_model_scores = [100 for _ in range(5)]
    best_model_score = 100

    step_num = 1

    start = time.time()
    temp = start

    for epoch in range(args.num_epoch):
        print("beginning of epoch: " + str(epoch+1))
        print(common_utils.get_mem_usage())
        # tachometer.start()
        # stat.reset()
        stopwatch.reset()
        loss_lst_train = None

        for batch_idx in range(args.epoch_len):
            if epoch == -1: # DBG
                break
            num_update = batch_idx + epoch * args.epoch_len
            if num_update > 0 and num_update % args.num_update_between_sync == 0:
                torch.save(model.state_dict(), "current_belief.pth")
                torch.save(torch.tensor([0]), "reflection_phase.pt")
                print("Begin research phase.")
                print("Waiting for reflection phase...")
                start_time = time.time()
                while torch.load("reflection_phase.pt").item() == 0:
                    time.sleep(10)
                    if ((time.time()-start_time)//60 > 0 and (time.time()-start_time)//60 % 6 == 0 and (time.time()-start_time)%60 < 15):
                        print("Waiting for reflection phase...")
                # model = Transformer(206, 28, 256, args.N, args.num_heads)
                model.load_state_dict(torch.load("current_belief.pth"))
                model = model.to(device)
                print("Recommence reflection phase.")
                time.sleep(10)
                

            if batch_idx % 2 == 0:
                obses *= 0
                # cks *= 0
                targets *= 0
                while len(os.listdir('single_replay/')) < 8:
                    print("Waiting for more samples...")
                    time.sleep(10)

                while True:
                    while len(os.listdir('single_replay/')) < 8:
                        print("Waiting for more samples...")
                        time.sleep(1)
                    candidate = random.choice(os.listdir("single_replay/"))
                    while not (candidate.startswith("batches_") and candidate.endswith(".pt")):
                        while len(os.listdir('single_replay/')) < 8:
                            print("Waiting for more samples...")
                            time.sleep(1)
                        candidate = random.choice(os.listdir("single_replay/"))
                    try:
                        obses = torch.load("single_replay/"+candidate.replace("trg","src").replace("ck","src"))
                        # cks  = torch.load("single_replay/"+candidate.replace("src","ck").replace("trg","ck"))
                        targets = torch.load("single_replay/"+candidate.replace("src","trg").replace("ck","trg"))

                        os.remove("single_replay/"+candidate.replace("trg","src").replace("ck","src"))
                        # os.remove("single_replay/"+candidate.replace("src","ck").replace("trg","ck"))
                        os.remove("single_replay/"+candidate.replace("src","trg").replace("ck","trg"))
                        break

                    except:
                        try:
                            os.remove("single_replay/"+candidate.replace("trg","src").replace("ck","src"))
                        except:
                            pass
                        # try:
                        #     os.remove("single_replay/"+candidate.replace("src","ck").replace("trg","ck"))
                        # except:
                        #     pass
                        try:
                            os.remove("single_replay/"+candidate.replace("src","trg").replace("ck","trg"))
                        except:
                            pass

            # torch.cuda.synchronize()
            stopwatch.time("sync and updating")

            # batch, weight = replay_buffer.sample(args.batchsize, args.train_device)

            if args.belief_type == "public":
                obs = batch.obs["priv_s"][:, :, -feature_size:] if args.num_player < 5 else batch.obs["priv_s"][:, :, 0, -feature_size:]
                # if args.num_player < 5:
                #     obs = torch.cat([obs, batch.action["a"].unsqueeze(-1).float()], -1)
                # else:
                obs = torch.cat([obs, batch.action["a"].unsqueeze(-2).repeat(1, 1, args.num_player, 1).float()], -1)
                obs = obs[:torch.max(batch.seq_len).long().item()]
                target = batch.obs["own_hand"].view(*batch.obs["own_hand"].shape[:3], args.hand_size, 25) # have to add empty card dimension to target!
                target_empty_mask = (target.sum(-1, keepdim=True)==0.0)
                target = torch.cat([target, target_empty_mask.float()], -1)
                target = target[:torch.max(batch.seq_len).long().item()]
            elif args.belief_type == "own_hand":
                obs *= 0
                target *= 0
                # ck *= 0
                obs[:] = obses[(batch_idx % 2)*args.batchsize:(1 + batch_idx % 2)*args.batchsize,:]
                # ck[:]  = cks[(batch_idx % 4)*args.batchsize:(1 + batch_idx % 4)*args.batchsize,:]
                target[:] = targets[(batch_idx % 2)*args.batchsize:(1 + batch_idx % 2)*args.batchsize,:]
                # obs[:], ck[:], target[:] = get_samples(batch, args)[:]
                stopwatch.time("sample data")

            else:
                assert False, "Unknown belief type: {}".format(args.belief_type)
            # loss_lst_train, priority = agent.loss_lbs(lbs_net, obs, target, stat) # , args.pred_weight,
            # loss_lst_train_lst.append( loss_lst_train.detach().cpu())

            # MOD2: ReplayBuffer aggregate_priority not needed
            # preds *= 0
            # preds_so_far *= 0

            total_losses, step_num = belief_run(model,obs.type(torch.LongTensor),None,target.type(torch.LongTensor),total_losses,args,optim,step_num,stopwatch,device,'train')

            # priority = torch.zeros(batch.obs["priv_s"].shape[:2], device=batch.obs["priv_s"].device) + 1.0
            # priority = rela.aggregate_priority(
            #     priority.cpu(), batch.seq_len.cpu(), args.eta
            # )
            # replay_buffer.update_priority(priority)
            # stopwatch.time("updating priority")
            torch.cuda.synchronize()

            # batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
            # obs[:], target[:] = get_samples(batch, args.batchsize)[:]
            
            # preds *= 0
            # preds_so_far *= 0
            # preds_so_far_ *= 0

            # _, total_tests = belief_run(model,obs,target,batch,total_tests,args,preds,preds_so_far,preds_so_far_,'eval')

            # priority = torch.zeros(batch.obs["priv_s"].shape[:2], device=batch.obs["priv_s"].device) + 1.0
            # priority = rela.aggregate_priority(
            #     priority.cpu(), batch.seq_len.cpu(), args.eta
            # )
            # replay_buffer.update_priority(priority)
            # torch.cuda.synchronize()

            # loss = (loss_lst_train).mean()
            # loss.backward()

            # load another policy in here
            

            # g_norm = torch.nn.utils.clip_grad_norm_(
            #     agent.online_net.parameters(), args.grad_clip
            # )
            # optim.step()
            # optim.zero_grad()

            # MOD1: ReplayBuffer update_priotity is not needed

            # stat["loss"].feed(loss.detach().item())
            # stat["grad_norm"].feed(g_norm)

            if (batch_idx + 1) % print_every == 0:
                loss_avgs = [total_losses[j] / print_every for j in range(5)]
                # loss_test_avgs = [total_tests[j] / print_every for j in range(5)]
                print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters" % ((time.time() - start) // 60,
                epoch + 1, batch_idx + 1, 0.2*sum(loss_avgs), time.time() - temp,
                print_every))
                # print("1st card loss = %.3f, 1st card test = %.3f, 2nd card loss = %.3f, 2nd card test = %.3f, 3rd card loss = %.3f, 3rd card test = %.3f, 4th card loss = %.3f, 4th card test = %.3f, 5th card loss = %.3f, 5th card test = %.3f" % 
                # (loss_avgs[0], loss_test_avgs[0], loss_avgs[1], loss_test_avgs[1], loss_avgs[2], loss_test_avgs[2], loss_avgs[3], loss_test_avgs[3], loss_avgs[4], loss_test_avgs[4]))

                for j in range(5):
                    running_losses[j].append(loss_avgs[j])
                    # running_tests[j].append(loss_test_avgs[j])
                # for j in range(5):
                if 0.2*sum(loss_avgs) < best_model_score:
                    # torch.save(model[j].state_dict(),"/pyhanabi/saves_while_training/model"+str(j)+"_2player_belief_online.pth")
                    torch.save(model.state_dict(),"exps/exp1/best_belief_module.pth")
                    best_model_score = 0.2*sum(loss_avgs)

                total_losses = np.zeros(5)
                total_tests  = np.zeros(5)
                temp = time.time()

            if (batch_idx + 1) % (save_loss_every) == 0:
                np.save(os.path.join('exps', 'exp1', 'belief_module_running_loss.npy'), np.array(running_losses))
                # for j in range(5):
                    # np.save(os.path.join('saves_while_training', 'train_loss'+str(j)+'_online.npy'), np.array(running_losses[j]))
                    # np.save(os.path.join('saves_while_training', 'valid_loss'+str(j)+'_online.npy'), np.array(running_tests[j]))
            # if (batch_idx + 1) % (save_entr_vs_timestep_every) == 0:
            #     print("Calculating per card cross entropies...")
            #     batch, weight = replay_buffer.sample(1000, args.train_device)
            #     x_test, y_test = get_samples(batch, args)
            #     # np.save(os.path.join('saves_while_training', 'per_card_scores_online.npy'), per_card_scores_online(x_test, y_test, model, batch))
            #     print("Calculated.")

        stopwatch.summary()

        count_factor = args.num_player if args.method == "vdn" else 1
        print("EPOCH: " + str(epoch+1))
        # tachometer.lap(
        #     act_group.actors, replay_buffer, args.epoch_len * args.batchsize, count_factor
        # )
        # stopwatch.summary()
        # stat.summary(epoch)

        # context.pause()

        # eval_seed = (9917 + epoch * 999999) % 7777777
        # eval_agent.load_state_dict(agent.state_dict())

        # print("EVALUATE - START")
        # print([eval_agent for _ in range(args.num_player)],
        #     1000,
        #     eval_seed,
        #     args.eval_bomb,
        #     0,  # explore eps
        #     args.sad,
        #     args.obs_n_most_recent_own_cards,
        #     args.hand_size,
        #     args.use_softmax_policy,
        #     [float(b) for b in args.eval_log_betas.split(",")])
        # print("EVALUATE - END")
        # quit()

        # print("Evaluating policy performance...")
        # d = evaluate(
        #     [eval_agent for _ in range(args.num_player)],
        #     1000,
        #     eval_seed,
        #     args.eval_bomb,
        #     0,  # explore eps
        #     args.sad,
        #     #args.obs_n_most_recent_own_cards,
        #     hand_size=args.hand_size,
        # )
        torch.cuda.synchronize()

        # commented from here epoch -1 break code v v v

        # print("Evaluating belief model performance...")
        # stopwatch.time("sync and updating")

        # # batch, weight = replay_buffer.sample(args.batchsize, args.train_device)

        # if args.belief_type == "public":#
        #     assert False, "public belief not currently supported!"
        #     # if args.num_player < 5:
        #     #     obs = batch.obs["priv_s"][:, :, -feature_size:]
        #     #     obs = torch.cat([obs, batch.action["a"].unsqueeze(-1).float()], -1)
        #     #     target = batch.obs["own_hand"].view(batch.obs["own_hand"].shape[0],
        #     #                                         batch.obs["own_hand"].shape[1],
        #     #                                         args.hand_size * args.num_player * args.card_feature_dim)
        #     # else:
        #     obs = batch.obs["priv_s"][:, :, 0, -feature_size:]
        #     obs = torch.cat([obs, batch.action["a"].unsqueeze(-2).repeat(1, 1, args.num_player, 1).float()], -1)
        #     # TODO: This does not work for public belief!
        #     target = batch.obs["own_hand"].view(*batch.obs["own_hand"].shape[:3], args.hand_size,
        #                                         25)  # have to add empty card dimension to target!
        #     target_empty_mask = (target.sum(-1, keepdim=True) == 0.0)
        #     target = torch.cat([target, target_empty_mask.float()], -1)
        #     obs = obs[:torch.max(batch.seq_len).long().item()]
        #     target = target[:torch.max(batch.seq_len).long().item()]
        # elif args.belief_type == "own_hand":

        #     obs *= 0
        #     target *= 0
        #     ck *= 0
        #     obs[:] = obses[(batch_idx % 25)*args.batchsize:(1 + batch_idx % 25)*args.batchsize,:]
        #     ck[:]  = cks[(batch_idx % 25)*args.batchsize:(1 + batch_idx % 25)*args.batchsize,:]
        #     target[:] = targets[(batch_idx % 25)*args.batchsize:(1 + batch_idx % 25)*args.batchsize,:]
        #     # obs[:], ck[:], target[:] = get_samples(batch, args)[:]
        #     stopwatch.time("sample data")

        # else:
        #     assert False, "Unknown belief type: {}".format(args.belief_type)
        # # loss_lst, priority = agent.loss_lbs(lbs_net, obs, target, stat, eval=True)  # , args.pred_weight,
        # # loss_lst_lst.append(loss_lst.detach().cpu())

        # # preds *= 0
        # # preds_so_far *= 0

        # total_losses = belief_run(model,obs,ck,target,total_losses,args,optim,stopwatch,device,'train')

        # # priority = torch.zeros(batch.obs["priv_s"].shape[:2], device=batch.obs["priv_s"].device) + 1.0
        # # priority = rela.aggregate_priority(
        # #     priority.cpu(), batch.seq_len.cpu(), args.eta
        # # )
        # # replay_buffer.update_priority(priority)
        # torch.cuda.synchronize()

        # # batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
        # # obs[:], target[:] = get_samples(batch, args.batchsize)[:]

        # # preds_test *= 0
        # # preds_so_far_test *= 0
        # # preds_so_far_test_ *= 0
        
        # # _, total_tests = belief_run(model,obs,target,batch,total_tests,args,preds_test,preds_so_far_test,preds_so_far_test_,'eval')

        # # priority = torch.zeros(batch.obs["priv_s"].shape[:2], device=batch.obs["priv_s"].device) + 1.0
        # # priority = rela.aggregate_priority(
        # #     priority.cpu(), batch.seq_len.cpu(), args.eta
        # # )
        # # replay_buffer.update_priority(priority)
        # # torch.cuda.synchronize()

        # # for j in range(5):
        # #     optim[j].zero_grad()
        # # loss = sum(losses)
        # # loss.backward()
        # # for j in range(5):
        # #     optim[j].step()


        # # MOD2: ReplayBuffer aggregate_priority not needed

        # # if loss_lst_train is not None:
        # #     loss_train = loss_lst_train.mean().detach()
        # #     print("CrossEntropyLoss: ", loss_train)

        # # loss = loss_lst.mean().detach()
        # # print("Category Loss: ", loss)

        # loss_avgs = [total_losses[j] for j in range(5)]
        # # loss_test_avgs = [total_tests[j]for j in range(5)]
        # print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters" % ((time.time() - start) // 60,
        # epoch + 1, batch_idx + 1, 0.2*sum(loss_avgs), time.time() - temp,
        # 1))
        # # print("1st card loss = %.3f, 1st card test = %.3f, 2nd card loss = %.3f, 2nd card test = %.3f, 3rd card loss = %.3f, 3rd card test = %.3f, 4th card loss = %.3f, 4th card test = %.3f, 5th card loss = %.3f, 5th card test = %.3f" % 
        # # (loss_avgs[0], loss_test_avgs[0], loss_avgs[1], loss_test_avgs[1], loss_avgs[2], loss_test_avgs[2], loss_avgs[3], loss_test_avgs[3], loss_avgs[4], loss_test_avgs[4]))
        # total_losses = np.zeros(5)
        # # total_tests  = np.zeros(5)
        # temp = time.time()

        # commented till here epoch -1 break code ^ ^ ^

        # if epoch > 0 and epoch % 50 == 0:
        #     force_save_name = "model_epoch%d" % epoch
        # else:
        #     force_save_name = None

        # if epoch == -1 or epoch%args.eval_epochs == 0:
        #     print("Saving belief model...")
        #     torch.save({"LBSBeliefNet": lbs_net.state_dict()}, os.path.join(args.save_dir, "lbs_model_ep{}.pthw".format(epoch)))
        #     print("Saving cross entropy graph...")
        #     import matplotlib.pyplot as plt

        #     loss_lst = loss_lst.detach().view(loss_lst.shape[0], -1).mean(-1).cpu()
        #     print("crossentropy data (val - hard):")
        #     print(loss_lst)

            # Plot average cross-entropy per step
            # plt.plot(list(range(len(loss_lst))), loss_lst, label="val")
            # if loss_lst_train is not None:
            #     loss_lst_train = loss_lst_train.detach().view(loss_lst_train.shape[0], -1).mean(-1).cpu()
            #     plt.plot(list(range(len(loss_lst_train))), loss_lst_train, label="train")
            #     print("crossentropy data (train - soft):")
            #     print(loss_lst_train)

            # plt.xlabel('steps')
            # plt.ylabel('cross entropy (whole hand)')
            # plt.title("Epoch {}".format(epoch))
            # plt.legend()
            # plt.savefig(os.path.join(args.save_dir, "lbs_model_ep{}.png".format(epoch)), bbox_inches='tight')
            # plt.clf()

            # Train loss over time plot
            # if loss_lst_train_lst:
            #     #lt = torch.stack(loss_lst_train_lst, 0)
            #     # lt = pad_sequence(loss_lst_train_lst, batch_first=True).permute(1,0,2,3)
            #     # lt = lt.view(lt.shape[0], -1).mean(-1, keepdim=True)
            #     _lt = [ _l.mean() for _l in loss_lst_train_lst]
            #     plt.plot(list(range(len(loss_lst_train_lst))), _lt, label="Train cross-entropy loss")
            #     # for i, _lt in enumerate(lt):
            #     #    plt.plot(list(range(i+1)), _lt, label="game steps {}".format(i))
            #     plt.xlabel('training episodes')
            #     plt.ylabel('cross entropy (whole hand)')
            #     plt.title("Epoch {}".format(epoch))
            #     plt.legend()
            #     plt.savefig(os.path.join(args.save_dir, "lbs_model_trainloss_ep{}.png".format(epoch)), bbox_inches='tight')
            #     plt.clf()

            # Eval loss over time plot
            # if loss_lst_lst:
            #     #lt = pad_sequence(loss_lst_lst, batch_first=True)
            #     # lt = pad_sequence(loss_lst_lst, batch_first=True).permute(1, 0, 2, 3)
            #     # lt = lt.view(lt.shape[0], -1).mean(-1, keepdim=True)
            #     _lt = [ _l.mean() for _l in loss_lst_lst]
            #     plt.plot(list(range(len(loss_lst_lst))), _lt, label="Eval category loss")
            #     plt.xlabel('training episodes')
            #     plt.ylabel('cross entropy (whole hand)')
            #     plt.title("Epoch {}".format(epoch))
            #     plt.legend()
            #     plt.savefig(os.path.join(args.save_dir, "lbs_model_valloss_ep{}.png".format(epoch)), bbox_inches='tight')
            #     plt.clf()

            #score, score_std, perfect, *_ = d
            # scores_mean = d["scores_mean"]
            # scores_std = d["scores_std"]
            # fraction_perfect = d["fraction_perfect"]
            # #model_saved = saver.save(
            # #    None, agent.online_net.state_dict(), scores_mean, force_save_name=force_save_name
            # #)
            # print(
            #     "epoch %d, eval score: %.4f, eval score std: %.4f, fraction perfect: %.2f"
            #     % (epoch, scores_mean, scores_std, fraction_perfect * 100)
            # )

        # context.resume()
        print("==========")
