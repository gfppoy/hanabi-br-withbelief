# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Tuple, Dict
import common_utils
from transformer_embedding import get_model
from td_methods import compute_belief

class R2D2Net(torch.jit.ScriptModule):
    __constants__ = [
        "hid_dim",
        "out_dim",
        "num_lstm_layer",
        "hand_size",
        "skip_connect",
    ]

    def __init__(
        self,
        device,
        in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        hand_size,
        num_fc_layer,
        skip_connect,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_fc_layer = num_fc_layer
        self.num_lstm_layer = num_lstm_layer
        self.hand_size = hand_size
        self.skip_connect = skip_connect

        ff_layers = [nn.Linear(self.in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_fc_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim, self.hid_dim, num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        # for aux task
        self.pred = nn.Linear(self.hid_dim, self.hand_size * 3)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self, priv_s: torch.Tensor, hid: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert priv_s.dim() == 2, "dim should be 2, [batch, dim], get %d" % s.dim()

        priv_s = priv_s.unsqueeze(0)
        x = self.net(priv_s)
        o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        if self.skip_connect:
            o = o + x
        a = self.fc_a(o)
        a = a.squeeze(0)
        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            one_step = True

        x = self.net(priv_s)
        if len(hid) == 0:
            o, (h, c) = self.lstm(x)
        else:
            o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        a = self.fc_a(o)
        v = self.fc_v(o)
        q = self._duel(v, a, legal_move)

        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)
        return qa, greedy_action, q, o

    @torch.jit.script_method
    def _duel(
        self, v: torch.Tensor, a: torch.Tensor, legal_move: torch.Tensor
    ) -> torch.Tensor:
        assert a.size() == legal_move.size()
        legal_a = a * legal_move
        q = v + legal_a - legal_a.mean(2, keepdim=True)
        return q

    def cross_entropy(self, net, lstm_o, target_p, hand_slot_mask, seq_len):
        # target_p: [seq_len, batch, num_player, 5, 3]
        # hand_slot_mask: [seq_len, batch, num_player, 5]
        logit = net(lstm_o).view(target_p.size())
        q = nn.functional.softmax(logit, -1)
        logq = nn.functional.log_softmax(logit, -1)
        plogq = (target_p * logq).sum(-1)
        xent = -(plogq * hand_slot_mask).sum(-1) / hand_slot_mask.sum(-1).clamp(
            min=1e-6
        )

        if xent.dim() == 3:
            # [seq, batch, num_player]
            xent = xent.mean(2)

        # save before sum out
        seq_xent = xent
        xent = xent.sum(0)
        assert xent.size() == seq_len.size()
        avg_xent = (xent / seq_len).mean().item()
        return xent, avg_xent, q, seq_xent.detach()

    def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
        return self.cross_entropy(self.pred, lstm_o, target, hand_slot_mask, seq_len)


class R2D2Agent(torch.jit.ScriptModule):
    __constants__ = ["vdn", "multi_step", "gamma", "eta", "uniform_priority"]

    def __init__(
        self,
        vdn,
        multi_step,
        gamma,
        eta,
        device,
        in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        hand_size,
        uniform_priority,
        *,
        num_fc_layer=1,
        skip_connect=False,
    ):
        super().__init__()
        self.online_net = R2D2Net(
            device,
            in_dim,
            hid_dim,
            out_dim,
            num_lstm_layer,
            hand_size,
            num_fc_layer,
            skip_connect,
        ).to(device)
        self.target_net = R2D2Net(
            device,
            in_dim,
            hid_dim,
            out_dim,
            num_lstm_layer,
            hand_size,
            num_fc_layer,
            skip_connect,
        ).to(device)
        self.belief_module = get_model(
            src_vocab = 206,
            trg_vocab = 28,
            d_model = 256,
            N = 6,
            heads = 8
        ).to(device)
        self.vdn = vdn
        self.multi_step = multi_step
        self.gamma = gamma
        self.eta = eta
        self.uniform_priority = uniform_priority
        self.device = device

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        return self.online_net.get_h0(batchsize)

    def clone(self, device, overwrite=None):
        if overwrite is None:
            overwrite = {}
        cloned = type(self)(
            overwrite.get("vdn", self.vdn),
            self.multi_step,
            self.gamma,
            self.eta,
            device,
            self.online_net.in_dim,
            self.online_net.hid_dim,
            self.online_net.out_dim,
            self.online_net.num_lstm_layer,
            self.online_net.hand_size,
            self.uniform_priority,
            num_fc_layer=self.online_net.num_fc_layer,
            skip_connect=self.online_net.skip_connect,
        )
        cloned.load_state_dict(self.state_dict())
        return cloned.to(device)

    def sync_target_with_online(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.jit.script_method
    def greedy_act(
        self,
        priv_s: torch.Tensor,
        legal_move: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        adv, new_hid = self.online_net.act(priv_s, hid)
        legal_adv = (1 + adv - adv.min()) * legal_move
        greedy_action = legal_adv.argmax(1).detach()
        return greedy_action, new_hid

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape
            [batchsize] or [batchsize, num_player]
        """
        obsize, ibsize, num_player = 0, 0, 0
        priv_s = obs["priv_s"].detach()

        #if self.vdn:
        #    nopeak_mask = torch.triu(torch.ones((1, 6, 6)), diagonal=1)
        #    nopeak_mask = (nopeak_mask == 0).to("cuda:1")

        #    obsize, ibsize, num_player = obs["priv_s"].size()[:3]

        #    priv_s[:,:,:,433:783] = 0

        #    if self.belief_module.use: 
        #        #TODO: might be mistake that for each player only do half the steps, do for all? 
        #        # src is bs x seq_len x 2 x 15
        #        # trg is bs x seq_len x 2 x 7
        #        src, trg = self.belief_module.get_samples(obs["priv_s"], obs["own_hand"], torch.zeros(obs["priv_s"].size(1), dtype=torch.int) + obs["priv_s"].size(0), device="cuda:1")

        #        leftover = src.size(1)%4
        #        for j in range(src.size(1)//4):
        #            temp = torch.zeros(([2*src.size(0), src.size(1), 15]), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 15
        #            for i in range(2):
        #                if i == 0:
        #                    temp[:] = src[:, :, i, :].repeat([2, 1, 1])
        #                    temp[0:src.size(0), 4*j+1:, :] = 205
        #                    temp[src.size(0):2*src.size(0), 4*j+3:, :] = 205
        #                    targets = torch.cat((trg[:, 4*j, i, :-1], 
        #                                        trg[:, 4*j+2, i, :-1]), dim=0).detach()
        #                    preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()

        #                    priv_s[4*j,:,i,433:563] = preds[0:src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
        #                    priv_s[4*j+2,:,i,433:563] = preds[src.size(0):2*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)

        #                else:
        #                    temp[:] = src[:, :, i, :].repeat([2, 1, 1])
        #                    temp[0:src.size(0), 4*j+2:, :] = 205
        #                    temp[src.size(0):2*src.size(0), 4*j+4:, :] = 205
        #                    targets = torch.cat((trg[:, 4*j+1, i, :-1],
        #                                        trg[:, 4*j+3, i, :-1]), dim=0).detach()
        #                    preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()

        #                    priv_s[4*j+1,:,i,433:563] = preds[0:src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
        #                    priv_s[4*j+3,:,i,433:563] = preds[src.size(0):2*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)

        #        if leftover:
        #            for i in range(2):
        #                if i == 0:
        #                    temp = torch.zeros(([(leftover+1)//2*src.size(0), src.size(1), 15]), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 15
        #                    temp[:] = src[:, :, i, :].repeat([(leftover+1)//2, 1, 1])
        #                    for k in range((leftover+1)//2):
        #                        temp[k*src.size(0):(k+1)*src.size(0), src.size(1)-leftover+2*k+1:, :] = 205
        #                    targets = trg[:, src.size(1)-leftover, i, :-1].detach()
        #                    for k in range(2, leftover, 2):
        #                        targets = torch.cat((targets, trg[:, src.size(1)-leftover+k, i, :-1]), dim=0).detach()

        #                    preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()
        #                    count = 0
        #                    for k in range(0, leftover, 2):
        #                        priv_s[src.size(1)-leftover+k,:,i,433:563] = preds[count*src.size(0):(count+1)*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
        #                        count += 1
        #                
        #                if i == 1:
        #                    if leftover >= 2:
        #                        temp = torch.zeros(([(leftover)//2*src.size(0), src.size(1), 15]), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 15
        #                        temp[:] = src[:, :, i, :].repeat([(leftover)//2, 1, 1])
        #                        for k in range((leftover)//2):
        #                            temp[k*src.size(0):(k+1)*src.size(0), src.size(1)-leftover+2*k+2:, :] = 205
        #                        targets = trg[:, src.size(1)-leftover+1, i, :-1].detach()
        #                        for k in range(2,leftover,2):
        #                            targets = torch.cat((targets, trg[:, src.size(1)-leftover+k, i, :-1]), dim=0).detach()

        #                        preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()
        #                        count = 0
        #                        for k in range(1, leftover, 2):
        #                            priv_s[src.size(1)-leftover+k,:,i,433:563] = preds[count*src.size(0):(count+1)*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
        #                            count += 1
        #    
        #    priv_s = priv_s.flatten(0, 2)
        #    legal_move = obs["legal_move"].flatten(0, 2)
        #    eps = obs["eps"].flatten(0, 2)
        #else:
        nopeak_mask = torch.triu(torch.ones((1, 6, 6)), diagonal=1)
        nopeak_mask = (nopeak_mask == 0).to(self.device)

        obsize, ibsize = obs["priv_s"].size()[:2]
        num_player = 1
            
        #priv_s[torch.zeros(obs["priv_s"].size(1), dtype=torch.int) + obs["priv_s"].size(0)-1, range(obs["priv_s"].size(0), 433:783] = 0
        assert(not self.vdn)
        #if self.belief_module.use:
            # src is bs x seq_len x 15
            # trg is bs x seq_len x 7
        if obs["priv_s"].size(1) == 1:
            priv_s = priv_s.transpose(0, 1)
            bs = obs["priv_s"].size(0)
            seq_len = obs["priv_s"].size(1)
            src, _  = self.belief_module.get_samples_one_player(torch.cat([obs["priv_s"].transpose(0,1), torch.zeros((79, bs, 838), device=self.device)]), obs["own_hand"].transpose(0,1), torch.zeros((bs), dtype=torch.long, device=self.device) + seq_len, device=self.device)
        else:
            src, _ = self.belief_module.get_samples_one_player(torch.cat([obs["priv_s"], torch.zeros((80-obs["priv_s"].size(0), obs["priv_s"].size(1), 838), device=self.device)]), obs["own_hand"], torch.zeros((obs["priv_s"].size(1)), dtype=torch.long, device=self.device) + obs["priv_s"].size(0), device=self.device)

        #  temp = torch.zeros(([src.size(0), src.size(1), 15]), dtype=torch.long, device="cuda:1") # bs x seq_len x 15
        #  for j in range(src.size(1)):
        #      temp[:] = src[:, :, :]
        #      temp[:, j+1:, :] = 205
        #      preds = self.belief_module(temp, trg[:, j, :-1].long(), None, nopeak_mask)#.to("cpu")
        #      priv_s[j,:,433:563] = F.softmax(preds[:, :-1, 0:26], dim=-1).reshape(src.size(0), 5 * 26)

        #gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor([0.]), torch.tensor([1.]))

        assert(torch.all(0 == torch.sum(priv_s[:,:,0:125], -1)))

        targets = 26 + torch.zeros((src.size(0), 6), dtype=torch.long, device=self.device).detach() # bs x seq_len x 6
        j_card_dist = torch.zeros((src.size(0), 28), dtype=torch.long, device=self.device).detach()
        temp = torch.zeros((src.size(0)), dtype=torch.long, device=self.device).detach()
        for j in range(5):
            while True:
                j_card_dist = F.softmax(self.belief_module(src, targets, None, nopeak_mask)[:,j,:], dim=-1).detach()
                temp = torch.multinomial(j_card_dist, 1)#torch.argmax(torch.log(j_card_dist) + gumbel_dist.sample(sample_shape=j_card_dist.shape).squeeze(-1), axis=1)
                if not torch.any(temp==26) and not torch.any(temp==27):
                    break
            targets[:,j+1] = temp.reshape(src.size(0))
       #     priv_s[:, :, 0:125] = 1
       #     priv_s[priv_s.size(0)-1, :, 25*j:25*(j+1)] = j_card_dist[:, 0:25]
            priv_s[:, :, 25*j:25*(j+1)] = j_card_dist[:, 0:25]

        #  leftover = src.size(1)%4
        #  for j in range(src.size(1)//4):
        #      temp = torch.zeros(([4*src.size(0), src.size(1), 15]), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 15
        #      temp[:] = src[:, :, :].repeat([4, 1, 1])
        #      temp[0:src.size(0), 4*j+1:, :] = 205
        #      temp[src.size(0):2*src.size(0), 4*j+2:, :] = 205
        #      temp[2*src.size(0):3*src.size(0), 4*j+3:, :] = 205
        #      temp[3*src.size(0):4*src.size(0), 4*j+4:, :] = 205

        #      targets = torch.cat((trg[:, 4*j, :-1], 
        #                              trg[:, 4*j+1, :-1],
        #                              trg[:, 4*j+2, :-1],
        #                              trg[:, 4*j+3, :-1]), dim=0).detach()

        #      preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()
        #      priv_s[4*j,:,433:563] = preds[0:src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
        #      priv_s[4*j+1,:,433:563] = preds[src.size(0):2*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
        #      priv_s[4*j+2,:,433:563] = preds[2*src.size(0):3*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
        #      priv_s[4*j+3,:,433:563] = preds[3*src.size(0):4*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
        #  if leftover:
        #      temp = torch.zeros(([leftover*src.size(0), src.size(1), 15]), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 15
        #      temp[:] = src[:, :, :].repeat([leftover, 1, 1])
        #      for k in range(leftover):
        #          temp[k*src.size(0):(k+1)*src.size(0), src.size(1)-(leftover-k-1):, :] = 205
        #      targets = trg[:, src.size(1)-leftover, :-1].detach()
        #      for k in range(1,leftover):
        #          targets = torch.cat((targets, trg[:, src.size(1)-(leftover-k), :-1]), dim=0).detach()

        #      preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()
        #      for k in range(leftover):
        #          priv_s[src.size(1)-(leftover-k),:,433:563] = preds[k*src.size(0):(k+1)*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)

        priv_s = priv_s.flatten(0, 1)
        legal_move = obs["legal_move"].flatten(0, 1)
        eps = obs["eps"].flatten(0, 1)

        hid = {
            "h0": obs["h0"].flatten(0, 1).transpose(0, 1).contiguous(),
            "c0": obs["c0"].flatten(0, 1).transpose(0, 1).contiguous(),
        }

        greedy_action, new_hid = self.greedy_act(priv_s, legal_move, hid)

        random_action = legal_move.multinomial(1).squeeze(1)
        rand = torch.rand(greedy_action.size(), device=greedy_action.device)
        assert rand.size() == eps.size()
        rand = (rand < eps).long()
        action = (greedy_action * (1 - rand) + random_action * rand).detach().long()

        if self.vdn:
            action = action.view(obsize, ibsize, num_player)
            greedy_action = greedy_action.view(obsize, ibsize, num_player)
            rand = rand.view(obsize, ibsize, num_player)
        else:
            action = action.view(obsize, ibsize)
            greedy_action = greedy_action.view(obsize, ibsize)
            rand = rand.view(obsize, ibsize)

        hid_shape = (
            obsize,
            ibsize * num_player,
            self.online_net.num_lstm_layer,
            self.online_net.hid_dim,
        )
        h0 = new_hid["h0"].transpose(0, 1).view(*hid_shape)
        c0 = new_hid["c0"].transpose(0, 1).view(*hid_shape)

        reply = {
            "a": action.detach().cpu(),
            "greedy_a": greedy_action.detach().cpu(),
            "h0": h0.contiguous().detach().cpu(),
            "c0": c0.contiguous().detach().cpu(),
        }
        return reply

    @torch.jit.script_method
    def compute_priority(
        self, input_: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        compute priority for one batch
        """
        if self.uniform_priority:
            return {"priority": torch.ones_like(input_["reward"]).detach().cpu()}

        obsize, ibsize, num_player = 0, 0, 0
        flatten_end = 0
        if self.vdn:
            obsize, ibsize, num_player = input_["priv_s"].size()[:3]
            flatten_end = 2
        else:
            obsize, ibsize = input_["priv_s"].size()[:2]
            num_player = 1
            flatten_end = 1

        priv_s = input_["priv_s"].detach()

        #priv_s[:,433:783] = 0

        nopeak_mask = torch.triu(torch.ones((1, 6, 6)), diagonal=1)
        nopeak_mask = (nopeak_mask == 0).to("cuda:1").detach()

       # if self.belief_module.use:

       #     # src is bs x seq_len x 2 x 15
       #     # trg is bs x seq_len x 2 x 7
       #     if self.vdn:
       #         src, trg = self.belief_module.get_samples(input_["priv_s"], input_["own_hand"], torch.zeros(input_["priv_s"].size(1), dtype=torch.int) + input_["priv_s"].size(0), device="cuda:1")
       #         
       #         leftover = src.size(1)%4
       #         for j in range(src.size(1)//4):
       #             temp = torch.zeros(([2*src.size(0), src.size(1), 15]), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 15
       #             for i in range(2):
       #                 if i == 0:
       #                     temp[:] = src[:, :, i, :].repeat([2, 1, 1])
       #                     temp[0:src.size(0), 4*j+1:, :] = 205
       #                     temp[src.size(0):2*src.size(0), 4*j+3:, :] = 205
       #                     targets = torch.cat((trg[:, 4*j, i, :-1], 
       #                                         trg[:, 4*j+2, i, :-1]), dim=0).detach()
       #                     preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()

       #                     priv_s[4*j,:,i,433:563] = preds[0:src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
       #                     priv_s[4*j+2,:,i,433:563] = preds[src.size(0):2*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)

       #                 else:
       #                     temp[:] = src[:, :, i, :].repeat([2, 1, 1])
       #                     temp[0:src.size(0), 4*j+2:, :] = 205
       #                     temp[src.size(0):2*src.size(0), 4*j+4:, :] = 205
       #                     targets = torch.cat((trg[:, 4*j+1, i, :-1],
       #                                         trg[:, 4*j+3, i, :-1]), dim=0).detach()
       #                     preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()

       #                     priv_s[4*j+1,:,i,433:563] = preds[0:src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
       #                     priv_s[4*j+3,:,i,433:563] = preds[src.size(0):2*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)

       #         if leftover:
       #             for i in range(2):
       #                 if i == 0:
       #                     temp = torch.zeros(([(leftover+1)//2*src.size(0), src.size(1), 15]), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 15
       #                     temp[:] = src[:, :, i, :].repeat([(leftover+1)//2, 1, 1])
       #                     for k in range((leftover+1)//2):
       #                         temp[k*src.size(0):(k+1)*src.size(0), src.size(1)-leftover+2*k+1:, :] = 205
       #                     targets = trg[:, src.size(1)-leftover, i, :-1].detach()
       #                     for k in range(2, leftover, 2):
       #                         targets = torch.cat((targets, trg[:, src.size(1)-leftover+k, i, :-1]), dim=0).detach()

       #                     preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()
       #                     count = 0
       #                     for k in range(0, leftover, 2):
       #                         priv_s[src.size(1)-leftover+k,:,i,433:563] = preds[count*src.size(0):(count+1)*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
       #                         count += 1
       #                 
       #                 if i == 1:
       #                     if leftover >= 2:
       #                         temp = torch.zeros(([(leftover)//2*src.size(0), src.size(1), 15]), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 15
       #                         temp[:] = src[:, :, i, :].repeat([(leftover)//2, 1, 1])
       #                         for k in range((leftover)//2):
       #                             temp[k*src.size(0):(k+1)*src.size(0), src.size(1)-leftover+2*k+2:, :] = 205
       #                         targets = trg[:, src.size(1)-leftover+1, i, :-1].detach()
       #                         for k in range(2,leftover,2):
       #                             targets = torch.cat((targets, trg[:, src.size(1)-leftover+k, i, :-1]), dim=0).detach()

       #                         preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()
       #                         count = 0
       #                         for k in range(1, leftover, 2):
       #                             priv_s[src.size(1)-leftover+k,:,i,433:563] = preds[count*src.size(0):(count+1)*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
       #                             count += 1

       #                 # temp[:] = src[:, :, i, :].repeat([leftover, 1, 1])
       #                 # for k in range(leftover):
       #                 #     temp[k*src.size(0):(k+1)*src.size(0), src.size(1)-(leftover-k-1):, :] = 205
       #                 # targets = trg[:, src.size(1)-leftover, i, :-1].detach()
       #                 # for k in range(1,leftover):
       #                 #     targets = torch.cat((targets, trg[:, src.size(1)-(leftover-k), i, :-1]), dim=0).detach()

       #                 # preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()
       #                 # for k in range(leftover):
       #                 #     priv_s[src.size(1)-(leftover-k),:,i,433:563] = preds[k*src.size(0):(k+1)*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
          
            # src is bs x seq_len x 15
            # trg is bs x seq_len x 7 
       #     else:

        if input_["priv_s"].size(1) == 1:
            priv_s = priv_s.transpose(0, 1)
            bs = input_["priv_s"].size(0)
            seq_len = input_["priv_s"].size(1)
            src, _  = self.belief_module.get_samples_one_player(torch.cat([input_["priv_s"].transpose(0,1), torch.zeros((79, bs, 838), device="cuda:1" )]), input_["own_hand"].transpose(0,1), torch.zeros((bs), dtype=torch.long, device="cuda:1") + seq_len, device="cuda:1")
        else:
            src, _ = self.belief_module.get_samples_one_player(torch.cat([input_["priv_s"], torch.zeros((80-input_["priv_s"].size(0), input_["priv_s"].size(1), 838), device="cuda:1" )]), input_["own_hand"], torch.zeros((input_["priv_s"].size(1)), dtype=torch.long, device="cuda:1") + input_["priv_s"].size(0), device="cuda:1")

        assert(torch.all(0 == torch.sum(priv_s[:,:,0:125], -1)))

        targets = 26 + torch.zeros((src.size(0), 6), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 6
        j_card_dist = torch.zeros((src.size(0), 28), dtype=torch.long, device="cuda:1").detach()
        temp = torch.zeros((src.size(0)), dtype=torch.long, device="cuda:1").detach()
        for j in range(5):
            while True:
                j_card_dist = F.softmax(self.belief_module(src, targets, None, nopeak_mask)[:,j,:], dim=-1).detach()
                temp = torch.multinomial(j_card_dist, 1)#torch.argmax(torch.log(j_card_dist) + gumbel_dist.sample(sample_shape=j_card_dist.shape).squeeze(-1), axis=1)
                if not torch.any(temp==26) and not torch.any(temp==27):
                    break
            targets[:,j+1] = temp.reshape(src.size(0))
      #      priv_s[:, :, 0:125] = 1
      #      priv_s[priv_s.size(0)-1, :, 25*j:25*(j+1)] = j_card_dist[:, 0:25]
            priv_s[:, :, 25*j:25*(j+1)] = j_card_dist[:, 0:25]

       # leftover = src.size(1)%4
       # for j in range(src.size(1)//4):
       #     temp = torch.zeros(([4*src.size(0), src.size(1), 15]), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 15
       #     temp[:] = src[:, :, :].repeat([4, 1, 1])
       #     temp[0:src.size(0), 4*j+1:, :] = 205
       #     temp[src.size(0):2*src.size(0), 4*j+2:, :] = 205
       #     temp[2*src.size(0):3*src.size(0), 4*j+3:, :] = 205
       #     temp[3*src.size(0):4*src.size(0), 4*j+4:, :] = 205

       #     targets = torch.cat((trg[:, 4*j, :-1],
       #                             trg[:, 4*j+1, :-1],
       #                             trg[:, 4*j+2, :-1],
       #                             trg[:, 4*j+3, :-1]), dim=0).detach()

       #     preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()
       #     priv_s[4*j,:,433:563] = preds[0:src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
       #     priv_s[4*j+1,:,433:563] = preds[src.size(0):2*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
       #     priv_s[4*j+2,:,433:563] = preds[2*src.size(0):3*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
       #     priv_s[4*j+3,:,433:563] = preds[3*src.size(0):4*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
       # if leftover:
       #     temp = torch.zeros(([leftover*src.size(0), src.size(1), 15]), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 15
       #     temp[:] = src[:, :, :].repeat([leftover, 1, 1])
       #     for k in range(leftover):
       #         temp[k*src.size(0):(k+1)*src.size(0), src.size(1)-(leftover-k-1):, :] = 205
       #     targets = trg[:, src.size(1)-leftover, :-1].detach()
       #     for k in range(1,leftover):
       #         targets = torch.cat((targets, trg[:, src.size(1)-(leftover-k), :-1]), dim=0).detach()

       #     preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()
       #     for k in range(leftover):
       #         priv_s[src.size(1)-(leftover-k),:,433:563] = preds[k*src.size(0):(k+1)*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)


        priv_s = priv_s.flatten(0, flatten_end)
        legal_move = input_["legal_move"].flatten(0, flatten_end)
        online_a = input_["a"].flatten(0, flatten_end)

        next_priv_s = input_["next_priv_s"].detach()

        #next_priv_s[:,433:783] = 0

       # if self.belief_module.use:

       #     # src is bs x seq_len x 2 x 15
       #     # trg is bs x seq_len x 2 x 7
       #     if self.vdn:
       #         src, trg = self.belief_module.get_samples(input_["next_priv_s"], input_["next_own_hand"], torch.zeros(input_["next_priv_s"].size(1), dtype=torch.int) + input_["next_priv_s"].size(0), device="cuda:1")

       #         leftover = src.size(1)%4
       #         for j in range(src.size(1)//4):
       #             temp = torch.zeros(([2*src.size(0), src.size(1), 15]), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 15
       #             for i in range(2):
       #                 if i == 0:
       #                     temp[:] = src[:, :, i, :].repeat([2, 1, 1])
       #                     temp[0:src.size(0), 4*j+1:, :] = 205
       #                     temp[src.size(0):2*src.size(0), 4*j+3:, :] = 205
       #                     targets = torch.cat((trg[:, 4*j, i, :-1], 
       #                                         trg[:, 4*j+2, i, :-1]), dim=0).detach()
       #                     preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()

       #                     next_priv_s[4*j,:,i,433:563] = preds[0:src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
       #                     next_priv_s[4*j+2,:,i,433:563] = preds[src.size(0):2*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)

       #                 else:
       #                     temp[:] = src[:, :, i, :].repeat([2, 1, 1])
       #                     temp[0:src.size(0), 4*j+2:, :] = 205
       #                     temp[src.size(0):2*src.size(0), 4*j+4:, :] = 205
       #                     targets = torch.cat((trg[:, 4*j+1, i, :-1],
       #                                         trg[:, 4*j+3, i, :-1]), dim=0).detach()
       #                     preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()

       #                     next_priv_s[4*j+1,:,i,433:563] = preds[0:src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
       #                     next_priv_s[4*j+3,:,i,433:563] = preds[src.size(0):2*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)

       #         if leftover:
       #             for i in range(2):
       #                 if i == 0:
       #                     temp = torch.zeros(([(leftover+1)//2*src.size(0), src.size(1), 15]), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 15
       #                     temp[:] = src[:, :, i, :].repeat([(leftover+1)//2, 1, 1])
       #                     for k in range((leftover+1)//2):
       #                         temp[k*src.size(0):(k+1)*src.size(0), src.size(1)-leftover+2*k+1:, :] = 205
       #                     targets = trg[:, src.size(1)-leftover, i, :-1].detach()
       #                     for k in range(2, leftover, 2):
       #                         targets = torch.cat((targets, trg[:, src.size(1)-leftover+k, i, :-1]), dim=0).detach()

       #                     preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()
       #                     count = 0
       #                     for k in range(0, leftover, 2):
       #                         next_priv_s[src.size(1)-leftover+k,:,i,433:563] = preds[count*src.size(0):(count+1)*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
       #                         count += 1
       #                 
       #                 if i == 1:
       #                     if leftover >= 2:
       #                         temp = torch.zeros(([(leftover)//2*src.size(0), src.size(1), 15]), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 15
       #                         temp[:] = src[:, :, i, :].repeat([(leftover)//2, 1, 1])
       #                         for k in range((leftover)//2):
       #                             temp[k*src.size(0):(k+1)*src.size(0), src.size(1)-leftover+2*k+2:, :] = 205
       #                         targets = trg[:, src.size(1)-leftover+1, i, :-1].detach()
       #                         for k in range(2,leftover,2):
       #                             targets = torch.cat((targets, trg[:, src.size(1)-leftover+k, i, :-1]), dim=0).detach()

       #                         preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()
       #                         count = 0
       #                         for k in range(1, leftover, 2):
       #                             next_priv_s[src.size(1)-leftover+k,:,i,433:563] = preds[count*src.size(0):(count+1)*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
       #                             count += 1

            # src is bs x seq_len x 15
            # trg is bs x seq_len x 7 
           # else:
        if input_["next_priv_s"].size(1) == 1:
            next_priv_s = next_priv_s.transpose(0, 1)
            bs = input_["next_priv_s"].size(0)
            seq_len = input_["next_priv_s"].size(1)
            src, _  = self.belief_module.get_samples_one_player(torch.cat([input_["next_priv_s"].transpose(0,1), torch.zeros((79, bs, 838), device="cuda:1" )]), input_["next_own_hand"].transpose(0,1), torch.zeros((bs), dtype=torch.long, device="cuda:1") + seq_len, device="cuda:1")
        else:
            src, _ = self.belief_module.get_samples_one_player(torch.cat([input_["next_priv_s"], torch.zeros((80-input_["next_priv_s"].size(0), input_["next_priv_s"].size(1), 838), device="cuda:1" )]), input_["next_own_hand"], torch.zeros((input_["next_priv_s"].size(1)), dtype=torch.long, device="cuda:1") + input_["next_priv_s"].size(0), device="cuda:1")
       
        assert(torch.all(0 == torch.sum(next_priv_s[:,:,0:125], -1)))

        targets = 26 + torch.zeros((src.size(0), 6), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 6
        j_card_dist = torch.zeros((src.size(0), 28), dtype=torch.long, device="cuda:1").detach()
        temp = torch.zeros((src.size(0)), dtype=torch.long, device="cuda:1").detach()
        for j in range(5):
            while True:
                j_card_dist = F.softmax(self.belief_module(src, targets, None, nopeak_mask)[:,j,:], dim=-1).detach()
                temp = torch.multinomial(j_card_dist, 1)#torch.argmax(torch.log(j_card_dist) + gumbel_dist.sample(sample_shape=j_card_dist.shape).squeeze(-1), axis=1)
                if not torch.any(temp==26) and not torch.any(temp==27):
                    break
            targets[:,j+1] = temp.reshape(src.size(0))
      #      next_priv_s[next_priv_s.size(0)-1, :, 0:125] = 1
      #      next_priv_s[next_priv_s.size(0)-1, :, 25*j:25*(j+1)] = j_card_dist[:, 0:25]
            next_priv_s[:, :, 25*j:25*(j+1)] = j_card_dist[:, 0:25]
      
        #leftover = src.size(1)%4
        #for j in range(src.size(1)//4):
        #    temp = torch.zeros(([4*src.size(0), src.size(1), 15]), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 15
        #    temp[:] = src[:, :, :].repeat([4, 1, 1])
        #    temp[0:src.size(0), 4*j+1:, :] = 205
        #    temp[src.size(0):2*src.size(0), 4*j+2:, :] = 205
        #    temp[2*src.size(0):3*src.size(0), 4*j+3:, :] = 205
        #    temp[3*src.size(0):4*src.size(0), 4*j+4:, :] = 205

        #    targets = torch.cat((trg[:, 4*j, :-1],
        #                            trg[:, 4*j+1, :-1],
        #                            trg[:, 4*j+2, :-1],
        #                            trg[:, 4*j+3, :-1]), dim=0).detach()

        #    preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()
        #    next_priv_s[4*j,:,433:563] = preds[0:src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
        #    next_priv_s[4*j+1,:,433:563] = preds[src.size(0):2*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
        #    next_priv_s[4*j+2,:,433:563] = preds[2*src.size(0):3*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
        #    next_priv_s[4*j+3,:,433:563] = preds[3*src.size(0):4*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)
        #if leftover:
        #    temp = torch.zeros(([leftover*src.size(0), src.size(1), 15]), dtype=torch.long, device="cuda:1").detach() # bs x seq_len x 15
        #    temp[:] = src[:, :, :].repeat([leftover, 1, 1])
        #    for k in range(leftover):
        #        temp[k*src.size(0):(k+1)*src.size(0), src.size(1)-(leftover-k-1):, :] = 205
        #    targets = trg[:, src.size(1)-leftover, :-1].detach()
        #    for k in range(1,leftover):
        #        targets = torch.cat((targets, trg[:, src.size(1)-(leftover-k), :-1]), dim=0).detach()

        #    preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()
        #    for k in range(leftover):
        #        next_priv_s[src.size(1)-(leftover-k),:,433:563] = preds[k*src.size(0):(k+1)*src.size(0), :-1, 0:26].reshape(-1, 5 * 26)

        next_priv_s = next_priv_s.flatten(0, flatten_end)
        next_legal_move = input_["next_legal_move"].flatten(0, flatten_end)
        temperature = input_["temperature"].flatten(0, flatten_end)

        hid = {
            "h0": input_["h0"].flatten(0, 1).transpose(0, 1).contiguous(),
            "c0": input_["c0"].flatten(0, 1).transpose(0, 1).contiguous(),
        }
        next_hid = {
            "h0": input_["next_h0"].flatten(0, 1).transpose(0, 1).contiguous(),
            "c0": input_["next_c0"].flatten(0, 1).transpose(0, 1).contiguous(),
        }
        reward = input_["reward"].flatten(0, 1)
        bootstrap = input_["bootstrap"].flatten(0, 1)

        online_qa = self.online_net(priv_s, legal_move, online_a, hid)[0]
        next_a, _ = self.greedy_act(next_priv_s, next_legal_move, next_hid)
        target_qa, _, _, _ = self.target_net(
            next_priv_s, next_legal_move, next_a, next_hid,
        )

        bsize = obsize * ibsize
        if self.vdn:
            # sum over action & player
            online_qa = online_qa.view(bsize, num_player).sum(1)
            target_qa = target_qa.view(bsize, num_player).sum(1)

        assert reward.size() == bootstrap.size()
        assert reward.size() == target_qa.size()
        target = reward + bootstrap * (self.gamma ** self.multi_step) * target_qa
        priority = (target - online_qa).abs()
        priority = priority.view(obsize, ibsize).detach().cpu()
        return {"priority": priority}

    ############# python only functions #############
    def flat_4d(self, data):
        """
        rnn_hid: [num_layer, batch, num_player, dim] -> [num_player, batch, dim]
        seq_obs: [seq_len, batch, num_player, dim] -> [seq_len, batch, dim]
        """
        bsize = 0
        num_player = 0
        for k, v in data.items():
            if num_player == 0:
                bsize, num_player = v.size()[1:3]

            if v.dim() == 4:
                d0, d1, d2, d3 = v.size()
                data[k] = v.view(d0, d1 * d2, d3)
            elif v.dim() == 3:
                d0, d1, d2 = v.size()
                data[k] = v.view(d0, d1 * d2)
        return bsize, num_player

    def td_error(self, obs, hid, action, reward, terminal, bootstrap, seq_len, stat):
        with torch.no_grad():
            # start_time = time.time()
            max_seq_len = obs["priv_s"].size(0)

            #if self.belief_module.use:

            nopeak_mask = torch.triu(torch.ones((1, 6, 6)), diagonal=1)
            nopeak_mask = (nopeak_mask == 0).to("cuda:0").detach()
                # src is bs x seq_len x 2 x 15
                # trg is bs x seq_len x 2 x 7        
           #     if self.vdn:
           #         src, trg = self.belief_module.get_samples(
           #                             obs["priv_s"].detach(), 
           #                             obs["own_hand"].detach(), 
           #                             seq_len.to(torch.int).detach(), 
           #                             device="cuda:0")
                # src is bs x seq_len x 15
                # trg is bs x seq_len x 7
           #     else:
            src, _ = self.belief_module.get_samples_one_player(obs["priv_s"].detach(),
                                                                    obs["own_hand"].detach(),
                                                                    seq_len.detach(),
                                                                    device="cuda:0")
            src = src.detach()
            #    trg = trg.detach()

            priv_s = obs["priv_s"].detach()
            #if self.vdn:
            #    priv_s[:,:,:,433:783] = 0
            #else:
            #    priv_s[:,:,433:783] = 0

            #if self.belief_module.use:
            #    if self.vdn:
            #        for j in range(src.size(1)//4):
            #            games_considered1 = torch.nonzero(seq_len > 4*j, as_tuple=True)[0].detach()
            #            games_considered2 = torch.nonzero(seq_len > 4*j+1, as_tuple=True)[0].detach()
            #            games_considered3 = torch.nonzero(seq_len > 4*j+2, as_tuple=True)[0].detach()
            #            games_considered4 = torch.nonzero(seq_len > 4*j+3, as_tuple=True)[0].detach()
            #            if games_considered1.numel() + games_considered2.numel() + games_considered3.numel() + games_considered4.numel() == 0: #empty
            #                break

            #            for i in range(2):
            #                if i == 0:
            #                    if games_considered1.numel() + games_considered3.numel():
            #                        temp = torch.zeros(([games_considered1.numel()+games_considered3.numel(), src.size(1), 15]), dtype=torch.long, device="cuda:0").detach() # bs x seq_len x 15
            #                        temp[0:games_considered1.numel()] = src[games_considered1, :, i, :]
            #                        temp[games_considered1.numel():games_considered1.numel()+games_considered3.numel()] = src[games_considered3, :, i, :]
            #                        temp[0:games_considered1.numel(), 4*j+1:, :] = 205
            #                        temp[games_considered1.numel():games_considered1.numel()+games_considered3.numel(), 4*j+3:, :] = 205

            #                        targets = torch.cat((trg[games_considered1, 4*j, i, :-1], 
            #                                            trg[games_considered3, 4*j+2, i, :-1]), dim=0).detach()

            #                        preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()

            #                        priv_s[4*j,games_considered1,i,433:563] = preds[0:games_considered1.numel(), :-1, 0:26].reshape(games_considered1.numel(), 5 * 26)
            #                        priv_s[4*j+2,games_considered3,i,433:563] = preds[games_considered1.numel():games_considered1.numel()+games_considered3.numel(), :-1, 0:26].reshape(games_considered3.numel(), 5 * 26)
            #                    
            #                else:
            #                    if games_considered2.numel() + games_considered4.numel():
            #                        temp = torch.zeros(([games_considered2.numel()+games_considered4.numel(), src.size(1), 15]), dtype=torch.long, device="cuda:0").detach() # bs x seq_len x 15
            #                        temp[0:games_considered2.numel()] = src[games_considered2, :, i, :]
            #                        temp[games_considered2.numel():games_considered2.numel()+games_considered4.numel()] = src[games_considered4, :, i, :]
            #                        temp[0:games_considered2.numel(), 4*j+2:, :] = 205
            #                        temp[games_considered2.numel():games_considered2.numel()+games_considered4.numel(), 4*j+4:, :] = 205
            #        
            #                        targets = torch.cat((trg[games_considered2, 4*j+1, i, :-1],
            #                                                trg[games_considered4, 4*j+3, i, :-1]), dim=0).detach()

            #                        preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()
            #                
            #                        priv_s[4*j+1,games_considered2,i,433:563] = preds[0:games_considered2.numel(), :-1, 0:26].reshape(games_considered2.numel(), 5 * 26)
            #                        priv_s[4*j+3,games_considered4,i,433:563] = preds[games_considered2.numel():games_considered2.numel()+games_considered4.numel(), :-1, 0:26].reshape(games_considered4.numel(), 5 * 26)
            
            targets = 26 + torch.zeros((src.size(0), 6), dtype=torch.long, device="cuda:0").detach() # bs x seq_len x 6
            j_card_dist = torch.zeros((src.size(0), 28), dtype=torch.long, device="cuda:0").detach()
            temp = torch.zeros((src.size(0)), dtype=torch.long, device="cuda:0").detach()

            assert(torch.all(0 == torch.sum(priv_s[:,:,0:125], -1)))

            for j in range(5):
                while True:
                    j_card_dist = F.softmax(self.belief_module(src, targets, None, nopeak_mask)[:,j,:], dim=-1).detach()
                    temp = torch.multinomial(j_card_dist, 1)#torch.argmax(torch.log(j_card_dist) + gumbel_dist.sample(sample_shape=j_card_dist.shape).squeeze(-1), axis=1)
                    if not torch.any(temp==26) and not torch.any(temp==27):
                        break
                targets[:,j+1] = temp.reshape(src.size(0))
         #       priv_s[:, :, 0:125] = 1
                for i in range(src.size(0)):
                    priv_s[0:seq_len[j], j, 25*j:25*(j+1)] = j_card_dist[j, 0:25]
         #       priv_s[:, :, 25*j:25*(j+1)] = j_card_dist[:, 0:25]
            
            #        for j in range(src.size(1)//4):
            #            games_considered1 = torch.nonzero(seq_len > 4*j, as_tuple=True)[0].detach()
            #            games_considered2 = torch.nonzero(seq_len > 4*j+1, as_tuple=True)[0].detach()
            #            games_considered3 = torch.nonzero(seq_len > 4*j+2, as_tuple=True)[0].detach()
            #            games_considered4 = torch.nonzero(seq_len > 4*j+3, as_tuple=True)[0].detach()
            #            if games_considered1.numel() + games_considered2.numel() + games_considered3.numel() + games_considered4.numel() == 0: #empty
            #                break
            #            temp = 205 + torch.zeros(([games_considered1.numel()+games_considered2.numel()+games_considered3.numel()+games_considered4.numel(), src.size(1), 15]), dtype=torch.long, device="cuda:0").detach() # bs x seq_len x 15
            #            temp[0:games_considered1.numel()] = src[games_considered1, :, :]
            #            temp[games_considered1.numel():games_considered1.numel()+games_considered2.numel()] = src[games_considered2, :, :]
            #            temp[games_considered1.numel()+games_considered2.numel():games_considered1.numel()+games_considered2.numel()+games_considered3.numel()] = src[games_considered3, :, :]
            #            temp[games_considered1.numel()+games_considered2.numel()+games_considered3.numel():games_considered1.numel()+games_considered2.numel()+games_considered3.numel()+games_considered4.numel()] = src[games_considered4, :, :]

            #          #  temp[0:games_considered1.numel(), 4*j+1:, :] = 205
            #          #  temp[games_considered1.numel():games_considered1.numel()+games_considered2.numel(), 4*j+2:, :] = 205
            #          #  temp[games_considered1.numel()+games_considered2.numel():games_considered1.numel()+games_considered2.numel()+games_considered3.numel(), 4*j+3:, :] = 205
            #          #  temp[games_considered1.numel()+games_considered2.numel()+games_considered3.numel():games_considered1.numel()+games_considered2.numel()+games_considered3.numel()+games_considered4.numel(), 4*j+4:, :] = 205

            #            targets = torch.cat((trg[games_considered1, 4*j, :-1],
            #                                 trg[games_considered2, 4*j+1, :-1],
            #                                 trg[games_considered3, 4*j+2, :-1],
            #                                 trg[games_considered4, 4*j+3, :-1]), dim=0).detach()

            #            preds = F.softmax(self.belief_module(temp, targets.long(), None, nopeak_mask), dim=-1).detach()

            #            priv_s[4*j,games_considered1,433:563] = preds[0:games_considered1.numel(), :-1, 0:26].reshape(games_considered1.numel(), 5 * 26)
            #            priv_s[4*j+1,games_considered2,433:563] = preds[games_considered1.numel():games_considered1.numel()+games_considered2.numel(), :-1, 0:26].reshape(games_considered2.numel(), 5 * 26)

            #            priv_s[4*j+2,games_considered3,433:563] = preds[games_considered1.numel()+games_considered2.numel():games_considered1.numel()+games_considered2.numel()+games_considered3.numel(), :-1, 0:26].reshape(games_considered3.numel(), 5 * 26)

            #            priv_s[4*j+3,games_considered4,433:563] = preds[games_considered1.numel()+games_considered2.numel()+games_considered3.numel():games_considered1.numel()+games_considered2.numel()+games_considered3.numel()+games_considered4.numel(), :-1, 0:26].reshape(games_considered4.numel(), 5 * 26)

             #   del temp
             #   del targets
             #   del j_card_dist
             #   del src
             #   del targets
             #   del nopeak_mask

            bsize, num_player = 0, 1
            if self.vdn:
                bsize, num_player = self.flat_4d(obs)
                self.flat_4d(action)
                priv_s = priv_s.reshape(priv_s.size(0), 2*priv_s.size(1), -1).detach()

            legal_move = obs["legal_move"].detach()
            action = action["a"].detach()

        hid = {}

        # this only works because the trajectories are padded,
        # i.e. no terminal in the 
        online_qa, greedy_a, _, lstm_o = self.online_net(
            priv_s, legal_move, action, hid
        )

        with torch.no_grad():
            target_qa, _, _, _ = self.target_net(priv_s, legal_move, greedy_a, hid)
            # assert target_q.size() == pa.size()
            # target_qe = (pa * target_q).sum(-1).detach()
            assert online_qa.size() == target_qa.size()

        if self.vdn:
            online_qa = online_qa.view(max_seq_len, bsize, num_player).sum(-1)
            target_qa = target_qa.view(max_seq_len, bsize, num_player).sum(-1)
            lstm_o = lstm_o.view(max_seq_len, bsize, num_player, -1)

        terminal = terminal.float()
        bootstrap = bootstrap.float()

        errs = []
        target_qa = torch.cat(
            [target_qa[self.multi_step :], target_qa[: self.multi_step]], 0
        )
        target_qa[-self.multi_step :] = 0

        assert target_qa.size() == reward.size()
        target = reward + bootstrap * (self.gamma ** self.multi_step) * target_qa
        mask = torch.arange(0, max_seq_len, device=seq_len.device)
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
        err = (target.detach() - online_qa) * mask
        # print("td time: " + str(time.time()-start_time))
        return err, lstm_o#, belief_losses

    def aux_task_iql(self, lstm_o, hand, seq_len, rl_loss_size, stat):
        seq_size, bsize, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, self.online_net.hand_size, 3)
        own_hand_slot_mask = own_hand.sum(3)
        pred_loss1, avg_xent1, _, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size

        stat["aux1"].feed(avg_xent1)
        return pred_loss1

    def aux_task_vdn(self, lstm_o, hand, t, seq_len, rl_loss_size, stat):
        """1st and 2nd order aux task used in VDN"""
        seq_size, bsize, num_player, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, num_player, self.online_net.hand_size, 3)
        own_hand_slot_mask = own_hand.sum(4)
        pred_loss1, avg_xent1, belief1, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size

        rotate = [num_player - 1]
        rotate.extend(list(range(num_player - 1)))
        partner_hand = own_hand[:, :, rotate, :, :]
        partner_hand_slot_mask = partner_hand.sum(4)
        partner_belief1 = belief1[:, :, rotate, :, :].detach()

        stat["aux1"].feed(avg_xent1)
        return pred_loss1

    def loss(self, batch, pred_weight, stat):
        err, lstm_o = self.td_error(
            batch.obs,
            batch.h0,
            batch.action,
            batch.reward,
            batch.terminal,
            batch.bootstrap,
            batch.seq_len.to(torch.long),
            stat,
        )
        rl_loss = nn.functional.smooth_l1_loss(
            err, torch.zeros_like(err), reduction="none"
        )
        rl_loss = rl_loss.sum(0)
        stat["rl_loss"].feed((rl_loss / batch.seq_len).mean().item())

        priority = err.abs()
        # priority = self.aggregate_priority(p, batch.seq_len)

        if pred_weight > 0:
            if self.vdn:
                pred_loss1 = self.aux_task_vdn(
                    lstm_o,
                    batch.obs["own_hand"],
                    batch.obs["temperature"],
                    batch.seq_len,
                    rl_loss.size(),
                    stat,
                )
                loss = rl_loss + pred_weight * pred_loss1
            else:
                pred_loss = self.aux_task_iql(
                    lstm_o, batch.obs["own_hand"], batch.seq_len, rl_loss.size(), stat,
                )
                loss = rl_loss + pred_weight * pred_loss
        else:
            loss = rl_loss
        return loss, priority#, belief_losses
