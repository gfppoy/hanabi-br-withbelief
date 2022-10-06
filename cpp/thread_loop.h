// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "rela/r2d2_actor.h"
#include "rela/thread_loop.h"

using HanabiVecEnv = rela::VectorEnv<HanabiEnv>;

using namespace torch::indexing;

class HanabiThreadLoop : public rela::ThreadLoop {
 public:
  HanabiThreadLoop(
      std::shared_ptr<rela::R2D2Actor> actor,
      std::shared_ptr<HanabiVecEnv> vecEnv,
      bool eval)
      : actors_({std::move(actor)})
      , vecEnv_(std::move(vecEnv))
      , eval_(eval) {
    assert(actors_.size() >= 1);
    if (eval_) {
      assert(vecEnv_->size() == 1);
    }
  }

  HanabiThreadLoop(
      std::vector<std::shared_ptr<rela::R2D2Actor>> actors,
      std::shared_ptr<HanabiVecEnv> vecEnv,
      bool eval)
      : actors_(std::move(actors))
      , vecEnv_(std::move(vecEnv))
      , eval_(eval) {
    assert(actors_.size() >= 1);
    if (eval_) {
      assert(vecEnv_->size() == 1);
    }
  }

  void mainLoop() final {
    rela::TensorDict obs = {};
    std::vector<int> reset_envs;
    torch::Tensor r;
    torch::Tensor t;
    torch::Tensor aoh1 = torch::ones({80, (int)vecEnv_->size(), 838});
    torch::Tensor aoh2 = torch::ones({80, (int)vecEnv_->size(), 838});
    torch::Tensor counts = torch::zeros({(int)vecEnv_->size()});
    while (!terminated()) {
      std::tie(obs, reset_envs) = vecEnv_->reset(obs);
      for (int i = 0; i < (int)reset_envs.size(); ++i) {
         counts[reset_envs[i]] = 1;
         aoh1.index({Slice(), reset_envs[i], Slice()}) = 0;
         aoh2.index({Slice(), reset_envs[i], Slice()}) = 0;
       }
      while (!vecEnv_->anyTerminated()) {
        if (terminated()) {
          break;
        }

        if (paused()) {
          waitUntilResume();
        }

        rela::TensorDict reply;
        if (actors_.size() == 1) {
          reply = actors_[0]->act(obs);
        } else {
          std::vector<rela::TensorDict> replyVec;
          for (int i = 0; i < (int)actors_.size(); ++i) {
            auto input = rela::tensor_dict::narrow(obs, 1, i, 1, true);
            for (int j = 0; j < (int)vecEnv_->size(); ++j) {
              if (i == 0) {
                if (counts[j].item<int>()-1 < 80) {
                  aoh1.index({counts[j].item<int>()-1, j, Slice()}) = input["priv_s"].index({j, Slice()});
                }
              }
              else {
                if (counts[j].item<int>()-1 < 80) {
                  aoh2.index({counts[j].item<int>()-1, j, Slice()}) = input["priv_s"].index({j, Slice()});
                }
              }
            }
            if (i == 0) {
              input["aoh"] = aoh1;
            }
            else {
              input["aoh"] = aoh2;
            }
            input["seq_len"] = counts;
            // if (!logFile_.empty()) {
            //   logState(*file, input);
            // }
            auto rep = actors_[i]->act(input);
            replyVec.push_back(rep);
          }
          for (int i = 0; i < (int)vecEnv_->size(); ++i) {
            counts[i] += 1;
          }
          reply = rela::tensor_dict::stack(replyVec, 1);
        }
        std::tie(obs, r, t) = vecEnv_->step(reply);

        if (eval_) {
          continue;
        }

        for (int i = 0; i < (int)actors_.size(); ++i) {
          actors_[i]->postAct(r, t);
        }
      }

      // eval only runs for one game
      if (eval_) {
        break;
      }
    }
  }

 private:
  std::vector<std::shared_ptr<rela::R2D2Actor>> actors_;
  std::shared_ptr<HanabiVecEnv> vecEnv_;
  const bool eval_;
};
