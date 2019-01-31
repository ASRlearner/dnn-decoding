// online/online-faster-decoder.cc

// Copyright 2012 Cisco Systems (author: Matthias Paulik)

//   Modifications to the original contribution by Cisco Systems made by:
//   Vassil Panayotov

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/timer.h"
#include "online-faster-decoder.h"
#include "fstext/fstext-utils.h"
#include "hmm/hmm-utils.h"

namespace kaldi {

void OnlineFasterDecoder::ResetDecoder(bool full) {
  ClearToks(toks_.Clear());
  StateId start_state = fst_.Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  Arc dummy_arc(0, 0, Weight::One(), start_state);
  Token *dummy_token = new Token(dummy_arc, NULL);
  toks_.Insert(start_state, dummy_token);
  prev_immortal_tok_ = immortal_tok_ = dummy_token;
  utt_frames_ = 0;
  if (full)
    frame_ = 0;
}


//由初始token和最终token回溯得到词图
void
OnlineFasterDecoder::MakeLattice(const Token *start,
                                 const Token *end,
                                 fst::MutableFst<LatticeArc> *out_fst) const {
  //删除所有的状态
  out_fst->DeleteStates();
  if (start == NULL) return;
  bool is_final = false;
  //这个代价是最后一个token的代价
  double this_cost = start->cost_ + fst_.Final(start->arc_.nextstate).Value();
  if (this_cost != std::numeric_limits<double>::infinity())
    is_final = true;
  std::vector<LatticeArc> arcs_reverse;  // 存储逆序的弧的向量
  //回溯得到解码图的路径结点
  for (const Token *tok = start; tok != end; tok = tok->prev_) {
    //得到总代价 语言模型代价 声学代价
    BaseFloat tot_cost = tok->cost_ -
        (tok->prev_ ? tok->prev_->cost_ : 0.0),
        graph_cost = tok->arc_.weight.Value(),
        ac_cost = tot_cost - graph_cost;
    //初始化词图弧
    LatticeArc l_arc(tok->arc_.ilabel,
                     tok->arc_.olabel,
                     LatticeWeight(graph_cost, ac_cost),
                     tok->arc_.nextstate);
    //将弧边存入
    arcs_reverse.push_back(l_arc);
  }
  //如果首条边的下一个状态等于解码图的初始状态
  if(arcs_reverse.back().nextstate == fst_.Start()) {
    //删除第一条边
    arcs_reverse.pop_back();  // that was a "fake" token... gives no info.
  }
  StateId cur_state = out_fst->AddState();
  //设定初始状态
  out_fst->SetStart(cur_state);
  //开始从头遍历弧边 (从尾到头访问向量中的元素)
  for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
    //得到一条弧边
    LatticeArc arc = arcs_reverse[i];
    //添加一个状态并返回其id
    arc.nextstate = out_fst->AddState();
    //给某个状态id添加弧边
    out_fst->AddArc(cur_state, arc);
    cur_state = arc.nextstate;
  }
  if (is_final) {
    Weight final_weight = fst_.Final(start->arc_.nextstate);
    out_fst->SetFinal(cur_state, LatticeWeight(final_weight.Value(), 0.0));
  } else {
    out_fst->SetFinal(cur_state, LatticeWeight::One());
  }
  //移除词图中的epsilon？？
  RemoveEpsLocal(out_fst);
}

///更新所有活跃结点的祖先
void OnlineFasterDecoder::UpdateImmortalToken() {
  unordered_set<Token*> emitting;
  ///遍历哈希表中的元素
  for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
    Token* tok = e->val;
    while (tok->arc_.ilabel == 0) //deal with non-emitting ones ...
      tok = tok->prev_;
    if (tok != NULL)
      emitting.insert(tok);
  }
  Token* the_one = NULL;
  while (1) {
    if (emitting.size() == 1) {
      the_one = *(emitting.begin());
      break;
    }
    if (emitting.size() == 0)
      break;
    unordered_set<Token*> prev_emitting;
    unordered_set<Token*>::iterator it;
    for (it = emitting.begin(); it != emitting.end(); ++it) {
      Token* tok = *it;
      Token* prev_token = tok->prev_;
      while ((prev_token != NULL) && (prev_token->arc_.ilabel == 0))
        prev_token = prev_token->prev_; //deal with non-emitting ones
      if (prev_token == NULL)
        continue;
      prev_emitting.insert(prev_token);
    } // for
    emitting = prev_emitting;
  } // while
  if (the_one != NULL) {
    prev_immortal_tok_ = immortal_tok_;
    immortal_tok_ = the_one;
    return;
  }
}


//返回部分解码的词图
bool
OnlineFasterDecoder::PartialTraceback(fst::MutableFst<LatticeArc> *out_fst) {
  //更新祖先token
  UpdateImmortalToken();
  //如果祖先未发生改变则认为没有部分回溯结果
  if(immortal_tok_ == prev_immortal_tok_)
    return false; //no partial traceback at that point of time
    //否则 根据前一时刻的祖先和当前时刻的祖先token生成词图
  MakeLattice(immortal_tok_, prev_immortal_tok_, out_fst);
  return true;
}

//认为一句话说完了
void
OnlineFasterDecoder::FinishTraceBack(fst::MutableFst<LatticeArc> *out_fst) {
  Token *best_tok = NULL;
  bool is_final = ReachedFinal();
  if (!is_final) {//如果不是最后一帧的最后一个状态
    for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail)
      //找到权重最小的token作为best_tok
      if (best_tok == NULL || *best_tok < *(e->val) )
        best_tok = e->val;
  } else {
    double best_cost = std::numeric_limits<double>::infinity();
    //遍历找到cost的最小值 即权重最小的token作为best_token
    for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
      //得到当前token的权重
      double this_cost = e->val->cost_ + fst_.Final(e->key).Value();
      if (this_cost != std::numeric_limits<double>::infinity() &&
          this_cost < best_cost) {
        best_cost = this_cost;
        best_tok = e->val;
      }
    }
  }
  //完成回溯 生成词图
  //immortal_tok是所有活跃token的祖先 由这个best_tok和祖先祖先得到一条路径 再生成词图
  MakeLattice(best_tok, immortal_tok_, out_fst);
}


void
OnlineFasterDecoder::TracebackNFrames(int32 nframes,
                                      fst::MutableFst<LatticeArc> *out_fst) {
  Token *best_tok = NULL;
  for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail)
    if (best_tok == NULL || *best_tok < *(e->val) )
      best_tok = e->val;
  if (best_tok == NULL) {
    out_fst->DeleteStates();
    return;
  }

  bool is_final = false;
  double this_cost = best_tok->cost_ +
      fst_.Final(best_tok->arc_.nextstate).Value();
                             
  if (this_cost != std::numeric_limits<double>::infinity())
    is_final = true;
  std::vector<LatticeArc> arcs_reverse;  // arcs in reverse order.
  for (Token *tok = best_tok; (tok != NULL) && (nframes > 0); tok = tok->prev_) {
    if (tok->arc_.ilabel != 0) // count only the non-epsilon arcs
      --nframes;
    BaseFloat tot_cost = tok->cost_ -
        (tok->prev_ ? tok->prev_->cost_ : 0.0);
    BaseFloat graph_cost = tok->arc_.weight.Value();
    BaseFloat ac_cost = tot_cost - graph_cost;
    LatticeArc larc(tok->arc_.ilabel,
                     tok->arc_.olabel,
                     LatticeWeight(graph_cost, ac_cost),
                     tok->arc_.nextstate);
    arcs_reverse.push_back(larc);
  }
  if(arcs_reverse.back().nextstate == fst_.Start())
    arcs_reverse.pop_back();  // that was a "fake" token... gives no info.
  StateId cur_state = out_fst->AddState();
  out_fst->SetStart(cur_state);
  for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
    LatticeArc arc = arcs_reverse[i];
    arc.nextstate = out_fst->AddState();
    out_fst->AddArc(cur_state, arc);
    cur_state = arc.nextstate;
  }
  if (is_final) {
    Weight final_weight = fst_.Final(best_tok->arc_.nextstate);
    out_fst->SetFinal(cur_state, LatticeWeight(final_weight.Value(), 0.0));
  } else {
    out_fst->SetFinal(cur_state, LatticeWeight::One());
  }
  RemoveEpsLocal(out_fst);
}


bool OnlineFasterDecoder::EndOfUtterance() {
  fst::VectorFst<LatticeArc> trace;
  int32 sil_frm = opts_.inter_utt_sil / (1 + utt_frames_ / opts_.max_utt_len_);
  TracebackNFrames(sil_frm, &trace);
  std::vector<int32> isymbols;
  fst::GetLinearSymbolSequence(trace, &isymbols,
                               static_cast<std::vector<int32>* >(0),
                               static_cast<LatticeArc::Weight*>(0));
  std::vector<std::vector<int32> > split;
  SplitToPhones(trans_model_, isymbols, &split);
  for (size_t i = 0; i < split.size(); i++) {
    int32 tid = split[i][0];
    int32 phone = trans_model_.TransitionIdToPhone(tid);
    if (silence_set_.count(phone) == 0)
      return false;
  }
  return true;
}

///解码并获取解码状态
OnlineFasterDecoder::DecodeState
OnlineFasterDecoder::Decode(DecodableInterface *decodable) {
  ///判断是否是新的语音
  if (state_ == kEndFeats || state_ == kEndUtt) // new utterance
    ///复位解码器状态
    ResetDecoder(state_ == kEndFeats);
  ///处理非发射状态 初始状态下不会剪枝
  ProcessNonemitting(std::numeric_limits<float>::max());
  int32 batch_frame = 0;
  Timer timer;
  double64 tstart = timer.Elapsed(), tstart_batch = tstart;
  BaseFloat factor = -1;
  ///如果不是最后一帧并且帧个数小于上限
  for (; !decodable->IsLastFrame(frame_ - 1) && batch_frame < opts_.batch_size;
       ++frame_, ++utt_frames_, ++batch_frame) {
    if (batch_frame != 0 && (batch_frame % opts_.update_interval) == 0) {
      /// 如果需要的话调整剪枝宽度
      BaseFloat tend = timer.Elapsed();
      BaseFloat elapsed = (tend - tstart) * 1000;
      /// warning: hardcoded 10ms frames assumption!
      factor = elapsed / (opts_.rt_max * opts_.update_interval * 10);
      BaseFloat min_factor = (opts_.rt_min / opts_.rt_max);
      if (factor > 1 || factor < min_factor) {
        BaseFloat update_factor = (factor > 1)?
            -std::min(opts_.beam_update * factor, opts_.max_beam_update):
             std::min(opts_.beam_update / factor, opts_.max_beam_update);
        effective_beam_ += effective_beam_ * update_factor;
        effective_beam_ = std::min(effective_beam_, max_beam_);
      }
      tstart = tend;
    }
    if (batch_frame != 0 && (frame_ % 200) == 0)
      // one log message at every 2 seconds assuming 10ms frames
      KALDI_VLOG(3) << "Beam: " << effective_beam_
          << "; Speed: "
          << ((timer.Elapsed() - tstart_batch) * 1000) / (batch_frame*10)
          << " xRT";
    ///获取中断似然度
    BaseFloat weight_cutoff = ProcessEmitting(decodable);
    ///处理非发射状态
    ProcessNonemitting(weight_cutoff);
  }
  ///如果帧数达到了上限并且还未到达最后一帧
  if (batch_frame == opts_.batch_size && !decodable->IsLastFrame(frame_ - 1)) {
    ///如果语音结束则设置相应的状态
    if (EndOfUtterance())
      state_ = kEndUtt;
    else
      state_ = kEndBatch;
  } else {
    state_ = kEndFeats;
  }
  return state_;
}

} // namespace kaldi
