// decoder/faster-decoder.cc

// Copyright 2009-2011 Microsoft Corporation
//           2012-2013 Johns Hopkins University (author: Daniel Povey)

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

#include "decoder/faster-decoder.h"

namespace kaldi {


FasterDecoder::FasterDecoder(const fst::Fst<fst::StdArc> &fst,
                             const FasterDecoderOptions &opts):
    fst_(fst), config_(opts), num_frames_decoded_(-1) {
  KALDI_ASSERT(config_.hash_ratio >= 1.0);  // less doesn't make much sense.
  KALDI_ASSERT(config_.max_active > 1);
  KALDI_ASSERT(config_.min_active >= 0 && config_.min_active < config_.max_active);
  toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
}


///初始化解码
void FasterDecoder::InitDecoding() {
  // clean up from last time:
  ClearToks(toks_.Clear());
  StateId start_state = fst_.Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  Arc dummy_arc(0, 0, Weight::One(), start_state);
  toks_.Insert(start_state, new Token(dummy_arc, NULL));
  ProcessNonemitting(std::numeric_limits<float>::max());
  num_frames_decoded_ = 0;
}


void FasterDecoder::Decode(DecodableInterface *decodable) {
  InitDecoding();
  while (!decodable->IsLastFrame(num_frames_decoded_ - 1)) {
    double weight_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(weight_cutoff);
  }
}

void FasterDecoder::AdvanceDecoding(DecodableInterface *decodable,
                                      int32 max_num_frames) {
  KALDI_ASSERT(num_frames_decoded_ >= 0 &&
               "You must call InitDecoding() before AdvanceDecoding()");
  int32 num_frames_ready = decodable->NumFramesReady();
  // num_frames_ready must be >= num_frames_decoded, or else
  // the number of frames ready must have decreased (which doesn't
  // make sense) or the decodable object changed between calls
  // (which isn't allowed).
  KALDI_ASSERT(num_frames_ready >= num_frames_decoded_);
  int32 target_frames_decoded = num_frames_ready;
  if (max_num_frames >= 0)
    target_frames_decoded = std::min(target_frames_decoded,
                                     num_frames_decoded_ + max_num_frames);
  while (num_frames_decoded_ < target_frames_decoded) {
    // note: ProcessEmitting() increments num_frames_decoded_
    double weight_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(weight_cutoff); 
  }    
}

//最后一帧的最后一个状态是否活跃
bool FasterDecoder::ReachedFinal() {
  for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
    if (e->val->cost_ != std::numeric_limits<double>::infinity() &&
        fst_.Final(e->key) != Weight::Zero())
      return true;
  }
  return false;
}

bool FasterDecoder::GetBestPath(fst::MutableFst<LatticeArc> *fst_out,
                                bool use_final_probs) {
  // GetBestPath gets the decoding output.  If "use_final_probs" is true
  // AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking into
  // account final-probs.  fst_out will be empty (Start() == kNoStateId) if
  // nothing was available.  It returns true if it got output (thus, fst_out
  // will be nonempty).
  fst_out->DeleteStates();
  Token *best_tok = NULL;
  bool is_final = ReachedFinal();
  if (!is_final) {
    for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail)
      if (best_tok == NULL || *best_tok < *(e->val) )
        best_tok = e->val;
  } else {
    double infinity =  std::numeric_limits<double>::infinity(),
        best_cost = infinity;
    for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
      double this_cost = e->val->cost_ + fst_.Final(e->key).Value();
      if (this_cost < best_cost && this_cost != infinity) {
        best_cost = this_cost;
        best_tok = e->val;
      }
    }
  }
  if (best_tok == NULL) return false;  // No output.

  std::vector<LatticeArc> arcs_reverse;  // arcs in reverse order.

  for (Token *tok = best_tok; tok != NULL; tok = tok->prev_) {
    BaseFloat tot_cost = tok->cost_ -
        (tok->prev_ ? tok->prev_->cost_ : 0.0),
        graph_cost = tok->arc_.weight.Value(),
        ac_cost = tot_cost - graph_cost;
    LatticeArc l_arc(tok->arc_.ilabel,
                     tok->arc_.olabel,
                     LatticeWeight(graph_cost, ac_cost),
                     tok->arc_.nextstate);
    arcs_reverse.push_back(l_arc);
  }
  KALDI_ASSERT(arcs_reverse.back().nextstate == fst_.Start());
  arcs_reverse.pop_back();  // that was a "fake" token... gives no info.

  StateId cur_state = fst_out->AddState();
  fst_out->SetStart(cur_state);
  for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
    LatticeArc arc = arcs_reverse[i];
    arc.nextstate = fst_out->AddState();
    fst_out->AddArc(cur_state, arc);
    cur_state = arc.nextstate;
  }
  if (is_final && use_final_probs) {
    Weight final_weight = fst_.Final(best_tok->arc_.nextstate);
    fst_out->SetFinal(cur_state, LatticeWeight(final_weight.Value(), 0.0));
  } else {
    fst_out->SetFinal(cur_state, LatticeWeight::One());
  }
  RemoveEpsLocal(fst_out);
  return true;
}


///这部分和词图快速在线解码器部分相同
///获取权重截断值 同时计算活跃token数
double FasterDecoder::GetCutoff(Elem *list_head, size_t *tok_count,
                                BaseFloat *adaptive_beam, Elem **best_elem) {
  double best_cost = std::numeric_limits<double>::infinity();
  size_t count = 0;
  if (config_.max_active == std::numeric_limits<int32>::max() &&
      config_.min_active == 0) {
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      double w = e->val->cost_;
      if (w < best_cost) {
        best_cost = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;
    if (adaptive_beam != NULL) *adaptive_beam = config_.beam;
    return best_cost + config_.beam;
  } else {
    tmp_array_.clear();
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      double w = e->val->cost_;
      tmp_array_.push_back(w);
      if (w < best_cost) {
        best_cost = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;
    double beam_cutoff = best_cost + config_.beam,
        min_active_cutoff = std::numeric_limits<double>::infinity(),
        max_active_cutoff = std::numeric_limits<double>::infinity();
    
    if (tmp_array_.size() > static_cast<size_t>(config_.max_active)) {
      std::nth_element(tmp_array_.begin(),
                       tmp_array_.begin() + config_.max_active,
                       tmp_array_.end());
      max_active_cutoff = tmp_array_[config_.max_active];
    }
    if (max_active_cutoff < beam_cutoff) { // max_active is tighter than beam.
      if (adaptive_beam)
        *adaptive_beam = max_active_cutoff - best_cost + config_.beam_delta;
      return max_active_cutoff;
    }    
    if (tmp_array_.size() > static_cast<size_t>(config_.min_active)) {
      if (config_.min_active == 0) min_active_cutoff = best_cost;
      else {
        std::nth_element(tmp_array_.begin(),
                         tmp_array_.begin() + config_.min_active,
                         tmp_array_.size() > static_cast<size_t>(config_.max_active) ?
                         tmp_array_.begin() + config_.max_active :
                         tmp_array_.end());
        min_active_cutoff = tmp_array_[config_.min_active];
      }
    }
    if (min_active_cutoff > beam_cutoff) { // min_active is looser than beam.
      if (adaptive_beam)
        *adaptive_beam = min_active_cutoff - best_cost + config_.beam_delta;
      return min_active_cutoff;
    } else {
      *adaptive_beam = config_.beam;
      return beam_cutoff;
    }
  }
}

void FasterDecoder::PossiblyResizeHash(size_t num_toks) {
  size_t new_sz = static_cast<size_t>(static_cast<BaseFloat>(num_toks)
                                      * config_.hash_ratio);
  if (new_sz > toks_.Size()) {
    toks_.SetSize(new_sz);
  }
}

// ProcessEmitting returns the likelihood cutoff used.
///处理发射态 返回剪枝使用的似然度
double FasterDecoder::ProcessEmitting(DecodableInterface *decodable) {
  int32 frame = num_frames_decoded_;
  ///获取哈希表的表头元素
  Elem *last_toks = toks_.Clear();
  size_t tok_cnt;
  BaseFloat adaptive_beam;
  Elem *best_elem = NULL;
  double weight_cutoff = GetCutoff(last_toks, &tok_cnt,
                                   &adaptive_beam, &best_elem);
  KALDI_VLOG(3) << tok_cnt << " tokens active.";
  PossiblyResizeHash(tok_cnt);  /// 确保当前的哈希表足够大

  ///这是我们添加了对数似然度(例如对于下一帧)以后的剪枝阈值。
  ///这是我们在下一帧将会使用的阈值范围
  double next_weight_cutoff = std::numeric_limits<double>::infinity();
  

  ///首先处理最佳token以获得对于下一帧剪枝阈值的一个合理的
  ///取值
  if (best_elem) {
    StateId state = best_elem->key;
    Token *tok = best_elem->val;
    ///遍历解码图的弧
    for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) {  // we'd propagate..
        ///得到该帧的声学代价 即其对数似然度
        ///弧输入标签id作为转移id 由转移模型得到转移id对应的pdf-id
        /// 再由pdf-id 当前特征序号计算似然度
        BaseFloat ac_cost = - decodable->LogLikelihood(frame, arc.ilabel);
        ///计算总的权重=弧边权重+tok代价+声学代价
        double new_weight = arc.weight.Value() + tok->cost_ + ac_cost;
        ///如果新的权重值+容错阈值<二次剪枝阈值
        ///则更新二次剪枝阈值
        if (new_weight + adaptive_beam < next_weight_cutoff)
          next_weight_cutoff = new_weight + adaptive_beam;
      }
    }
  }

  // int32 n = 0, np = 0;

  // the tokens are now owned here, in last_toks, and the hash is empty.
  // 'owned' is a complex thing here; the point is we need to call TokenDelete
  // on each elem 'e' to let toks_ know we're done with them.
  ///从哈希表的表头开始遍历哈系表
  for (Elem *e = last_toks, *e_tail; e != NULL; e = e_tail) {  // loop this way
    // n++;
    // because we delete "e" as we go.
    StateId state = e->key;
    Token *tok = e->val;
    ///如果当前token的代价小于第一次剪枝阈值 则保留
    if (tok->cost_ < weight_cutoff) {  // not pruned.
      // np++;
      KALDI_ASSERT(state == tok->arc_.nextstate);
      ///遍历该token对应的状态id相连的所有弧边
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
           !aiter.Done();
           aiter.Next()) {
        Arc arc = aiter.Value();
        ///如果弧边的输入不为空 即为发射态 则继续传递下去
        if (arc.ilabel != 0) {  // propagate..
          BaseFloat ac_cost =  - decodable->LogLikelihood(frame, arc.ilabel);
          double new_weight = arc.weight.Value() + tok->cost_ + ac_cost;
          ///如果下一时刻token的总代价小于第二次剪枝阈值 则保留
          if (new_weight < next_weight_cutoff) {
              ///新建一个token结点 并寻找该状态id对应的元素是否已存在
            Token *new_tok = new Token(arc, ac_cost, tok);
            Elem *e_found = toks_.Find(arc.nextstate);
            ///如果下一时刻token处总代价+容错值<二次剪枝阈值 则更新该阈值
            if (new_weight + adaptive_beam < next_weight_cutoff)
              next_weight_cutoff = new_weight + adaptive_beam;
            if (e_found == NULL) {///如果该状态id还未出现在哈希表中 则将该状态id对应的元素插入哈希表中
              toks_.Insert(arc.nextstate, new_tok);
            } else {///如果该状态id对应的元素已经存在
              if ( *(e_found->val) < *new_tok ) {///如果原来的token处总代价大于新的token处代价 则替换原来的token
                Token::TokenDelete(e_found->val);
                e_found->val = new_tok;
              } else {///否则删除该新token
                Token::TokenDelete(new_tok);
              }
            }
          }
        }
      }
    }
    ///访问下一个元素
    e_tail = e->tail;
    ///释放当前元素中token所占的内存空间
    Token::TokenDelete(e->val);
    ///删除当前元素
    toks_.Delete(e);
  }
  ///认为解码的帧数递增
  num_frames_decoded_++;
  ///返回得到的二次剪枝阈值
  return next_weight_cutoff;
}

// TODO: first time we go through this, could avoid using the queue.
///处理输入为epsilon的弧边 cutoff初始为无穷大 否则由处理发射态得到
void FasterDecoder::ProcessNonemitting(double cutoff) {
  /// 处理帧的非发射状态弧
  KALDI_ASSERT(queue_.empty());
  ///toks_是当前时刻的哈希表
  ///遍历当前的哈希表
  for (const Elem *e = toks_.GetList(); e != NULL;  e = e->tail)
    ///将哈希表中的状态id依次存入向量中
    queue_.push_back(e->key);

  ///只要向量不为空循环
  ///实际顺序为从后往前
  while (!queue_.empty()) {
      ///从向量中将状态id取出
    StateId state = queue_.back();
    queue_.pop_back();
    ///得到该状态id对应的token
    Token *tok = toks_.Find(state)->val;  // would segfault if state not
    // in toks_ but this can't happen.
    ///如果当前token处的代价小于等于剪枝阈值 则保留
    if (tok->cost_ > cutoff) { // Don't bother processing successors.
      continue;
    }
    ///这里token的定义和快速在线解码器中不同 弧连接的是前一个token
    KALDI_ASSERT(tok != NULL && state == tok->arc_.nextstate);
    ///如果当前token的代价小于等于截断值
    ///访问该状态id由解码图约束的弧边
    for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      ///只处理非发射态
      ///如果是输入为epsilon的弧
      if (arc.ilabel == 0) {
          ///由当前token和从当前token出发的弧生成下一个token
        Token *new_tok = new Token(arc, tok);
        if (new_tok->cost_ > cutoff) {  ///如果下一个token的代价大于截断值 则剪枝
          Token::TokenDelete(new_tok);
        } else {///如果其代价小于等于截断值 则保留
          ///在哈希表中寻找弧边下一个状态对应的元素
          Elem *e_found = toks_.Find(arc.nextstate);
          if (e_found == NULL) {///如果在哈希表中不存在该状态id
            ///则在哈系表中插入新的元素
            ///状态id为弧边下一个StateId token为下一个token
            toks_.Insert(arc.nextstate, new_tok);
            ///把弧边上下一个状态id插入到向量尾部
            queue_.push_back(arc.nextstate);
          } else {///如果哈希表中已经存在该状态id
            if ( *(e_found->val) < *new_tok ) {///如果前者的代价大于后者的代价 则用新的token代替旧的
              Token::TokenDelete(e_found->val);
              e_found->val = new_tok;
              queue_.push_back(arc.nextstate);///将弧边下一个状态id插入向量中
            } else {///否则删除这个新的token
              Token::TokenDelete(new_tok);
            }
          }
        }
      }
    }
  }
}

///处理输入为空的弧 对于代价小于等于剪枝阈值的继续传递 否则删除
void FasterDecoder::ProcessNone(double cutoff) {
    KALDI_ASSERT(queue_.empty());
    ///遍历哈系表 将所有元素的状态id存入向量queue中
    for(const Elem *e=toks_.GetList();e!=NULL;e=e->tail)
        queue_.push_back(e->key);

    ///只要向量不为空 从后往前访问所有的状态id对应的token
    ///对于代价小于等于剪枝阈值的继续传递下去 否则不予处理
    while(!queue_.empty()){
        StateId state = queue_.back();
        queue_.pop_back();
        Token *tok = toks_.Find(state)->val;
        ///如果当前token处的代价大于剪枝阈值 则不予处理
        if(tok->cost_>cutoff)
            continue;

        KALDI_ASSERT(tok!=NULL&&state==tok->arc_.nextstate);
        ///对于当前代价小于等于剪枝阈值的token
        ///遍历由其状态id出发的所有弧(在解码图的约束下)
        for(fst::ArcIterator<fst::Fst<Arc>> iterator(fst_,state);
            !iterator.Done();
            iterator.Next()){
            Arc arc=iterator.Value();
            ///如果是输入为空的弧边
            if(arc.ilabel==0){
                Token *new_tok = new Token(arc,tok);
                ///如果新token的代价大于剪枝阈值 则删除
                if(new_tok->cost_>cutoff)
                    Token::TokenDelete(new_tok);
                else{
                    ///在哈希表中查找该状态id是否已存在
                    Elem *e = toks_.Find(arc.nextstate);
                    if(e==NULL){///如果哈希表中不存在该状态id 则插入
                        toks_.Insert(arc.nextstate,new_tok);
                        ///插入状态id访问队列中
                        queue_.push_back(arc.nextstate);
                    }
                    else{///如果哈希表中已经存在该状态id
                        if((e->val->cost_)>(new_tok->cost_)){///如果新的token的代价小于原token的代价
                         Token::TokenDelete(e->val);
                         ///将旧token替换成新的token
                         e->val = new_tok;
                         queue_.push_back(arc.nextstate);
                        }else
                            Token::TokenDelete(new_tok);
                    }
                }
            }
        }
    }
}

void FasterDecoder::ClearToks(Elem *list) {
  for (Elem *e = list, *e_tail; e != NULL; e = e_tail) {
    Token::TokenDelete(e->val);
    e_tail = e->tail;
    toks_.Delete(e);
  }
}

} // end namespace kaldi.
