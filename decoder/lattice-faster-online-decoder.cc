// decoder/lattice-faster-online-decoder.cc

// Copyright 2009-2012  Microsoft Corporation  Mirko Hannemann
//           2013-2014  Johns Hopkins University (Author: Daniel Povey)
//                2014  Guoguo Chen
//                2014  IMSL, PKU-HKUST (author: Wei Shi)
//                2018  Zhehuai Chen

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

// see note at the top of lattice-faster-decoder.cc, about how to maintain this
// file in sync with lattice-faster-decoder.cc

#include "decoder/lattice-faster-online-decoder.h"
#include "lat/lattice-functions.h"

namespace kaldi {

// instantiate this class once for each thing you have to decode.
LatticeFasterOnlineDecoder::LatticeFasterOnlineDecoder(
    const fst::Fst<fst::StdArc> &fst,
    const LatticeFasterDecoderConfig &config):
    fst_(fst), delete_fst_(false), config_(config), num_toks_(0) {
  config.Check();
  toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
}


LatticeFasterOnlineDecoder::LatticeFasterOnlineDecoder(const LatticeFasterDecoderConfig &config,
                                                       fst::Fst<fst::StdArc> *fst):
    fst_(*fst), delete_fst_(true), config_(config), num_toks_(0) {
  config.Check();
  toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
}


LatticeFasterOnlineDecoder::~LatticeFasterOnlineDecoder() {
  DeleteElems(toks_.Clear());
  ClearActiveTokens();
  if (delete_fst_) delete &(fst_);
}

void LatticeFasterOnlineDecoder::InitDecoding() {
  /// 从上一次解码处清理
  DeleteElems(toks_.Clear());
  cost_offsets_.clear();
  ///清理活跃token
  ClearActiveTokens();
  warned_ = true;
  num_toks_ = 0;
  decoding_finalized_ = false;
  final_costs_.clear();
  ///获得解码图初始状态id
  StateId start_state = fst_.Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  ///初始token链表长度置为1
  active_toks_.resize(1);
  Token *start_tok = new Token(0.0, 0.0, NULL, NULL, NULL);
  active_toks_[0].toks = start_tok;
  ///插入初始token
  toks_.Insert(start_state, start_tok);
  ///活跃token个数加1
  num_toks_++;
  ///根据解码图的类型选择不同非发射态处理函数
  ProcessNonemittingWrapper(config_.beam);
}


///如果有任何可用的回溯返回true(不必非得是最终状态).该函数很少返回false;
///如果返回false通常表明这是一个不同寻常的搜索错误
bool LatticeFasterOnlineDecoder::Decode(DecodableInterface *decodable) {
  ///初始化解码 包含清理哈希表中的elem 清除cost_offset_
  ///清理活跃token 重置活跃结点个数 解码状态 最终代价等等
  ///同时还包括获取fst初始状态id 初始化一个空的token
  /// 初始化token链表(初始化时间0处token链表头结点)
  /// 将初始化的token插入哈希表中
  ///最后还要根据fst的类型选择一个处理非发射态的函数
  InitDecoding();

  ///我们在这个解码器中使用从1开始的值索引帧，但是注意可解码对象中
  ///使用的是从0开始计数，所以当我们调用它时不得不注意一下
  while (!decodable->IsLastFrame(NumFramesDecoded() - 1)) {
    ///每25帧剪枝一次
    if (NumFramesDecoded() % config_.prune_interval == 0)
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
    BaseFloat cost_cutoff = ProcessEmittingWrapper(decodable);  // Note: the value returned by
    ProcessNonemittingWrapper(cost_cutoff);
  }
  FinalizeDecoding();

  // Returns true if we have any kind of traceback available (not necessarily
  // to the end state; query ReachedFinal() for that).
  ///如果有任何可用的回溯结果(不必在一个最终的状态)返回真值
  return !active_toks_.empty() && active_toks_.back().toks != NULL;
}





bool LatticeFasterOnlineDecoder::TestGetBestPath(bool use_final_probs) const {
  Lattice lat1;
  {
    Lattice raw_lat;
    GetRawLattice(&raw_lat, use_final_probs);
    ShortestPath(raw_lat, &lat1);
  }
  Lattice lat2;
  GetBestPath(&lat2, use_final_probs);  
  BaseFloat delta = 0.1;
  int32 num_paths = 1;
  if (!fst::RandEquivalent(lat1, lat2, num_paths, delta, rand())) {
    KALDI_WARN << "Best-path test failed";
    return false;
  } else {
    return true;
  }
}


// Outputs an FST corresponding to the single best path through the lattice.
bool LatticeFasterOnlineDecoder::GetBestPath(Lattice *olat,
                                             bool use_final_probs) const {
  olat->DeleteStates();
  BaseFloat final_graph_cost;
  BestPathIterator iter = BestPathEnd(use_final_probs, &final_graph_cost);
  if (iter.Done())
    return false;  // would have printed warning.
  StateId state = olat->AddState();
  olat->SetFinal(state, LatticeWeight(final_graph_cost, 0.0));
  while (!iter.Done()) {
    LatticeArc arc;
    iter = TraceBackBestPath(iter, &arc);
    arc.nextstate = state;
    StateId new_state = olat->AddState();
    olat->AddArc(new_state, arc);
    state = new_state;
  }
  olat->SetStart(state);
  return true;
}


///输出一个对应于未加工的状态级回溯的FST
bool LatticeFasterOnlineDecoder::GetRawLattice(Lattice *ofst,
                                               bool use_final_probs) const {
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  typedef Arc::Label Label;

  ///如果你想要使用use_final_probs=false获取词图的话你不能使用旧的接口(Decode())
  ///你将不得不使用InitDecoding然后调用AdvanceDecoding
  if (decoding_finalized_ && !use_final_probs)
    KALDI_ERR << "You cannot call FinalizeDecoding() and then call "
              << "GetRawLattice() with use_final_probs == false";

  ///token对应最终代价的无序映射表
  unordered_map<Token*, BaseFloat> final_costs_local;

  ///如果解码完成了 则使用最终代价的映射表 否则如果解码未结束则重新定义
  const unordered_map<Token*, BaseFloat> &final_costs =
      (decoding_finalized_ ? final_costs_ : final_costs_local);
  ///如果解码为结束并且使用最终概率 则计算最终代价
  if (!decoding_finalized_ && use_final_probs)
    ComputeFinalCosts(&final_costs_local, NULL, NULL);

  ofst->DeleteStates();

  ///帧数+1(因为帧从1开始计数，并且我们对于开始状态有一个额外的帧)
  int32 num_frames = active_toks_.size() - 1;
  KALDI_ASSERT(num_frames > 0);
  ///桶的个数为 活跃token个数/2+3 (这什么意思阿？)
  const int32 bucket_count = num_toks_/2 + 3;
  ///token到状态id映射表的长度设为桶的个数
  unordered_map<Token*, StateId> tok_map(bucket_count);
  /// 首先创建所有的状态
  /// 初始化存储token的向量
  std::vector<Token*> token_list;
  ///从头开始遍历token链表中每一个token
  for (int32 f = 0; f <= num_frames; f++) {
    if (active_toks_[f].toks == NULL) {///如果该帧下没有活跃的token
      KALDI_WARN << "GetRawLattice: no tokens active on frame " << f
                 << ": not producing lattice.\n";
      return false;
    }
    ///这个函数的作用了解一下
    TopSortTokens(active_toks_[f].toks, &token_list);
    for (size_t i = 0; i < token_list.size(); i++)
      if (token_list[i] != NULL)
        tok_map[token_list[i]] = ofst->AddState();    
  }
  // The next statement sets the start state of the output FST.  Because we
  // topologically sorted the tokens, state zero must be the start-state.
  ofst->SetStart(0);
  
  KALDI_VLOG(4) << "init:" << num_toks_/2 + 3 << " buckets:"
                << tok_map.bucket_count() << " load:" << tok_map.load_factor()
                << " max:" << tok_map.max_load_factor();
  // Now create all arcs.
  for (int32 f = 0; f <= num_frames; f++) {
    for (Token *tok = active_toks_[f].toks; tok != NULL; tok = tok->next) {
      StateId cur_state = tok_map[tok];
      for (ForwardLink *l = tok->links;
           l != NULL;
           l = l->next) {
        unordered_map<Token*, StateId>::const_iterator iter =
            tok_map.find(l->next_tok);
        StateId nextstate = iter->second;
        KALDI_ASSERT(iter != tok_map.end());
        BaseFloat cost_offset = 0.0;
        if (l->ilabel != 0) {  // emitting..
          KALDI_ASSERT(f >= 0 && f < cost_offsets_.size());
          cost_offset = cost_offsets_[f];
        }
        Arc arc(l->ilabel, l->olabel,
                Weight(l->graph_cost, l->acoustic_cost - cost_offset),
                nextstate);
        ofst->AddArc(cur_state, arc);
      }
      if (f == num_frames) {
        if (use_final_probs && !final_costs.empty()) {
          unordered_map<Token*, BaseFloat>::const_iterator iter =
              final_costs.find(tok);
          if (iter != final_costs.end())
            ofst->SetFinal(cur_state, LatticeWeight(iter->second, 0));
        } else {
          ofst->SetFinal(cur_state, LatticeWeight::One());
        }
      }
    }
  }
  return (ofst->NumStates() > 0);
}

bool LatticeFasterOnlineDecoder::GetRawLatticePruned(
    Lattice *ofst,
    bool use_final_probs,
    BaseFloat beam) const {
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  typedef Arc::Label Label;

  // Note: you can't use the old interface (Decode()) if you want to
  // get the lattice with use_final_probs = false.  You'd have to do
  // InitDecoding() and then AdvanceDecoding().
  if (decoding_finalized_ && !use_final_probs)
    KALDI_ERR << "You cannot call FinalizeDecoding() and then call "
              << "GetRawLattice() with use_final_probs == false";

  unordered_map<Token*, BaseFloat> final_costs_local;

  const unordered_map<Token*, BaseFloat> &final_costs =
      (decoding_finalized_ ? final_costs_ : final_costs_local);
  if (!decoding_finalized_ && use_final_probs)
    ComputeFinalCosts(&final_costs_local, NULL, NULL);

  ofst->DeleteStates();
  // num-frames plus one (since frames are one-based, and we have
  // an extra frame for the start-state).
  int32 num_frames = active_toks_.size() - 1;
  KALDI_ASSERT(num_frames > 0);
  for (int32 f = 0; f <= num_frames; f++) {
    if (active_toks_[f].toks == NULL) {
      KALDI_WARN << "GetRawLattice: no tokens active on frame " << f
                 << ": not producing lattice.\n";
      return false;
    }
  }

  unordered_map<Token*, StateId> tok_map;
  std::queue<std::pair<Token*, int32> > tok_queue;
  // First initialize the queue and states.  Put the initial state on the queue;
  // this is the last token in the list active_toks_[0].toks.
  for (Token *tok = active_toks_[0].toks; tok != NULL; tok = tok->next) {
    if (tok->next == NULL) {
      tok_map[tok] = ofst->AddState();
      ofst->SetStart(tok_map[tok]);
      std::pair<Token*, int32> tok_pair(tok, 0);  // #frame = 0
      tok_queue.push(tok_pair);
    }
  }  
  
  // Next create states for "good" tokens
  while (!tok_queue.empty()) {
    std::pair<Token*, int32> cur_tok_pair = tok_queue.front();
    tok_queue.pop();
    Token *cur_tok = cur_tok_pair.first;
    int32 cur_frame = cur_tok_pair.second;
    KALDI_ASSERT(cur_frame >= 0 && cur_frame <= cost_offsets_.size());
    
    unordered_map<Token*, StateId>::const_iterator iter =
        tok_map.find(cur_tok);
    KALDI_ASSERT(iter != tok_map.end());
    StateId cur_state = iter->second;

    for (ForwardLink *l = cur_tok->links;
         l != NULL;
         l = l->next) {
      Token *next_tok = l->next_tok;
      if (next_tok->extra_cost < beam) {
        // so both the current and the next token are good; create the arc
        int32 next_frame = l->ilabel == 0 ? cur_frame : cur_frame + 1;
        StateId nextstate;
        if (tok_map.find(next_tok) == tok_map.end()) {
          nextstate = tok_map[next_tok] = ofst->AddState();
          tok_queue.push(std::pair<Token*, int32>(next_tok, next_frame));
        } else {
          nextstate = tok_map[next_tok];
        }
        BaseFloat cost_offset = (l->ilabel != 0 ? cost_offsets_[cur_frame] : 0);
        Arc arc(l->ilabel, l->olabel,
                Weight(l->graph_cost, l->acoustic_cost - cost_offset),
                nextstate);
        ofst->AddArc(cur_state, arc);
      }
    }
    if (cur_frame == num_frames) {
      if (use_final_probs && !final_costs.empty()) {
        unordered_map<Token*, BaseFloat>::const_iterator iter =
            final_costs.find(cur_tok);
        if (iter != final_costs.end())
          ofst->SetFinal(cur_state, LatticeWeight(iter->second, 0));
      } else {        
        ofst->SetFinal(cur_state, LatticeWeight::One());
      }
    }
  }
  return (ofst->NumStates() != 0);
}

///如果需要的话 重新设定哈希表的大小 以保证哈希表的大小足够
void LatticeFasterOnlineDecoder::PossiblyResizeHash(size_t num_toks) {
  size_t new_sz = static_cast<size_t>(static_cast<BaseFloat>(num_toks)
                                      * config_.hash_ratio);
  if (new_sz > toks_.Size()) {
    toks_.SetSize(new_sz);
  }
}

///该函数要么在哈希toks_中定位一个token，要么在必要的情况下插入一个对于当前帧新的,
/// 空的token(不带前向链接)。
///注意：如果有必要会插入到哈希表toks_中
///并且同时还会插入到这帧活跃tokens的单链表中(该链表的头结点位于active_toks_[frame]中)
inline LatticeFasterOnlineDecoder::Token *LatticeFasterOnlineDecoder::FindOrAddToken(
    StateId state, int32 frame_plus_one, BaseFloat tot_cost,
    Token *backpointer, bool *changed) {

  ///返回token的指针。
  /// 如果changed设为true那么token是新创建的或者代价发生了改变
  KALDI_ASSERT(frame_plus_one < active_toks_.size());
  ///当前帧下token链表的链表头
  Token *&toks = active_toks_[frame_plus_one].toks;
  ///在哈希表中寻找对应的状态id
  ///实际的做法是由状态id映射到对应下标的bucket中 然后根据bucket的数据结构寻找
  ///上一个bucket的最后一个元素 由最后一个元素得到该元素的下一个元素(即为当前bucket的第一个元素)
  ///在当前bucket的第一个元素和最后一个元素间寻找该状态id对应的token
  Elem *e_found = toks_.Find(state);
  if (e_found == NULL) {  ///当前没有这样一个token
    const BaseFloat extra_cost = 0.0;
    // tokens on the currently final frame have zero extra_cost
    // as any of them could end up
    // on the winning path.
    ///创建该状态id的token 该token的下一个token是toks(需要搞清楚)
    ///这里为什么当前帧的新建token的下一个token是当前帧的token链表表头结点呢？？
    Token *new_tok = new Token (tot_cost, extra_cost, NULL, toks, backpointer);
    /// NULL: 表示当前还没有前向链表
    toks = new_tok;   ///toks指向这个当前时刻下的新token
    num_toks_++;   ///当前已分配的token总数加1
    toks_.Insert(state, new_tok);  ///在哈希表中插入新的elem(stateid,token)
    if (changed) *changed = true;

    ///返回这个新建的token
    return new_tok;
  } else {///如果哈希表中已经存在该token 则更新它的参数
    Token *tok = e_found->val;  ///获取该状态下已有的这个token
    if (tok->tot_cost > tot_cost) {  ///取代原来旧的token
      tok->tot_cost = tot_cost;
      tok->backpointer = backpointer;
      ///我们并不分配新的token,原来的token仍然位于active_toks_中,我们只是
      /// 替换了当前帧的tot_cost
      ///当前帧无前向链接
      // in the current frame, there are no forward links (and no extra_cost)
      // only in ProcessNonemitting we have to delete forward links
      // in case we visit a state for the second time
      // those forward links, that lead to this replaced token before:
      // they remain and will hopefully be pruned later (PruneForwardLinks...)
      if (changed) *changed = true;
    } else {
      if (changed) *changed = false;
    }
    return tok; ///返回这个被更新后的token
  }
}

///剪枝在active_toks_[frame]中所有token的前向链接
///该函数由PruneActiveTokens调用 所有前向链接 只要link_extra_cost>lattice_beam都会被剪枝
void LatticeFasterOnlineDecoder::PruneForwardLinks(
    int32 frame_plus_one, bool *extra_costs_changed,
    bool *links_pruned, BaseFloat delta) {
  // delta is the amount by which the extra_costs must change
  // If delta is larger,  we'll tend to go back less far
  //    toward the beginning of the file.
  ///如果任何token的extra_cost发生了改变 则将extra_costs_changed设为true
  ///如果任何token的任何一个link被剪枝 则links_pruned设为true
  *extra_costs_changed = false;
  *links_pruned = false;

  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  ///如果是空token链表 这不应该发生
  if (active_toks_[frame_plus_one].toks == NULL) {
    if (!warned_) {
      KALDI_WARN << "No tokens alive [doing pruning].. warning first "
          "time only for each utterance\n";
      warned_ = true;
    }
  }

  ///我们必须迭代直到不再有更多的变化，因为这些链接不保证是拓扑顺序
  bool changed = true;  ///新的tok_extra_cost - 旧的tok_extra_cost > 1.0 ?
  while (changed) {
    changed = false;
    ///遍历当前帧下的token链表中的所有token
    for (Token *tok = active_toks_[frame_plus_one].toks;
         tok != NULL; tok = tok->next) {
      ForwardLink *link, *prev_link = NULL;
      ///重新计算tok的tok_extra_cost
      BaseFloat tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      ///遍历每一个token的前向链接 tok_extra_cost取前向链接中最小的link_extra_cost
      for (link = tok->links; link != NULL; ) {
        ///看看我们是否需要剪除这个链接
        ///这里next_tok是前向链接指向的token
        Token *next_tok = link->next_tok;
        ///extra_cost初始化的时候为0 和最优路径的差距
        BaseFloat link_extra_cost = next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
             - next_tok->tot_cost);  /// 括号内的差值大于等于0
        ///link_extra_cost是链接开始状态到最终状态的最佳路径打分的差值

        KALDI_ASSERT(link_extra_cost == link_extra_cost);  // check for NaN
        ///如果这个差值大于词图beam 则切除链接
        if (link_extra_cost > config_.lattice_beam) {
          ///获取该token的下一个前向链接
          ForwardLink *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          delete link;
          link = next_link;  /// 更新前向链接为该token的下一个前向链接
          ///成功剪枝前向链接
          *links_pruned = true;
        } else {   /// 如果差值小于词图beam 则保留并且有必要的话更新tok_extra_cost
          if (link_extra_cost < 0.0) {  // this is just a precaution.
            if (link_extra_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          ///如果有更优的link_extra_cost则更新tok_extra_cost为该值
          if (link_extra_cost < tok_extra_cost)
            tok_extra_cost = link_extra_cost;
          prev_link = link;  ///这里开始转移到下一个前向链接 但是同时也保留了没被删除的最后一个
          link = link->next;
        }
      }  ///对于所有的前向链接

      ///delta默认1.0
      if (fabs(tok_extra_cost - tok->extra_cost) > delta)
        changed = true;   /// 新的extra_cost和旧的extra_cost之间的差值大于1.0的话
        ///更新tok的extra_cost为tok_extra_cost
      tok->extra_cost = tok_extra_cost;
      // will be +infinity or <= lattice_beam_.
      // infinity indicates, that no forward link survived pruning
    }  ///对于当前帧的所有的token
    ///只要extra_cost更新了一次 changed就为true 则extra_costs_changed也为true 和变量名一样
    if (changed) *extra_costs_changed = true;

    // Note: it's theoretically possible that aggressive compiler
    // optimizations could cause an infinite loop here for small delta and
    // high-dynamic-range scores.
  } // while changed
}

///该函数是PruneForwardLinks函数的一个我们在最后一帧处调用的版本
///如果存在最终的活跃tokens,它会使用最终概率用于剪枝，否则它将所有tokens视为最终状态
void LatticeFasterOnlineDecoder::PruneForwardLinksFinal() {
  KALDI_ASSERT(!active_toks_.empty());
  ///获取最终的帧id
  int32 frame_plus_one = active_toks_.size() - 1;

  if (active_toks_[frame_plus_one].toks == NULL )  // empty list; should not happen.
    KALDI_WARN << "No tokens alive at end of file\n";

  typedef unordered_map<Token*, BaseFloat>::const_iterator IterType;
  ///计算最终的代价
  ///根据是否存在最终状态计算最终最优代价 需要遍历整张哈希表
  ComputeFinalCosts(&final_costs_, &final_relative_cost_, &final_best_cost_);
  decoding_finalized_ = true;

  ///我们调用该函数作为一个细节，并不是因为真正有必要
  ///否则，可能会有一个时刻，在最后一帧调用PruneTokensForFrame()函数后
  ///toks_.GetList()或者toks_Clear()函数将会包含不存在的tokens
  DeleteElems(toks_.Clear());

  ///现在遍历这一帧的所有tokens，剪枝前向链接...可能需要迭代几次直到不再有更多的变化，
  ///因为链表并非拓扑顺序。这是PruneForwardLinks代码的修改版本，
  /// 但是这里我们也会考虑到最终概率
  bool changed = true;
  BaseFloat delta = 1.0e-05;
  while (changed) {
    changed = false;
    ///遍历该帧的所有token
    for (Token *tok = active_toks_[frame_plus_one].toks;
         tok != NULL; tok = tok->next) {
      ForwardLink *link, *prev_link = NULL;
      // will recompute tok_extra_cost.  It has a term in it that corresponds
      // to the "final-prob", so instead of initializing tok_extra_cost to infinity
      // below we set it to the difference between the (score+final_prob) of this token,
      // and the best such (score+final_prob).
      ///将会重新计算tok_extra_cost 它有一项对应于最终概率，所以我们不会将tok_extra_cost初始化成无穷大
      ///而是将其设置成(score+最终概率)之间的差值，并且将会是最佳的
      BaseFloat final_cost;
      ///如果最终概率的映射表为空 则最终概率初始化为0
      if (final_costs_.empty()) {
        final_cost = 0.0;
      } else {///如果不为空 获取映射表的迭代器
        IterType iter = final_costs_.find(tok);
        ///如果当前不是最后一个最终代价 则获取最终代价
        if (iter != final_costs_.end())
          final_cost = iter->second;
        else///如果当前是最终的代价 则最终代价初始化为无穷大
          final_cost = std::numeric_limits<BaseFloat>::infinity();
      }
      ///该值将会是当前token处的总代价+最终代价-最终最优代价
      BaseFloat tok_extra_cost = tok->tot_cost + final_cost - final_best_cost_;
      // tok_extra_cost will be a "min" over either directly being final, or
      // being indirectly final through other links, and the loop below may
      // decrease its value:
      ///tok_extra_cost无论是直接由其他链接的最终还是间接的最终都将是最小的，
      ///而下面的循环可能会减小其值
      for (link = tok->links; link != NULL; ) {
        ///看看我们是否需要剪除这个链接
        ///获取前向链接的token
        Token *next_tok = link->next_tok;

        ///前向链接额外代价=下一个token的额外代价+(当前token的总代价+前向链接的声学
        /// +语言模型代价-下一个token的总代价)
        BaseFloat link_extra_cost = next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
             - next_tok->tot_cost);
        ///如果该代价大于词图beam则剪除前向链接
        if (link_extra_cost > config_.lattice_beam) {
          ///next_link指向下一个前向链接
          ForwardLink *next_link = link->next;
          ///如果前一个link不为空 则令其的下一个前向链接指向next_link
          if (prev_link != NULL) prev_link->next = next_link;
          ///如果为空的话 则令tok的前向链接直接指向下一个前向链接
          else tok->links = next_link;
          ///释放link的存储空间
          delete link;
          ///更新link 但是保留prev_link
          link = next_link;
        } else { ///保留前向链接并且如果有必要的话更新tok_extra_cost的值
          if (link_extra_cost < 0.0) { ///只是为了以防万一
            if (link_extra_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          ///如果link_extra_cost小于当前token的额外代价则更新
          if (link_extra_cost < tok_extra_cost)
            tok_extra_cost = link_extra_cost;
          ///保留link作为前一个前向链接
          prev_link = link;
          ///更新link指向下一个前向链接
          link = link->next;
        }
      }

      ///剪枝那些在最佳路径上代价大于词图beam的tokens。这一步在非最终状态的情况下
      ///不是必要的，因为在这种情况下没有前向链接。这里，tok_extra_cost拥有与最终概率
      ///相关的额外部分
      if (tok_extra_cost > config_.lattice_beam)
        tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      ///在PruneTokensForFrame被剪枝

      if (!ApproxEqual(tok->extra_cost, tok_extra_cost, delta))
        changed = true;
      ///token的extra_cost将会是无穷大或者比词图beam更小的值
      tok->extra_cost = tok_extra_cost;
    }
  } // while changed

}

BaseFloat LatticeFasterOnlineDecoder::FinalRelativeCost() const {
  if (!decoding_finalized_) {
    BaseFloat relative_cost;
    ComputeFinalCosts(NULL, &relative_cost, NULL);
    return relative_cost;
  } else {
    // we're not allowed to call that function if FinalizeDecoding() has
    // been called; return a cached value.
    return final_relative_cost_;
  }
}


/// 剪去在任何在当前帧下没有前向链接的token
// [we don't do this in PruneForwardLinks because it would give us
// a problem with dangling pointers].
// It's called by PruneActiveTokens if any forward links have been pruned
///该函数由PruneActiveTokens调用 如果有任何前向链接被剪枝了
void LatticeFasterOnlineDecoder::PruneTokensForFrame(int32 frame_plus_one) {
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  ///获得当前帧的token链表头
  Token *&toks = active_toks_[frame_plus_one].toks;
  if (toks == NULL)
    KALDI_WARN << "No tokens alive [doing pruning]\n";
  Token *tok, *next_tok, *prev_tok = NULL;
  for (tok = toks; tok != NULL; tok = next_tok) {
    ///next_tok指向下一个token
    next_tok = tok->next;
    ///如果tok的extra_cost的值为无穷大 则认为该token无前向链接 需要剪除
    if (tok->extra_cost == std::numeric_limits<BaseFloat>::infinity()) {
      // token is unreachable from end of graph; (no forward links survived)
      // excise tok from list and delete tok.
      ///这种情况下 如果其存在前向token 令该token指向当前被删除token的下一个token
      if (prev_tok != NULL) prev_tok->next = tok->next;
      ///如果不存在之前的token 则直接让当前帧的链表头指向被删除token的下一个token
      else toks = tok->next;
      ///删除该token
      delete tok;
      ///token总数-1
      num_toks_--;
    } else {  ///如果该token存在前向链接 则保留
      prev_tok = tok;
    }
  }
}

///回溯活跃的token，并非从当前帧(我们希望保留所有的token)开始剪枝它们
/// 而是从当前帧以前的帧.我们回溯每一帧 直到我们达到一个节点 在那个节点下
///delta-costs不再变化(delta控制着我们什么时候认为一个代价未发生改变)
///delta由config_.lattice_beam*config_.prune_scale决定
void LatticeFasterOnlineDecoder::PruneActiveTokens(BaseFloat delta) {
  ///获取当前已经解码的帧数
  int32 cur_frame_plus_one = NumFramesDecoded();
  int32 num_toks_begin = num_toks_;

  ///下面的索引"f"代表"帧+1"，例如你将不得不减1以得到该可解码对象的对应索引
  for (int32 f = cur_frame_plus_one - 1; f >= 0; f--) {
    ///我们为什么需要在这种情况下剪枝前向链接：
    ///1.我们还未剪枝过它们(新的token链表),初始化的时候两个bool值都为true
    ///2.在任何tokens已经改变了它们的extra_cost以后
    /// 所有之前时刻都需要重新计算
    if (active_toks_[f].must_prune_forward_links) {
      bool extra_costs_changed = false, links_pruned = false;
      ///剪去一些token的前向链接
      PruneForwardLinks(f, &extra_costs_changed, &links_pruned, delta);
      if (extra_costs_changed && f > 0) ///如果有任何一个token的extra_costs发生改变
      ///则认为其前一个时刻的token也必须被剪枝前向链接
        active_toks_[f-1].must_prune_forward_links = true;
      ///如果有任何一个前向链接被剪除
      if (links_pruned)
        ///则认为当前帧token必须剪枝
        active_toks_[f].must_prune_tokens = true;
      ///当前帧的前向链接剪枝完毕 设为false
      active_toks_[f].must_prune_forward_links = false; // job done
    }
    ///f+1时刻
    if (f+1 < cur_frame_plus_one &&      ///f+1 != cur_frame_plus_one - 1 还没有ForwardLink
        active_toks_[f+1].must_prune_tokens) {
      ///将前向链接为空的token删去
      PruneTokensForFrame(f+1);
      ///认为f+1时刻的token已经剪枝完毕
      active_toks_[f+1].must_prune_tokens = false;
    }
  }
  KALDI_VLOG(4) << "PruneActiveTokens: pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}

///计算最终代价
void LatticeFasterOnlineDecoder::ComputeFinalCosts(
    unordered_map<Token*, BaseFloat> *final_costs,
    BaseFloat *final_relative_cost,
    BaseFloat *final_best_cost) const {
  KALDI_ASSERT(!decoding_finalized_);
  ///如果最终代价不为空 则将其重置
  if (final_costs != NULL)
    final_costs->clear();
  ///获取哈希表的表头元素
  const Elem *final_toks = toks_.GetList();
  BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
  ///初始化最优代价和加入最终概率的最优代价为无穷大
  BaseFloat best_cost = infinity,
      best_cost_with_final = infinity;
  ///在遍历到哈希表的末尾之前一直循环
  ///这一步得到所有元素token到状态最终代价的映射
  ///同时得到当前token处的最优代价
  while (final_toks != NULL) {
    StateId state = final_toks->key;
    Token *tok = final_toks->val;
    ///这里获取下一个元素
    const Elem *next = final_toks->tail;
    ///这里得到状态的最终权重值
    BaseFloat final_cost = fst_.Final(state).Value();
    BaseFloat cost = tok->tot_cost,
        cost_with_final = cost + final_cost;///最终代价为当前总代价+状态最终代价
    ///最佳代价为当前代价和最优代价的最小值 实时更新
    best_cost = std::min(cost, best_cost);
    best_cost_with_final = std::min(cost_with_final, best_cost_with_final);
    ///如果无序映射不为空 并且最终代价不为无穷大
    ///给该tok的最终代价赋值
    if (final_costs != NULL && final_cost != infinity)
      (*final_costs)[tok] = final_cost;
    ///访问下一个元素
    final_toks = next;
  }

  if (final_relative_cost != NULL) {
    if (best_cost == infinity && best_cost_with_final == infinity) {
      // Likely this will only happen if there are no tokens surviving.
      // This seems the least bad way to handle it.
      *final_relative_cost = infinity;
    } else {
      ///正常情况下该值为带最终概率的最佳代价-最佳代价
      *final_relative_cost = best_cost_with_final - best_cost;
    }
  }
  if (final_best_cost != NULL) {
    if (best_cost_with_final != infinity) { ///如果存在最终状态
      *final_best_cost = best_cost_with_final;
    } else { ///如果不存在最终状态.
      *final_best_cost = best_cost;
    }
  }
}


LatticeFasterOnlineDecoder::BestPathIterator LatticeFasterOnlineDecoder::BestPathEnd(
    bool use_final_probs,
    BaseFloat *final_cost_out) const {
  if (decoding_finalized_ && !use_final_probs)
    KALDI_ERR << "You cannot call FinalizeDecoding() and then call "
              << "BestPathEnd() with use_final_probs == false";
  KALDI_ASSERT(NumFramesDecoded() > 0 &&
               "You cannot call BestPathEnd if no frames were decoded.");
  
  unordered_map<Token*, BaseFloat> final_costs_local;

  const unordered_map<Token*, BaseFloat> &final_costs =
      (decoding_finalized_ ? final_costs_ : final_costs_local);
  if (!decoding_finalized_ && use_final_probs)
    ComputeFinalCosts(&final_costs_local, NULL, NULL);
  
  // Singly linked list of tokens on last frame (access list through "next"
  // pointer).
  BaseFloat best_cost = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat best_final_cost = 0;
  Token *best_tok = NULL;
  for (Token *tok = active_toks_.back().toks; tok != NULL; tok = tok->next) {
    BaseFloat cost = tok->tot_cost, final_cost = 0.0;
    if (use_final_probs && !final_costs.empty()) {
      // if we are instructed to use final-probs, and any final tokens were
      // active on final frame, include the final-prob in the cost of the token.
      unordered_map<Token*, BaseFloat>::const_iterator iter = final_costs.find(tok);
      if (iter != final_costs.end()) {
        final_cost = iter->second;
        cost += final_cost;
      } else {
        cost = std::numeric_limits<BaseFloat>::infinity();
      }
    }
    if (cost < best_cost) {
      best_cost = cost;
      best_tok = tok;
      best_final_cost = final_cost;
    }
  }    
  if (best_tok == NULL) {  // this should not happen, and is likely a code error or
    // caused by infinities in likelihoods, but I'm not making
    // it a fatal error for now.
    KALDI_WARN << "No final token found.";
  }
  if (final_cost_out)
    *final_cost_out = best_final_cost;
  return BestPathIterator(best_tok, NumFramesDecoded() - 1);
}


LatticeFasterOnlineDecoder::BestPathIterator LatticeFasterOnlineDecoder::TraceBackBestPath(
    BestPathIterator iter, LatticeArc *oarc) const {
  KALDI_ASSERT(!iter.Done() && oarc != NULL);
  Token *tok = static_cast<Token*>(iter.tok);
  int32 cur_t = iter.frame, ret_t = cur_t;
  if (tok->backpointer != NULL) {
    ForwardLink *link;
    for (link = tok->backpointer->links;
         link != NULL; link = link->next) {
      if (link->next_tok == tok) { // this is the link to "tok"
        oarc->ilabel = link->ilabel;
        oarc->olabel = link->olabel;
        BaseFloat graph_cost = link->graph_cost,
            acoustic_cost = link->acoustic_cost;
        if (link->ilabel != 0) {
          KALDI_ASSERT(static_cast<size_t>(cur_t) < cost_offsets_.size());
          acoustic_cost -= cost_offsets_[cur_t];
          ret_t--;
        }
        oarc->weight = LatticeWeight(graph_cost, acoustic_cost);
        break;
      }
    }
    if (link == NULL) { // Did not find correct link.
      KALDI_ERR << "Error tracing best-path back (likely "
                << "bug in token-pruning algorithm)";
    }
  } else {
    oarc->ilabel = 0;
    oarc->olabel = 0;
    oarc->weight = LatticeWeight::One(); // zero costs.
  }
  return BestPathIterator(tok->backpointer, ret_t);
}

///推动解码的进行
void LatticeFasterOnlineDecoder::AdvanceDecoding(DecodableInterface *decodable,
                                                   int32 max_num_frames) {
  ///如果活跃的token数不为零并且解码未完成
  KALDI_ASSERT(!active_toks_.empty() && !decoding_finalized_ &&
               "You must call InitDecoding() before AdvanceDecoding");
  ///这里获取可解码对象中已经准备好的帧数
  int32 num_frames_ready = decodable->NumFramesReady();

  ///num_frames_ready必须>=num_frames_decoded,否则准备好的帧数必须减少(不切实际)
  ///或者可解码对象发生改动(不被允许)
  KALDI_ASSERT(num_frames_ready >= NumFramesDecoded());
  ///得到解码的目标帧数
  int32 target_frames_decoded = num_frames_ready;
  ///如果设置了帧数上限(该值默认为-1)
  if (max_num_frames >= 0)
    target_frames_decoded = std::min(target_frames_decoded,
                                     NumFramesDecoded() + max_num_frames);
  ///只要已解码的帧数少于目标解码的帧数 则一直循环
  while (NumFramesDecoded() < target_frames_decoded) {
    ///kaldi中默认每25帧剪枝一次
    if (NumFramesDecoded() % config_.prune_interval == 0) {
      ///该函数也会删除一些不必要的token，但是阈值的计算是在
      ///active_toks_中进行的
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
    }
    ///注意：ProcessEmitting() 函数使NumFramesDecoded()函数的返回值累加
    BaseFloat cost_cutoff = ProcessEmittingWrapper(decodable);
    ProcessNonemittingWrapper(cost_cutoff);
  }
}

///该函数是PruneActivceTokens函数的一个我们可以在最后一帧处任意调用的版本
///考虑到tokens的最终概率 该函数曾经叫PruneActiveTokensFinal
// FinalizeDecoding() is a version of PruneActiveTokens that we call
// (optionally) on the final frame.  Takes into account the final-prob of
// tokens.  This function used to be called PruneActiveTokensFinal().
void LatticeFasterOnlineDecoder::FinalizeDecoding() {
  ///得到当前解码的帧id+1的值
  int32 final_frame_plus_one = NumFramesDecoded();
  ///获得活跃token的个数
  int32 num_toks_begin = num_toks_;
  // PruneForwardLinksFinal() prunes final frame (with final-probs), and
  // sets decoding_finalized_.
  ///该函数剪枝最终帧(带有最终概率的),并且设定decoding_finalized_
  PruneForwardLinksFinal();
  ///回溯每一帧
  for (int32 f = final_frame_plus_one - 1; f >= 0; f--) {
    bool b1, b2; // values not used.
    ///不断更新的delta值
    BaseFloat dontcare = 0.0; // delta of zero means we must always update
    ///对于f时刻下的所有token 判断是否需要剪除前向链接
    PruneForwardLinks(f, &b1, &b2, dontcare);
    ///对于f+1时刻删除前向链接为空的token
    PruneTokensForFrame(f + 1);
  }
  ///剪除第一帧无前向链接的token
  PruneTokensForFrame(0);
  KALDI_VLOG(4) << "pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}

/// 获取权重截断值.  同时计算活跃tokens.
/// 获取剪枝阈值的调整beam值 还有最佳elem
BaseFloat LatticeFasterOnlineDecoder::GetCutoff(Elem *list_head, size_t *tok_count,
                                                BaseFloat *adaptive_beam, Elem **best_elem) {
  BaseFloat best_weight = std::numeric_limits<BaseFloat>::infinity();
  // positive == high cost == bad.
  size_t count = 0;
  ///如果max和min都是用默认值
  if (config_.max_active == std::numeric_limits<int32>::max() &&
      config_.min_active == 0) {
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      ///w为当前token的代价
      BaseFloat w = static_cast<BaseFloat>(e->val->tot_cost);
      ///找到最小的代价
      if (w < best_weight) {
        best_weight = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;
    if (adaptive_beam != NULL) *adaptive_beam = config_.beam;
    ///返回剪枝的阈值
    return best_weight + config_.beam;
  } else {///如果max和min不是默认值 修改成了特定值
    tmp_array_.clear();
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      BaseFloat w = e->val->tot_cost;
      tmp_array_.push_back(w);
      if (w < best_weight) {
        best_weight = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;

    ///剪枝的阈值
    BaseFloat beam_cutoff = best_weight + config_.beam,
        min_active_cutoff = std::numeric_limits<BaseFloat>::infinity(),
        max_active_cutoff = std::numeric_limits<BaseFloat>::infinity();

    KALDI_VLOG(6) << "Number of tokens active on frame " << NumFramesDecoded()
                  << " is " << tmp_array_.size();

    if (tmp_array_.size() > static_cast<size_t>(config_.max_active)) {
      ///求第n大的数把它放在位置n上 从0开始计数
      std::nth_element(tmp_array_.begin(),
                       tmp_array_.begin() + config_.max_active,
                       tmp_array_.end());
      ///获取第max_active大的数
      max_active_cutoff = tmp_array_[config_.max_active];
    }
    ///如果第max_active个token的代价小于原剪枝阈值 则更新
    if (max_active_cutoff < beam_cutoff) { // max_active is tighter than beam.
      if (adaptive_beam)
        *adaptive_beam = max_active_cutoff - best_weight + config_.beam_delta;
      ///返回这个更小的剪枝阈值
      return max_active_cutoff;
    }    
    if (tmp_array_.size() > static_cast<size_t>(config_.min_active)) {
      ///默认情况下最优化的剪枝阈值为best_weight
      if (config_.min_active == 0) min_active_cutoff = best_weight;
      else {
        std::nth_element(tmp_array_.begin(),
                         tmp_array_.begin() + config_.min_active,
                         tmp_array_.size() > static_cast<size_t>(config_.max_active) ?
                         tmp_array_.begin() + config_.max_active :
                         tmp_array_.end());
        min_active_cutoff = tmp_array_[config_.min_active];
      }
    }
    ///如果最小剪枝阈值比原剪枝阈值更宽松 则更新
    if (min_active_cutoff > beam_cutoff) { // min_active is looser than beam.
      if (adaptive_beam)
        *adaptive_beam = min_active_cutoff - best_weight + config_.beam_delta;
      return min_active_cutoff;
    } else {
      *adaptive_beam = config_.beam;
      return beam_cutoff;
    }
  }
}


template <typename FSTtype>
BaseFloat LatticeFasterOnlineDecoder::Processtransition(kaldi::DecodableInterface *decodable) {
  KALDI_ASSERT(active_toks_.size()>0);
  ///获取当前帧的id 帧序号为帧数-1
  int32 frame=active_toks_.size() - 1;

  ///拓展tokens列表的大小 相当于下一时刻的token
  active_toks_.resize(active_toks_.size() + 1);

  ///获取哈希表头元素
  Elem *final_ele = toks_.Clear();
  ///存储最佳元素
  Elem *best_ele = NULL;

  ///记录活跃tok的个数以及beam值调整项
  size_t tok_cnt;
  BaseFloat adaptive_beam;

  ///调用函数获取第一层剪枝阈值
  BaseFloat cur_cutoff = GetCutoff(final_ele,&tok_cnt,&adaptive_beam,&best_ele);

  ///确保哈希表大小足够大
  PossiblyResizeHash(tok_cnt);

  ///初始化第二层剪枝阈值为浮点数最大值
  BaseFloat next_cutoff = std::numeric_limits<BaseFloat >::infinity();
  ///初始化代价偏置值
  BaseFloat cost_offset = 0.0;

  ///获取解码图
  const FSTtype &fst = dynamic_cast<const FSTtype&>(fst_);
  ///由最佳元素的状态id所对应的弧边搜索更新最佳权重
  if(best_ele){
    StateId state = best_ele->key;

    Token *tok = best_ele->val;

    ///这边使用权重的负值还不明白什么意思
    cost_offset = -tok->tot_cost;

    ///遍历该状态下的所有发射态的弧边 更新第二次剪枝阈值的值
    for(fst::ArcIterator<FSTtype> iterator(fst,state);
                                   !iterator.Done();
                                   iterator.Next()){
      ///获取当前迭代器下的弧边
      Arc arc=iterator.Value();
      ///只处理有输入的边
      if(arc.ilabel!=0){
        ///这里又是为什么要减去声学似然度
         BaseFloat new_weight=arc.weight.Value()+cost_offset
                              -decodable->LogLikelihood(frame,arc.ilabel)
                              +tok->tot_cost;
         ///如果新的权重值小于第二次剪枝阈值 则更新该值
         if(new_weight+adaptive_beam<next_cutoff)
           next_cutoff = new_weight+adaptive_beam;
      }
    }
  }
  ///拓展代价偏置值
  cost_offsets_.resize(frame+1,0.0);
  cost_offsets_[frame] = cost_offset;

  ///开始由表头元素遍历哈希表 处理有输入的状态 以及状态对应的弧边
  for(Elem *e = final_ele,*e_tail;e!=NULL;e=e_tail){
     StateId state = e->key;
     Token *tok = e->val;
     ///如果当前token的总代价小于等于第一层剪枝权重 则保留token继续传递下去
     ///遍历该状态id对应的所有有输入的弧边
     if(tok->tot_cost <= cur_cutoff)
       ///需要注意 遍历哈希表中元素时 如果token的总代价小于剪枝阈值
       ///那么在访问该token对应的状态id所对应的弧边时应该受到解码图的约束
       for(fst::ArcIterator<FSTtype>iterator(fst,state);
           !iterator.Done();
           iterator.Next()){
         Arc arc = iterator.Value();
         if(arc.ilabel!=0){
           ///声学代价的计算需要加上代价偏直值使其值维持在一个
           /// 可以接收的动态范围内
            BaseFloat ac_cost = cost_offset-
                      decodable->LogLikelihood(frame,arc.ilabel),
                      graph_cost = arc.weight.Value(),
                      cur_cost = tok->tot_cost;
            ///总代价为三者的和 其中声学代价需要加上一个偏置值 比较特殊
            BaseFloat total_cost = ac_cost+graph_cost+cur_cost;
            ///如果当前代价已经超过了第二次剪枝阈值 则忽略
            if(total_cost > next_cutoff) continue;
            ///如果当前代价足够小 则更新第二次剪枝的阈值
            else if(total_cost + adaptive_beam <= next_cutoff)
              next_cutoff = total_cost + adaptive_beam;

            ///如果弧上下一个状态id未在哈希表中出现则新建token
            ///否则更新已有token的参数
            Token *next_tok = FindOrAddToken(arc.nextstate,frame+1,total_cost,tok,NULL);

            //ForwardLink *next_link = ForwardLink(next_tok,arc.ilabel,arc.olabel,arc.weight.Value(),)
            ///对于新建的token则其前向链接应该为空 若是已存在的token 则仅仅更新其前向链接
            ///相当于在原前向链接中插入新的前向链接
            tok->links = new ForwardLink(next_tok,arc.ilabel,arc.olabel,graph_cost,ac_cost,tok->links);
         }
       }
       e_tail = e->tail;
       toks_.Delete(e);
  }
  return next_cutoff;
}
///处理发射态 这部分代码主要实现两个功能
///一个是token信息沿着输入不为空的弧上的传递
///另一个就是若干剪枝阈值的评估
template <typename FstType>
BaseFloat LatticeFasterOnlineDecoder::ProcessEmitting(
    DecodableInterface *decodable) {
  KALDI_ASSERT(active_toks_.size() > 0);
  int32 frame = active_toks_.size() - 1; // frame is the frame-index
  // (zero-based) used to get likelihoods
  // from the decodable object.
  ///tokens的列表 由帧进行索引 用于从可解码对象那里获取似然度
  ///每运行一次processemitting都会扩充一次
  active_toks_.resize(active_toks_.size() + 1);

  ///这里获取哈希表的头结点
  Elem *final_toks = toks_.Clear(); // analogous to swapping prev_toks_ / cur_toks_
  // in simple-decoder.h.   Removes the Elems from
  // being indexed in the hash in toks_.
  Elem *best_elem = NULL;
  BaseFloat adaptive_beam;
  ///记录活跃token的个数
  size_t tok_cnt;
  ///adaptive_beam由GetCutoff函数估计得到 用于next_cutoff
  ///这里还得到了前一时刻token链表中活跃token的个数 以及best_elem 即最优的token
  ///还有前一时刻剪枝的阈值
  BaseFloat cur_cutoff = GetCutoff(final_toks, &tok_cnt, &adaptive_beam, &best_elem);
  /// 确保哈希表总是足够大
  PossiblyResizeHash(tok_cnt);

  ///第二层剪枝的阈值初始化
  BaseFloat next_cutoff = std::numeric_limits<BaseFloat>::infinity();
  // pruning "online" before having seen all tokens

  ///对于每一帧包含一个偏置值以使声学似然度保留在一个合适的动态范围
  BaseFloat cost_offset = 0.0;

  const FstType &fst = dynamic_cast<const FstType&>(fst_);

  ///首先处理最佳token以使next_cutoff获得一个合理的紧缩范围
  ///下一块代码的唯二产物是next_cutoff和cost_offset
  if (best_elem) {
    StateId state = best_elem->key;
    ///获取最佳token
    Token *tok = best_elem->val;
    ///代价偏直值是最佳token的总代价
    cost_offset = - tok->tot_cost;
    ///遍历状态的连接弧
    for (fst::ArcIterator<FstType> aiter(fst, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      ///如果是发射状态
      if (arc.ilabel != 0) {  // propagate..
        ///新的权重值 即新的总代价 是图代价+声学代价+之前的总代价
        BaseFloat new_weight = arc.weight.Value() + cost_offset - 
            decodable->LogLikelihood(frame, arc.ilabel) + tok->tot_cost;
        ///如果新的权重+调整beam<第二层剪枝阈值 则更新该剪枝阈值
        if (new_weight + adaptive_beam < next_cutoff)
          next_cutoff = new_weight + adaptive_beam;
      }
    }
  }

  ///存储我们运用在声学似然度上的偏置值
  // Could just do cost_offsets_.push_back(cost_offset), but we
  // do it this way as it's more robust to future code changes.
  cost_offsets_.resize(frame + 1, 0.0);
  cost_offsets_[frame] = cost_offset;

  ///tokens现在由final_toks所拥有，并且哈希表为空
  ///重点是我们对于每一个elem'e'需要调用DeleteElem以让toks_知道我们已经处理完它们了
  // 'owned' is a complex thing here; the point is we need to call DeleteElem
  // on each elem 'e' to let toks_ know we're done with them.
  for (Elem *e = final_toks, *e_tail; e != NULL; e = e_tail) {
    // loop this way because we delete "e" as we go.
    StateId state = e->key;
    Token *tok = e->val;
    ///如果到当前token处总代价小于等于第一轮剪枝阈值 则保留
    if (tok->tot_cost <= cur_cutoff) {
      ///遍历状态链接的弧
      for (fst::ArcIterator<FstType> aiter(fst, state);
           !aiter.Done();
           aiter.Next()) {
        const Arc &arc = aiter.Value();
        ///如果是发射状态
        if (arc.ilabel != 0) {  // propagate..
          BaseFloat ac_cost = cost_offset -
              decodable->LogLikelihood(frame, arc.ilabel),
              graph_cost = arc.weight.Value(),
              cur_cost = tok->tot_cost,
              tot_cost = cur_cost + ac_cost + graph_cost;
          ///如果当前总代价大于第二轮剪枝阈值 则舍弃
          if (tot_cost > next_cutoff) continue;
          ///否则保留 且更新新的第二轮阈值
          else if (tot_cost + adaptive_beam < next_cutoff)
            next_cutoff = tot_cost + adaptive_beam; /// 由当前最佳token进行剪枝

          ///注意：帧索引在active_toks_中是从1开始的 所以要+1
          ///找到或者添加token函数的作用是创建新的token(当前时刻)
          ///或者更新已有的token参数
          ///这里的tok作为前一时刻最优的token
          Token *next_tok = FindOrAddToken(arc.nextstate,
                                           frame + 1, tot_cost, tok, NULL);
          // NULL: no change indicator needed

          // Add ForwardLink from tok to next_tok (put on head of list tok->links)
          ///从tok到next_tok添加前向链接 links是上一时刻token的前向链表
          ///把当前时刻新的token加入上一时刻的前向链表中(需要理解其意思)
          ///这里的意思是说把当前时刻新的token加入这个前向链表中
          tok->links = new ForwardLink(next_tok, arc.ilabel, arc.olabel,
                                       graph_cost, ac_cost, tok->links);
        }
      } ///对于所有的弧
    }
    ///e_tail指向哈希表中下一个元素(对应不同的状态id)
    e_tail = e->tail;
    ///由链表中回收
    toks_.Delete(e); // delete Elem
  }
  ///返回二次剪枝的阈值
  return next_cutoff;
}

template BaseFloat LatticeFasterOnlineDecoder::
    ProcessEmitting<fst::ConstFst<fst::StdArc>>(DecodableInterface *decodable);
template BaseFloat LatticeFasterOnlineDecoder::
    ProcessEmitting<fst::VectorFst<fst::StdArc>>(DecodableInterface *decodable);
template BaseFloat LatticeFasterOnlineDecoder::
    ProcessEmitting<fst::Fst<fst::StdArc>>(DecodableInterface *decodable);

BaseFloat LatticeFasterOnlineDecoder::ProcessEmittingWrapper(
        DecodableInterface *decodable) {
  if (fst_.Type() == "const") {
    return LatticeFasterOnlineDecoder::
        ProcessEmitting<fst::ConstFst<Arc>>(decodable);
  } else if (fst_.Type() == "vector") {
    return LatticeFasterOnlineDecoder::
        ProcessEmitting<fst::VectorFst<Arc>>(decodable);
  } else {
    return LatticeFasterOnlineDecoder::
        ProcessEmitting<fst::Fst<Arc>>(decodable);
  }
}

///由函数名可看出 这一步处理当前帧下输入为epsilon的跳转/弧
///也就是说没有声学的似然概率
template <typename FstType> 
void LatticeFasterOnlineDecoder::ProcessNonemitting(BaseFloat cutoff) {
  KALDI_ASSERT(!active_toks_.empty());
  ///得到当前帧的序号
  int32 frame = static_cast<int32>(active_toks_.size()) - 2;
  // Note: "frame" is the time-index we just processed, or -1 if
  // we are processing the nonemitting transitions before the
  // first frame (called from InitDecoding()).
  const FstType &fst = dynamic_cast<const FstType&>(fst_);

  // Processes nonemitting arcs for one frame.  Propagates within toks_.
  // Note-- this queue structure is is not very optimal as
  // it may cause us to process states unnecessarily (e.g. more than once),
  // but in the baseline code, turning this vector into a set to fix this
  // problem did not improve overall speed.
 ///处理一帧非发射态的弧
 ///push当前时刻到达的状态 获取的是在processemitting中新
 ///产生的token也就是当前时刻的token
  KALDI_ASSERT(queue_.empty());

  ///获取哈希表的表头 循环遍历哈希表 找到有空跳转的状态进行后期的遍历处理
  for (const Elem *e = toks_.GetList(); e != NULL;  e = e->tail) {
    StateId state = e->key;
    ///如果该状态对应的弧存在空跳转则入队 在之后会遍历这些状态以找到空跳转的弧
    ///然后进行处理
    if (fst_.NumInputEpsilons(state) != 0)
      queue_.push_back(state);
  }
  ///默认不需要警告

  if (queue_.empty()) {
    if (!warned_) {
      KALDI_WARN << "Error, no surviving tokens: frame is " << frame;
      warned_ = true;
    }
  }

  ///queue_中存储的是由当前时刻的token对应的状态出发，可以通过
  ///空跳转到达的所有状态
  while (!queue_.empty()) {
    StateId state = queue_.back();
    queue_.pop_back();

    ///在哈希表中根据状态id找到对应的token
    Token *tok = toks_.Find(state)->val;  // would segfault if state not in toks_ but this can't happen.
    ///当前token处的代价
    BaseFloat cur_cost = tok->tot_cost;
    ///如果当前代价大于截断值则不需要后续操作 看下一个token
    if (cur_cost > cutoff) // Don't bother processing successors.
      continue;

    ///如果tok存在任何现有的前向链接 删除它们,因为我们打算重新生成它们
    // If "tok" has any existing forward links, delete them,
    // because we're about to regenerate them.  This is a kind
    // of non-optimality (remember, this is the simple decoder),
    // but since most states are emitting it's not a huge issue.
    tok->DeleteForwardLinks(); // necessary when re-visiting
    tok->links = NULL;
    ///遍历当前状态id在解码图中每一条弧 并找到空跳转的弧
    for (fst::ArcIterator<FstType> aiter(fst, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      ///如果输入标签为0 也就是非发射态
      if (arc.ilabel == 0) {  // propagate nonemitting only...
        ///语言模型权重为弧的权重值
        ///总代价为当前代价+语言模型代价+声学代价(非发射态为0)
        BaseFloat graph_cost = arc.weight.Value(),
            tot_cost = cur_cost + graph_cost;
        if (tot_cost < cutoff) {
          bool changed;

          Token *new_tok = FindOrAddToken(arc.nextstate, frame + 1, tot_cost,
                                          tok, &changed);

          tok->links = new ForwardLink(new_tok, 0, arc.olabel,
                                       graph_cost, 0, tok->links);

          // "changed" tells us whether the new token has a different
          // cost from before, or is new [if so, add into queue].
          ///形成新的token或者更新了权值说明新状态可达
          if (changed && fst_.NumInputEpsilons(arc.nextstate) != 0) 
            queue_.push_back(arc.nextstate);
        }
      }
    } // for all arcs
  } // while queue not empty
}///对于所有的弧 只要队列不为空 做类似的操作

template void LatticeFasterOnlineDecoder::
    ProcessNonemitting<fst::ConstFst<fst::StdArc>>(BaseFloat cutoff);
template void LatticeFasterOnlineDecoder::
    ProcessNonemitting<fst::VectorFst<fst::StdArc>>(BaseFloat cutoff);
template void LatticeFasterOnlineDecoder::
    ProcessNonemitting<fst::Fst<fst::StdArc>>(BaseFloat cutoff);

///这里代价截断值是beam的值 beam作为一个容错阈值存在
void LatticeFasterOnlineDecoder::ProcessNonemittingWrapper(
        BaseFloat cost_cutoff) {
  if (fst_.Type() == "const") {
    return LatticeFasterOnlineDecoder::
        ProcessNonemitting<fst::ConstFst<Arc>>(cost_cutoff);
  } else if (fst_.Type() == "vector") {
    return LatticeFasterOnlineDecoder::
        ProcessNonemitting<fst::VectorFst<Arc>>(cost_cutoff);
  } else {
    return LatticeFasterOnlineDecoder::
        ProcessNonemitting<fst::Fst<Arc>>(cost_cutoff);
  }
}

///这个函数其实是由哈希表的表头 访问整张哈希表并且删除表中所有元素
void LatticeFasterOnlineDecoder::DeleteElems(Elem *list) {
  for (Elem *e = list, *e_tail; e != NULL; e = e_tail) {
    // Token::TokenDelete(e->val);
    e_tail = e->tail;
    toks_.Delete(e);
  }
}

///删除token链表中的所有token 外循环遍历链表中所有头节点(访问每一帧对应的所有token)
///内循环根据每一帧 访问该帧下所有token 删除其前向链接
void LatticeFasterOnlineDecoder::ClearActiveTokens() { // a cleanup routine, at utt end/begin
  for (size_t i = 0; i < active_toks_.size(); i++) {
    ///访问该帧下每一个token 并删除这些token的前向链接
    for (Token *tok = active_toks_[i].toks; tok != NULL; ) {
      tok->DeleteForwardLinks();
      Token *next_tok = tok->next;
      delete tok;
      num_toks_--;
      tok = next_tok;
    }
  }
  active_toks_.clear();
  KALDI_ASSERT(num_toks_ == 0);
}

///拓扑排序tokens 这里tok_list是每一帧token单链表的表头
void LatticeFasterOnlineDecoder::TopSortTokens(Token *tok_list,
                                               std::vector<Token*> *topsorted_list) {
  ///token到位置的无序映射表
  unordered_map<Token*, int32> token2pos;
  typedef unordered_map<Token*, int32>::iterator IterType;
  int32 num_toks = 0;
  ///遍历当前帧下的token链表 得到当前时刻下活跃token的个数
  for (Token *tok = tok_list; tok != NULL; tok = tok->next)
    num_toks++;
  int32 cur_pos = 0;

  ///我们赋值token的数字为num_toks-1,...,2,1,0 这可能比我们给予它们递减顺序
  /// 更接近于拓扑顺序，因为新的token会在链表的头部放入
  ///按逆序给token的位置赋值
  for (Token *tok = tok_list; tok != NULL; tok = tok->next)
    token2pos[tok] = num_toks - ++cur_pos;

  ///预处理的token无序结合
  unordered_set<Token*> reprocess;

  ///迭代访问无序映射表
  for (IterType iter = token2pos.begin(); iter != token2pos.end(); ++iter) {
      ///得到token及其位置参数
    Token *tok = iter->first;
    int32 pos = iter->second;
    ///遍历当前token的所有前向链接
    for (ForwardLink *link = tok->links; link != NULL; link = link->next) {
        ///如果前向链接为非发射态
      if (link->ilabel == 0) {
        // We only need to consider epsilon links, since non-epsilon links
        // transition between frames and this function only needs to sort a list
        // of tokens from a single frame.
        ///我们只需要考虑
        IterType following_iter = token2pos.find(link->next_tok);
        if (following_iter != token2pos.end()) { // another token on this frame,
                                                 // so must consider it.
          int32 next_pos = following_iter->second;
          if (next_pos < pos) { // reassign the position of the next Token.
            following_iter->second = cur_pos++;
            reprocess.insert(link->next_tok);
          }
        }
      }
    }
    // In case we had previously assigned this token to be reprocessed, we can
    // erase it from that set because it's "happy now" (we just processed it).
    reprocess.erase(tok);
  }

  size_t max_loop = 1000000, loop_count; // max_loop is to detect epsilon cycles.
  for (loop_count = 0;
       !reprocess.empty() && loop_count < max_loop; ++loop_count) {
    std::vector<Token*> reprocess_vec;
    for (unordered_set<Token*>::iterator iter = reprocess.begin();
         iter != reprocess.end(); ++iter)
      reprocess_vec.push_back(*iter);
    reprocess.clear();
    for (std::vector<Token*>::iterator iter = reprocess_vec.begin();
         iter != reprocess_vec.end(); ++iter) {
      Token *tok = *iter;
      int32 pos = token2pos[tok];
      // Repeat the processing we did above (for comments, see above).
      for (ForwardLink *link = tok->links; link != NULL; link = link->next) {
        if (link->ilabel == 0) {
          IterType following_iter = token2pos.find(link->next_tok);
          if (following_iter != token2pos.end()) {
            int32 next_pos = following_iter->second;
            if (next_pos < pos) {
              following_iter->second = cur_pos++;
              reprocess.insert(link->next_tok);
            }
          }
        }
      }
    }
  }
  KALDI_ASSERT(loop_count < max_loop && "Epsilon loops exist in your decoding "
               "graph (this is not allowed!)");

  topsorted_list->clear();
  topsorted_list->resize(cur_pos, NULL);  // create a list with NULLs in between.
  for (IterType iter = token2pos.begin(); iter != token2pos.end(); ++iter)
    (*topsorted_list)[iter->second] = iter->first;
}




} // end namespace kaldi.
