// online/online-faster-decoder.h

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

#ifndef KALDI_ONLINE_ONLINE_FASTER_DECODER_H_
#define KALDI_ONLINE_ONLINE_FASTER_DECODER_H_

#include "util/stl-utils.h"
#include "decoder/faster-decoder.h"
#include "hmm/transition-model.h"

namespace kaldi {

// Extends the definition of FasterDecoder's options to include additional
// parameters. The meaning of the "beam" option is also redefined as
// the _maximum_ beam value allowed.
struct OnlineFasterDecoderOpts : public FasterDecoderOptions {
  BaseFloat rt_min; // minimum decoding runtime factor
  BaseFloat rt_max; // maximum decoding runtime factor
  int32 batch_size; // number of features decoded in one go
  int32 inter_utt_sil; // minimum silence (#frames) to trigger end of utterance
  int32 max_utt_len_; // if utt. is longer, we accept shorter silence as utt. separators
  int32 update_interval; // beam update period in # of frames
  BaseFloat beam_update; // rate of adjustment of the beam
  BaseFloat max_beam_update; // maximum rate of beam adjustment

  OnlineFasterDecoderOpts() :
    rt_min(.7), rt_max(.75), batch_size(27),
    inter_utt_sil(50), max_utt_len_(1500),
    update_interval(3), beam_update(.01),
    max_beam_update(0.05) {}

  void Register(OptionsItf *opts, bool full) {
    FasterDecoderOptions::Register(opts, full);
    opts->Register("rt-min", &rt_min,
                   "Approximate minimum decoding run time factor");
    opts->Register("rt-max", &rt_max,
                   "Approximate maximum decoding run time factor");
    opts->Register("update-interval", &update_interval,
                   "Beam update interval in frames");
    opts->Register("beam-update", &beam_update, "Beam update rate");
    opts->Register("max-beam-update", &max_beam_update, "Max beam update rate");
    opts->Register("inter-utt-sil", &inter_utt_sil,
                   "Maximum # of silence frames to trigger new utterance");
    opts->Register("max-utt-length", &max_utt_len_,
                   "If the utterance becomes longer than this number of frames, "
                   "shorter silence is acceptable as an utterance separator");
  }
};

class OnlineFasterDecoder : public FasterDecoder {
 public:

  ///由Decode()函数返回 以表示当前解码的状态
  enum DecodeState {
    kEndFeats = 1, ///可解码对象中没有更多特征
    kEndUtt = 2, ///语音的结束 例如由足够长的静音引起
    kEndBatch = 4 ///批度块的结束 但是语音还未结束
  };

  // "sil_phones" - the IDs of all silence phones
  OnlineFasterDecoder(const fst::Fst<fst::StdArc> &fst,
                      const OnlineFasterDecoderOpts &opts,
                      const std::vector<int32> &sil_phones,
                      const TransitionModel &trans_model)
      : FasterDecoder(fst, opts), opts_(opts),
        silence_set_(sil_phones), trans_model_(trans_model),
        max_beam_(opts.beam), effective_beam_(FasterDecoder::config_.beam),
        state_(kEndFeats), frame_(0), utt_frames_(0) {}

  DecodeState Decode(DecodableInterface *decodable);
  
  ///通过从最后的永久token到前一个token的回溯来生成一张线性图
  bool PartialTraceback(fst::MutableFst<LatticeArc> *out_fst);

  // Makes a linear graph, by tracing back from the best currently active token
  // to the last immortal token. This method is meant to be invoked at the end
  // of an utterance in order to get the last chunk of the hypothesis
  ///通过从当前最佳活跃token到上一个祖先token的回溯生成线性图。
  ///这个方法意味着在语音的结束部分被唤醒以获取最后一块的假设
  void FinishTraceBack(fst::MutableFst<LatticeArc> *fst_out);

  // Returns "true" if the best current hypothesis ends with long enough silence
  ///如果当前最佳假设因为足够长的静音结束 返回true
  bool EndOfUtterance();

  int32 frame() { return frame_; }

 private:
  void ResetDecoder(bool full);


  ///回溯最后N帧返回一个线性fst，从当前最佳token开始
  void TracebackNFrames(int32 nframes, fst::MutableFst<LatticeArc> *out_fst);


  ///生成一个线性词图，通过由两个token限定的路径回溯
  void MakeLattice(const Token *start,
                   const Token *end,
                   fst::MutableFst<LatticeArc> *out_fst) const;
  ///搜索最后一个token 它是当前所有活跃token的祖先
  void UpdateImmortalToken();

  const OnlineFasterDecoderOpts opts_;
  const ConstIntegerSet<int32> silence_set_; ///静音音素id
  const TransitionModel &trans_model_; /// needed for trans-id -> phone conversion 用于转移id->音素的转换
  const BaseFloat max_beam_; ///最大允许的beam值
  BaseFloat &effective_beam_; /// 当前使用的beam值
  DecodeState state_; /// 当前的解码器状态
  int32 frame_; ///需要被处理的下一帧
  int32 utt_frames_; ///当前语音处理的帧
  Token *immortal_tok_;      // "immortal" token means it's an ancestor of ...
  Token *prev_immortal_tok_; // ... all currently active tokens
  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineFasterDecoder);
};

} // namespace kaldi
#endif // KALDI_ONLINE_ONLINE_FASTER_DECODER_H_
