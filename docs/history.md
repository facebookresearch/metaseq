## History
Metaseq originated as a fork of [fairseq](https://github.com/pytorch/fairseq) that merged [FSDP](https://fairscale.readthedocs.io/en/stable/api/nn/fsdp.html) with [Megatron's](https://github.com/ngoyal2707/Megatron-LM/tree/fairseq_v2) tensor parallel libraries in order to train a 175B using 1k 80GB A100s.

In order to enable faster iteration, we have removed most features offered by fairseq, leaving only the bare minimum set needed to work at 175B scale.  We have also renamed a lot of the Fairseq* classes to be prefixed with Base* or Metaseq*.  The following includes a full list of renamed classes:
* Training internals renaming (optimizer related changes + dropout)
  * FairseqOptimizer &rarr; BaseOptimizer 
  * LegacyFairseqOptimizer &rarr; LegacyOptimizer
  * FairseqLRScheduler &rarr; BaseLRScheduler 
  * FairseqCriterion &rarr; BaseCriterion
  * FairseqIncrementalState &rarr; IncrementalState
  * FairseqAdam &rarr; MetaseqAdam
    *  FairseqAdamConfig &rarr; MetaseqAdamConfig
  * FairseqSGDW &rarr; MetaseqSGDW 
  * FairseqDropout &rarr; Dropout 

* Model arch related renaming
  * FairseqDecoder &rarr; BaseDecoder 
  * FairseqEncoder &rarr; BaseEncoder 
  * DistributedFairseqModel &rarr; DistributedModel 
  * BaseFairseqModel &rarr; BaseModel 
  * FairseqEncoderDecoderModel &rarr; EncoderDecoderModel (to be ripped out, only affected tests)
  * FairseqLanguageModel &rarr; LanguageModel

* Config and circuitry renaming 
  * FairseqTask &rarr; BaseTask 
  * LegacyFairseqTask &rarr; LegacyTask 
  * FairseqDataclass &rarr; MetaseqDataclass 
  * FairseqConfig &rarr; MetaseqConfig 
  * FairseqDataset &rarr; BaseDataset

* Module renaming
  * fairseq &rarr; metaseq
  * fairseq_cli &rarr; metaseq_cli