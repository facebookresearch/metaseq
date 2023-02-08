## About OPT-IML
[OPT-IML (OPT + Instruction Meta-Learning)](https://arxiv.org/pdf/2212.12017.pdf) is a set of instruction-tuned versions of OPT, on a collection of ~2000 NLP tasks gathered from 8 NLP benchmarks, called OPT-IML Bench. 

We provide two model versions: 
* OPT-IML trained on 1500 tasks with several tasks held-out for purposes of downstream evaluation, and 
* OPT-IML-Max trained on all ~2000 tasks

## Pretrained Model Weights
| Model               | Parameters |       Pretrained weights                                 |
|---------------------|:----------:|:--------------------------------------------------------:|
| OPT-IML-30B         |    30B     |      [part0](https://dl.fbaipublicfiles.com/optiml/aws.v7.prop10.30b.eps_4096.docsep_2.mu4000.wu60.bsz8.clip1.0.fp32adam.rs1234.lr5e-05.pat_8000.ngpu64/mp2/checkpoint_1_4000.pt-model_part-0.pt), [part1](https://dl.fbaipublicfiles.com/optiml/aws.v7.prop10.30b.eps_4096.docsep_2.mu4000.wu60.bsz8.clip1.0.fp32adam.rs1234.lr5e-05.pat_8000.ngpu64/mp2/checkpoint_1_4000.pt-model_part-1.pt)   |
| OPT-IML-Max-30B     |    30B     |      [part0](https://dl.fbaipublicfiles.com/optiml/aws.v7.2000.prop10.30b.eps_4096.docsep_2.mu6000.wu60.bsz8.clip1.0.fp32adam.rs1234.lr5e-05.pat_8000.ngpu64/mp2/checkpoint_1_6000.pt-model_part-0.pt), [part1](https://dl.fbaipublicfiles.com/optiml/aws.v7.2000.prop10.30b.eps_4096.docsep_2.mu6000.wu60.bsz8.clip1.0.fp32adam.rs1234.lr5e-05.pat_8000.ngpu64/mp2/checkpoint_1_6000.pt-model_part-1.pt)    |
| OPT-IML-175B        |    175B    |      [Request](https://docs.google.com/forms/d/19jE4WVSMcUy1YcVFGZcU2Q4KvDWGgwFy1tYqGZ02x1k) models and then follow [download instructions](download_optiml175b.md).            |


## License
The use of OPT-IML model weights is subject to the [OPT Model License](../OPT/MODEL_LICENSE.md).

