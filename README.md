

# Metaseq
A codebase for working with [Open Pre-trained Transformers](projects/OPT), originally forked from [fairseq](https://github.com/facebookresearch/fairseq).


## Community Integrations

### Using OPT with 🤗 Transformers

The OPT 125M--66B models are now available in [Hugging Face Transformers](https://github.com/huggingface/transformers/releases/tag/v4.19.0). You can access them under the `facebook` organization on the [Hugging Face Hub](https://huggingface.co/facebook)

### Using OPT-175B with Alpa

The OPT 125M--175B models are now supported in the [Alpa project](https://alpa-projects.github.io/tutorials/opt_serving.html), which 
enables serving OPT-175B with more flexible parallelisms on older generations of GPUs, such as 40GB A100, V100, T4, M60, etc.

### Using OPT with Colossal-AI

The OPT models are now supported in the [Colossal-AI](https://github.com/hpcaitech/ColossalAI#OPT), which helps users to efficiently and quickly deploy OPT models training and inference, reducing large AI model budgets and scaling down the labor cost of learning and deployment.

### Using OPT with CTranslate2

The OPT 125M--66B models can be executed with [CTranslate2](https://github.com/OpenNMT/CTranslate2/), which is a fast inference engine for Transformer models. The project integrates the [SmoothQuant](https://github.com/mit-han-lab/smoothquant) technique to allow 8-bit quantization of OPT models. See the [usage example](https://opennmt.net/CTranslate2/guides/transformers.html#opt) to get started.

### Using OPT with FasterTransformer

The OPT models can be served with [FasterTransformer](https://github.com/NVIDIA/FasterTransformer), a highly optimized inference framework written and maintained by NVIDIA. We provide instructions to convert OPT checkpoints into FasterTransformer format and [a usage example](docs/faster-transformer.md) with some benchmark results.

### Using OPT with DeepSpeed

The OPT models can be finetuned using [DeepSpeed](https://github.com/microsoft/DeepSpeed). See the [DeepSpeed-Chat example](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) to get started.

## Getting Started in Metaseq
Follow [setup instructions here](docs/setup.md) to get started.

### Documentation on workflows
* [Training](docs/training.md)
* [API](docs/api.md)

### Background Info
* [Background & relationship to fairseq](docs/history.md)
* [Chronicles of training OPT-175B](projects/OPT/chronicles/README.md)

## Support
If you have any questions, bug reports, or feature requests regarding either the codebase or the models released in the projects section, please don't hesitate to post on our [Github Issues page](https://github.com/facebookresearch/metaseq/issues).

Please remember to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Contributing
We welcome PRs from the community!

You can find information about contributing to metaseq in our [Contributing](docs/CONTRIBUTING.md) document.

## The Team
Metaseq is currently maintained by the CODEOWNERS: [Susan Zhang](https://github.com/suchenzang), [Naman Goyal](https://github.com/ngoyal2707), [Punit Singh Koura](https://github.com/punitkoura), [Moya Chen](https://github.com/moyapchen), [Kurt Shuster](https://github.com/klshuster), [David Esiobu](https://github.com/davides), [Igor Molybog](https://github.com/igormolybogFB), [Peter Albert](https://github.com/Xirider), [Andrew Poulton](https://github.com/andrewPoulton), [Nikolay Bashlykov](https://github.com/bashnick), [Binh Tang](https://github.com/tangbinh), [Uriel Singer](https://github.com/urielsinger), [Yuchen Zhang](https://github.com/zycalice), [Armen Aghajanya](https://github.com/ArmenAg), [Lili Yu](https://github.com/lilisierrayu), and [Adam Polyak](https://github.com/adampolyak).

## License

The majority of metaseq is licensed under the MIT license, however portions of the project are available under separate license terms: 
* Megatron-LM is licensed under the [Megatron-LM license](https://github.com/NVIDIA/Megatron-LM/blob/main/LICENSE)

