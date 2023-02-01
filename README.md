
# Metaseq
A codebase for working with [Open Pre-trained Transformers](projects/OPT).


## Community Integrations

### Using OPT with 🤗 Transformers

The OPT 125M--66B models are now available in [Hugging Face Transformers](https://github.com/huggingface/transformers/releases/tag/v4.19.0). You can access them under the `facebook` organization on the [Hugging Face Hub](https://huggingface.co/facebook)

### Using OPT-175B with Alpa

The OPT 125M--175B models are now supported in the [Alpa project](https://alpa-projects.github.io/tutorials/opt_serving.html), which 
enables serving OPT-175B with more flexible parallelisms on older generations of GPUs, such as 40GB A100, V100, T4, M60, etc.

### Using OPT with Colossal-AI

The OPT models are now supported in the [Colossal-AI](https://github.com/hpcaitech/ColossalAI#OPT), which helps users to efficiently and quickly deploy OPT models training and inference, reducing large AI model budgets and scaling down the labor cost of learning and deployment.

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
Metaseq is currently maintained by the CODEOWNERS: [Susan Zhang](https://github.com/suchenzang), [Naman Goyal](https://github.com/ngoyal2707), [Punit Singh Koura](https://github.com/punitkoura), [Moya Chen](https://github.com/moyapchen), [Kurt Shuster](https://github.com/klshuster), [Ruan Silva](https://github.com/ruanslv), [David Esiobu](https://github.com/davides), [Igor Molybog](https://github.com/igormolybogFB), [Peter Albert](https://github.com/Xirider), [Sharan Narang](https://github.com/sharannarang), and [Andrew Poulton](https://github.com/andrewPoulton).

Previous maintainers include:
[Stephen Roller](https://github.com/stephenroller), [Anjali Sridhar](https://github.com/anj-s), [Christopher Dewan](https://github.com/m3rlin45).


## License

The majority of metaseq is licensed under the MIT license, however portions of the project are available under separate license terms: 
* Megatron-LM is licensed under the [Megatron-LM license](https://github.com/NVIDIA/Megatron-LM/blob/main/LICENSE)

