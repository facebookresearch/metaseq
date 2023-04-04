# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from metaseq.models import register_model, register_model_architecture
from typing import Optional
from omegaconf import II
from metaseq import utils

try:
    from metaseq.modules.transformer_layer import _ffn
except (ImportError, ModuleNotFoundError):
    try:
        from metaseq.modules import FeedForwardNetwork as _ffn
    except (ImportError, ModuleNotFoundError):
        from metaseq.modules import FeedForward as _ffn

import torch.nn.functional as F

from dataclasses import dataclass, field
from metaseq.modules.dropout import Dropout
from metaseq.modules.linear import Linear
from metaseq.distributed import utils as distributed_utils, fsdp_wrap
from metaseq.dataclass.utils import gen_parser_from_dataclass

from metaseq.models.transformer_decoder import ModelParallelTransformerDecoder
from metaseq.models.transformer_lm import (
    base_lm_architecture,
    transformer_lm_megatron,
    ModelParallelTransformerLanguageModel,
    TransformerLanguageModelConfig,
)

try:
    from megatron.mpu import (
        copy_to_tensor_model_parallel_region,
        gather_from_tensor_model_parallel_region,
    )

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False

import logging

logger = logging.getLogger(__name__)


class FeedForwardNetworkLayer(nn.Module):
    """
    Wrapper for Feed Forward Network layer in the Transformer model
    """

    def __init__(self, args, fc1, fc2, dropout_module=None):
        super().__init__()
        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        self.fc1 = fc1
        self.fc2 = fc2

        self.dropout_module = (
            Dropout(args.dropout, module_name=self.__class__.__name__)
            if not dropout_module
            else dropout_module
        )

    def forward(self, x):
        return _ffn(
            x,
            fc1=self.fc1,
            activation_fn=self.activation_fn,
            fc2=self.fc2,
            dropout_module=self.dropout_module,
        )


DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class DirectorTransformerLanguageModelConfig(TransformerLanguageModelConfig):
    train_mixing_weight: float = field(
        default=0.0,
        metadata={"help": "Exact same init as Megatron"},
    )
    infer_mixing_weight: float = field(
        default=-1,
        metadata={"help": "Exact same init as Megatron"},
    )
    train_gamma: float = field(
        default=0,
        metadata={"help": "Exact same init as Megatron"},
    )
    infer_gamma: float = field(
        default=-1,
        metadata={"help": "Exact same init as Megatron"},
    )
    freeze_decoder: bool = field(
        default=False,
        metadata={"help": "Exact same init as Megatron"},
    )
    use_one_plus_gamma_variant: bool = field(
        default=False,
        metadata={"help": "Exact same init as Megatron"},
    )
    director_classifier_layers: str = field(
        default="linear",
        metadata={"help": "Exact same init as Megatron"},
    )
    massage_and_resave_pt_weights: bool = field(
        default=False,
        metadata={
            "help": "Whether to copy and resave weights or not. If yes then simply load the PT checkpoint with strict = False and resave"
        },
    )
    criterion: str = II("criterion.criterion")
    bf16: bool = II("common.bf16")
    use_sharded_state: Optional[bool] = II("distributed_training.use_sharded_state")


@register_model(
    "model_parallel_director_transformer_lm",
    dataclass=DirectorTransformerLanguageModelConfig,
)
class ModelParallelDirectorTransformerLanguageModel(
    ModelParallelTransformerLanguageModel
):
    def __init__(
        self,
        decoder,
        args,
        classifier_head,
        classifier_ffn: Optional[nn.Module],
        classifier_layers: Optional[nn.ModuleList],
    ):
        super().__init__(decoder)
        self.classifier_head = classifier_head
        self.classifier_ffn = classifier_ffn
        self.classifier_layers = classifier_layers
        assert 0 <= args.train_mixing_weight <= 1, "Invalid train_mixing_weight"
        self.train_mixing_weight = torch.tensor(args.train_mixing_weight).cuda()
        self.infer_mixing_weight = self.train_mixing_weight
        if args.infer_mixing_weight >= 0:
            self.infer_mixing_weight = torch.tensor(args.infer_mixing_weight).cuda()
        self.use_one_plus_gamma_variant = args.use_one_plus_gamma_variant
        self.train_gamma = torch.tensor(args.train_gamma).cuda()
        self.infer_gamma = self.train_gamma
        if args.infer_gamma >= 0:
            self.infer_gamma = torch.tensor(args.infer_gamma).cuda()

        self.freeze_decoder = args.freeze_decoder
        self.director_classifier_layer_choice = args.director_classifier_layers
        self.massage_and_resave_pt_weights = args.massage_and_resave_pt_weights
        self._generating = False

    @staticmethod
    def _layer_floating_point_precision_convertor(layer, args):
        fp16 = getattr(args, "fp16", False)
        memory_efficient_fp16 = getattr(args, "memory_efficient_fp16", False)
        bf16 = getattr(args, "bf16", False)
        with torch.no_grad():
            for _, param in layer.named_parameters():
                param_fp = utils.floating_point_precision_convertor(
                    param.cuda(),
                    fp16=fp16,
                    memory_efficient_fp16=memory_efficient_fp16,
                    bf16=bf16,
                )
                param.copy_(param_fp)
        return layer

    @classmethod
    def build_director_layers(cls, args, decoder, dictionary):
        initialize_params_on_gpu = getattr(
            args, "tensor_parallel_init_model_on_gpu", False
        )
        fp16 = getattr(args, "fp16", False)
        memory_efficient_fp16 = getattr(args, "memory_efficient_fp16", False)
        bf16 = getattr(args, "bf16", False)

        # Notice that classifier_head is NOT vocab parallel (in constrast to the output_project layer in transformer_lm_megatron)
        # TODO classifier_head = Linear(args.decoder_output_dim, vocab_partition_size) and implement vocab_parallel_BCE in lm_and_classification_cross_entropy
        classifier_head = Linear(
            args.decoder_output_dim,
            len(dictionary),
            bias=True,
            initialize_params_on_gpu=initialize_params_on_gpu,
        )
        nn.init.normal_(
            classifier_head.weight, mean=0, std=args.decoder_output_dim**-0.5
        )
        if initialize_params_on_gpu:
            classifier_head = utils.floating_point_precision_convertor(
                classifier_head.cuda(),
                fp16=fp16,
                memory_efficient_fp16=memory_efficient_fp16,
                bf16=bf16,
            )
        classifier_head = fsdp_wrap(
            classifier_head,
            min_num_params=0,
            process_group=distributed_utils.get_data_parallel_group(),
        )
        classifier_ffn = None
        if args.director_classifier_layers == "ffn":
            lm_decoder_layer = decoder.layers[-1]
            # use model_parallel's transformer layer for why fc1 and fc2 are separate methods.
            fc1 = lm_decoder_layer.build_fc1(
                lm_decoder_layer.embed_dim,
                args.decoder_ffn_embed_dim,
                initialize_params_on_gpu=getattr(
                    args, "tensor_parallel_init_model_on_gpu", False
                ),
                full_megatron_init=getattr(args, "full_megatron_init", False),
                megatron_init_sigma=getattr(args, "megatron_init_sigma", 0.006),
                dtype=lm_decoder_layer._get_model_init_dtype(),
            )
            fc2 = lm_decoder_layer.build_fc2(
                lm_decoder_layer.embed_dim,
                args.decoder_ffn_embed_dim,
                initialize_params_on_gpu=getattr(
                    args, "tensor_parallel_init_model_on_gpu", False
                ),
                full_megatron_init=getattr(args, "full_megatron_init", False),
                megatron_init_sigma=getattr(args, "megatron_init_sigma", 0.006),
                num_layers=args.decoder_layers,
                dtype=lm_decoder_layer._get_model_init_dtype(),
            )
            classifier_ffn = FeedForwardNetworkLayer(args, fc1, fc2)

        classifier_layers = None
        layers = []
        num_director_decoder_layers = 0
        if args.director_classifier_layers == "layer":
            num_director_decoder_layers = 1
        elif args.director_classifier_layers == "2layers":
            num_director_decoder_layers = 2
        for i in range(num_director_decoder_layers):
            try:
                lay_i = decoder.build_decoder_layer(
                    args,
                    no_encoder_attn=True,  # TODO comment out after metaseq updated to main
                )
            except:
                lay_i = lay_i = decoder.build_decoder_layer(args)
            layers.append(lay_i)
        if layers:
            classifier_layers = nn.ModuleList(layers)
        return classifier_head, classifier_ffn, classifier_layers

    # def _register_classification_head(
    #     self, name, num_classes=None, inner_dim=None, **kwargs
    # ):
    #     """Register a classification head."""
    #     if name in self.classification_heads:
    #         prev_num_classes = self.classification_heads[name].out_proj.out_features
    #         prev_inner_dim = self.classification_heads[name].dense.out_features
    #         if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
    #             logger.warning(
    #                 're-registering head "{}" with num_classes {} (prev: {}) '
    #                 "and inner_dim {} (prev: {})".format(
    #                     name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
    #                 )
    #             )
    #     self.classification_heads[name] = RobertaClassificationHead(
    #         input_dim=self.args.encoder_embed_dim,
    #         inner_dim=inner_dim or self.args.encoder_embed_dim,
    #         num_classes=num_classes,
    #         activation_fn=self.args.pooler_activation_fn,
    #         pooler_dropout=self.args.pooler_dropout,
    #         q_noise=self.args.quant_noise_pq,
    #         qn_block_size=self.args.quant_noise_pq_block_size,
    #         do_spectral_norm=self.args.spectral_norm_classification_head,
    #     )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        if not has_megatron_submodule:
            raise ImportError(
                "\n\nPlease install megatron using the setup instructions!"
            )

        # make sure all arguments are present in older models
        base_director_lm_config(args)

        task.source_dictionary.pad_to_multiple_(8)
        task.target_dictionary.pad_to_multiple_(8)

        # task.source_dictionary.pad_to_multiple_(args.model_parallel_size * 8)
        # task.target_dictionary.pad_to_multiple_(args.model_parallel_size * 8)

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        embed_tokens = cls.build_embedding(
            args, task.source_dictionary, args.decoder_input_dim
        )
        assert getattr(
            args, "use_sharded_state", False
        ), "Use sharded state must be True for tensor parallel, otherwise model saving and loaded might be broken"

        try:
            decoder = ModelParallelTransformerDecoder(
                args,
                task.target_dictionary,
                embed_tokens,
                no_encoder_attn=True,  # TODO comment out after metaseq updated to main
            )
        except:
            decoder = ModelParallelTransformerDecoder(
                args,
                task.target_dictionary,
                embed_tokens,
            )
        classifier_head, classifier_ffn, classifier_layers = cls.build_director_layers(
            args, decoder, task.target_dictionary
        )
        return cls(decoder, args, classifier_head, classifier_ffn, classifier_layers)

    @classmethod
    def add_args(cls, parser):
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            # keep defaults so that settings defaults
            gen_parser_from_dataclass(parser, dc(), delete_default=True)
        else:
            ModelParallelTransformerLanguageModel.add_args(parser)

    def classifier_output(self, input):
        # Note that the classifier_head is of emb_dim * len(dictionary) thus no need to reduce/gather
        # [From Naman] you just add as nn.Linear(self.output_embed_dim, len(dictionary))
        # and in that case, each tensor parallel gpu has same input and same projection layer weights i.e. they are duplicating the computation, which is okay from loss correctness perspective as we average the gradients, so you will still get the same overall gradients
        clf_output = input
        if self.freeze_decoder:
            clf_output = clf_output.detach()

        if self.director_classifier_layer_choice == "ffn":
            clf_output = self.classifier_ffn(clf_output)
        # TODO elif director_classifier_layer_choice = 1layer, 2layers
        clf_output = self.classifier_head(clf_output)
        # clf_output = gather_from_tensor_model_parallel_region(clf_output).contiguous()
        return clf_output

    def lm_output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        return self.output_layer(features, **kwargs)

    @property
    def generating(self):
        if self._generating:
            assert not self.training
            assert not self.massage_and_resave_pt_weights
        return self._generating

    def set_generation_mode(self):
        self.massage_and_resave_pt_weights = False
        self._generating = True

    def forward(self, src_tokens, **kwargs):
        """
        Run the forward pass for a decoder-only model.

        Feeds a batch of tokens through the decoder to predict the next tokens.

        Args:
            src_tokens (LongTensor): tokens on which to condition the decoder,
                of shape `(batch, tgt_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, seq_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        if self.massage_and_resave_pt_weights:
            self._copy_weights_from_pt_model()
            return None
        features, extra = self.extract_features(src_tokens, **kwargs)
        # features: bsz * seqlen * emb_dim

        lm_output = self.lm_output_layer(features)
        clf_output = self.classifier_output(features)
        if self.generating:
            extra["lm_output"] = lm_output
            mixed_lm_output = self._mixing_weights(lm_output, clf_output)
            return mixed_lm_output, clf_output, features, extra
        else:
            return lm_output, clf_output, features, extra

    def set_infer_mixing_coef(self, infer_mixing_weight=-1, infer_gamma=-1):
        if self.use_one_plus_gamma_variant:
            assert (
                infer_gamma == -1 or infer_gamma >= 0
            ), f"Invalid infer_gamma value {infer_gamma}"
            self.infer_gamma = self.train_gamma
            if infer_gamma >= 0:
                self.infer_gamma = torch.tensor(infer_gamma).cuda()
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f"setting infer_gamma = {self.infer_gamma}")
            return self.infer_gamma
        else:
            assert (
                infer_mixing_weight <= 1
            ), "Invalid infer_mixing_weight value, must fall into [0, 1] or -1 "
            self.infer_mixing_weight = self.train_mixing_weight
            if infer_mixing_weight >= 0:
                self.infer_mixing_weight = torch.tensor(infer_mixing_weight).cuda()
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f"setting infer_mixing_weight = {self.infer_mixing_weight}")
            return self.infer_mixing_weight

    def _mixing_weights(self, lm_output, clf_output):
        classifier_outputs = F.logsigmoid(clf_output)
        log_predictor_scores = F.log_softmax(lm_output, dim=-1)
        if self.use_one_plus_gamma_variant:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f"infer_gamma = {self.infer_gamma}")
            scores = log_predictor_scores + self.infer_gamma * classifier_outputs
        else:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f"infer_mixing_weight = {self.infer_mixing_weight}")
            scores = (
                2 * (1.0 - self.infer_mixing_weight) * log_predictor_scores
                + 2 * self.infer_mixing_weight * classifier_outputs
            )
        return F.log_softmax(scores, dim=-1)

    def _copy_weights_from_pt_model(self):
        if self.director_classifier_layer_choice == "linear":
            pass
        elif self.director_classifier_layer_choice == "ffn":
            pass

    @property
    def state_to_massage(self):
        return ["classifier_head"]


def base_director_lm_config(args):
    args.train_mixing_weight = getattr(args, "train_mixing_weight", 0.0)
    args.infer_mixing_weight = getattr(args, "infer_mixing_weight", -1)
    args.train_gamma = getattr(args, "train_gamma", 0.0)
    args.infer_gamma = getattr(args, "infer_gamma", -1)
    args.freeze_decoder = getattr(args, "freeze_decoder", False)
    args.use_one_plus_gamma_variant = getattr(args, "use_one_plus_gamma_variant", False)
    args.director_classifier_layers = getattr(
        args, "director_classifier_layers", "linear"
    )
    base_lm_architecture(args)


@register_model_architecture(
    "model_parallel_director_transformer_lm", "director_transformer_lm_megatron"
)
def director_transformer_lm_megatron(args):
    transformer_lm_megatron(args)