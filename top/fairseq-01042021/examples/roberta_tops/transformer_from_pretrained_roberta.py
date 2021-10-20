# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict

from fairseq import checkpoint_utils
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture as transformer_base_architecture,
)


@register_model("transformer_from_pretrained_roberta")
class TransformerFromPretrainedRobertaModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-roberta-checkpoint",
            type=str,
            metavar="STR",
            help="roberta model to use for initializing transformer encoder and/or decoder",
        )
        parser.add_argument(
            "--init-encoder-only",
            action="store_true",
            help="if set, don't load the roberta weights and embeddings into decoder",
        )
        parser.add_argument(
            "--init-decoder-only",
            action="store_true",
            help="if set, don't load the roberta weights and embeddings into encoder",
        )

    @classmethod
    def build_model(self, args, task, cls_dictionary=MaskedLMDictionary):
        assert hasattr(args, "pretrained_roberta_checkpoint"), (
            "You must specify a path for --pretrained-roberta-checkpoint to use "
            "--arch transformer_from_pretrained_roberta"
        )
        assert isinstance(task.source_dictionary, cls_dictionary) or isinstance(
            task.target_dictionary, cls_dictionary
        ), (
            "You should use a MaskedLMDictionary when using --arch "
            "transformer_from_pretrained_roberta because the pretrained Roberta model "
            "was trained using data binarized with MaskedLMDictionary. "
            "For translation, you may want to use --task "
            "translation_from_pretrained_roberta"
        )
        assert not (
            getattr(args, "init_encoder_only", False)
            and getattr(args, "init_decoder_only", False)
        ), "Only one of --init-encoder-only and --init-decoder-only can be set."
        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoderFromPretrainedRoBERTa(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoderFromPretrainedRoBERTa(args, tgt_dict, embed_tokens)


def upgrade_state_dict_with_roberta_weights(
    state_dict: Dict[str, Any], pretrained_roberta_checkpoint: str, type: str,
) -> Dict[str, Any]:
    """
    Load roberta weights into a Transformer encoder or decoder model.

    Args:
        state_dict: state dict for either TransformerEncoder or
            TransformerDecoder
        pretrained_roberta_checkpoint: checkpoint to load roberta weights from

    Raises:
        AssertionError: If architecture (num layers, attention heads, etc.)
            does not match between the current Transformer encoder or
            decoder and the pretrained_roberta_checkpoint
    """
    if not os.path.exists(pretrained_roberta_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_roberta_checkpoint))

    state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_roberta_checkpoint)
    roberta_state_dict = state["model"]
    for key in roberta_state_dict.keys():

        for search_key in ["embed_tokens", "embed_positions", "layers"]:
            if search_key in key:
                subkey = key[key.find(search_key) :]
                if subkey.endswith("in_proj_weight"):
                    # in_proj_weight used to be q + k + v with same dimensions
                    dim = int(roberta_state_dict[key].shape[0] / 3)
                    state_dict[subkey.replace("in_proj_weight", "q_proj.weight")] = roberta_state_dict[key][:dim]
                    state_dict[subkey.replace("in_proj_weight", "k_proj.weight")] = roberta_state_dict[key][dim: 2 * dim]
                    state_dict[subkey.replace("in_proj_weight", "v_proj.weight")] = roberta_state_dict[key][2 * dim:]
                elif subkey.endswith("in_proj_bias"):
                    dim = int(roberta_state_dict[key].shape[0] / 3)
                    state_dict[subkey.replace("in_proj_bias", "q_proj.bias")] = roberta_state_dict[key][:dim]
                    state_dict[subkey.replace("in_proj_bias", "k_proj.bias")] = roberta_state_dict[key][dim: 2 * dim]
                    state_dict[subkey.replace("in_proj_bias", "v_proj.bias")] = roberta_state_dict[key][2 * dim:]
                else:
                    if not subkey in state_dict:
                        print("{} not in transformer".format(subkey))
                        assert (False)
                    state_dict[subkey] = roberta_state_dict[key]

    return state_dict


class TransformerEncoderFromPretrainedRoBERTa(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if getattr(args, "init_decoder_only", False):
            # Don't load roberta weights for encoder if --init-decoder-only
            return

        assert hasattr(args, "pretrained_roberta_checkpoint"), (
            "--pretrained-roberta-checkpoint must be specified to load Transformer "
            "encoder from pretrained roberta"
        )
        roberta_loaded_state_dict = upgrade_state_dict_with_roberta_weights(
            state_dict=self.state_dict(),
            pretrained_roberta_checkpoint=args.pretrained_roberta_checkpoint,
            type='encoder',
        )
        self.load_state_dict(roberta_loaded_state_dict, strict=True)


class TransformerDecoderFromPretrainedRoBERTa(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if getattr(args, "init_encoder_only", False):
            # Don't load roberta weights for decoder if --init-encoder-only
            return
        assert hasattr(args, "pretrained_roberta_checkpoint"), (
            "--pretrained-roberta-checkpoint must be specified to load Transformer "
            "decoder from pretrained roberta"
        )

        roberta_loaded_state_dict = upgrade_state_dict_with_roberta_weights(
            state_dict=self.state_dict(),
            pretrained_roberta_checkpoint=args.pretrained_roberta_checkpoint,
            type='decoder',
        )
        self.load_state_dict(roberta_loaded_state_dict, strict=True)


@register_model_architecture(
    "transformer_from_pretrained_roberta", "transformer_from_pretrained_roberta"
)
def transformer_from_pretrained_roberta(args):
    transformer_base_architecture(args)


@register_model_architecture(
    "transformer_from_pretrained_roberta", "transformer_roberta_encoder_iwslt_de_en"
)
def transformer_roberta_encoder_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    transformer_base_architecture(args)
