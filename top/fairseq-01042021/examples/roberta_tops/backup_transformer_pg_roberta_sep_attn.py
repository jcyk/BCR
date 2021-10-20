# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, Optional, List, Tuple

import torch
import os
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from fairseq import metrics, utils
from fairseq import checkpoint_utils
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import MultiheadAttention
from fairseq.models.transformer import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture,
)
from fairseq.models.roberta.model import RobertaEncoder
from torch import Tensor


logger = logging.getLogger(__name__)


@register_model("transformer_pg_roberta")
class TransformerPGRoBERTaModel(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_, augmented with a pointer-generator
    network from `"Get To The Point: Summarization with Pointer-Generator
    Networks" (See et al, 2017) <https://arxiv.org/abs/1704.04368>`_ and
    the RoBERTa encoder.

    Args:
        encoder (TransformerPointerGeneratorEncoder): the encoder
        decoder (TransformerPointerGeneratorDecoder): the decoder

    The Transformer pointer-generator model provides the following named
    architectures and command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_pointer_generator_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        TransformerModel.add_args(parser)
        parser.add_argument('--copy-attention-dropout', type=float, metavar='N', default=0.0,
                            help='copy attention dropout for the pointer mechanism')
        parser.add_argument('--copy-attention-heads', type=int, metavar='N', default=1,
                            help='copy attention dropout for the pointer mechanism')
        parser.add_argument('--source-position-markers', type=int, metavar='N',
                            help='dictionary includes N additional items that '
                                 'represent an OOV token at a particular input '
                                 'position')
        parser.add_argument('--copy-encoder-embed', default='False', type=str, metavar='BOOL',
                            help='copy embeddings of the encoder for pointer indices in the decoder')
        # fmt: on
        # RoBERTa arguments
        parser.add_argument(
            "--pretrained-roberta-checkpoint",
            type=str,
            metavar="STR",
            help="roberta model to use for initializing transformer encoder and/or decoder",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        args.copy_encoder_embed = utils.eval_bool(args.copy_encoder_embed)

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
        if getattr(args, "source_position_markers", None) is None:
            args.source_position_markers = args.max_source_positions

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, is_copy_embedding=False, path=None):
            # The dictionary may include additional items that can be used in
            # place of the normal OOV token and that all map to the same
            # embedding. Using a different token for each input position allows
            # one to restore the word identities from the original source text.
            num_embeddings = len(dictionary) - args.source_position_markers
            padding_idx = dictionary.pad()
            unk_idx = dictionary.unk()

            if is_copy_embedding:
                emb = CopyEmbedding(num_embeddings, args.source_position_markers, embed_dim, padding_idx, unk_idx)
            else:
                emb = Embedding(num_embeddings, args.source_position_markers, embed_dim, padding_idx, unk_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        assert (args.share_all_embeddings == False)

        # regular embedding here
        encoder_embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim, \
                                            src_dict.pad(), src_dict.unk())
        decoder_embed_tokens = build_embedding(
            tgt_dict, args.decoder_embed_dim, is_copy_embedding=args.copy_encoder_embed, path=args.decoder_embed_path
        )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerPGRoBERTaEncoder(args, src_dict)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerPointerGeneratorDecoder(args, tgt_dict, embed_tokens)

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
        subkey = ".".join(key.split('.')[1:])
        if key.endswith("in_proj_weight"):
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
        elif subkey.startswith("lm_head"):
            print ("Not loading {}".format(subkey))
            continue
        else:
            if not subkey in state_dict:
                print("{} not in transformer".format(subkey))
                assert (False)
            state_dict[subkey] = roberta_state_dict[key]
    print ("Done loading Roberta encoder")
    return state_dict


class TransformerPGRoBERTaEncoder(RobertaEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`. The pointer-generator variant adds
    the source tokens to the encoder output as these are otherwise not passed
    to the decoder. Encoder initialized with RoBERTa
    """
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.lm_head = None

        assert hasattr(args, "pretrained_roberta_checkpoint"), (
            "--pretrained-roberta-checkpoint must be specified to load Transformer "
            "encoder from pretrained roberta"
        )

        self.dictionary = dictionary
        roberta_loaded_state_dict = upgrade_state_dict_with_roberta_weights(
            state_dict=self.state_dict(),
            pretrained_roberta_checkpoint=args.pretrained_roberta_checkpoint,
            type='encoder',
        )
        self.load_state_dict(roberta_loaded_state_dict, strict=True)

    def forward(self, src_tokens, src_lengths, **kwargs):
        """
        Runs the `forward()` method of the parent Transformer class. Then adds
        the source tokens into the encoder output tuple.

        While it might be more elegant that the model would pass the source
        tokens to the `forward()` method of the decoder too, this would require
        changes to `SequenceGenerator`.

        Args:
            src_tokens (torch.LongTensor): tokens in the source language of
                shape `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
                - **src_tokens** (Tensor): input token ids of shape
                  `(batch, src_len)`
        """
        # return features of roberta encoder
        encoder_out, encoder_extra = super().forward(src_tokens, features_only=True, return_all_hiddens=True)
        encoder_extra = encoder_extra['inner_states']
        encoder_padding_mask = src_tokens.eq(self.dictionary.pad_index)
        return {
            "encoder_out": [encoder_out.transpose(0,1)],  # T x B x C (done) List
            "encoder_padding_mask": [encoder_padding_mask],  # B x T (done) List
            "encoder_embedding": [encoder_extra[0].transpose(0,1)],  # B x T x C (done) List
            "encoder_states": encoder_extra[1:],  # List[T x B x C] (done)
            "src_tokens": [src_tokens],  # B x T (done)
            "src_lengths": [],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }


class TransformerPointerGeneratorDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`. The pointer-generator variant mixes
    the output probabilities with an attention distribution in the output layer.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)

        # In the pointer-generator model these arguments define the decoder
        # layer and the number of attention heads that will be averaged to
        # create the alignment for pointing.
        self.copy_encoder_embed = args.copy_encoder_embed

        input_embed_dim = embed_tokens.embedding_dim

        # The dictionary may include a separate entry for an OOV token in each
        # input position, so that their identity can be restored from the
        # original source text.
        self.num_types = len(dictionary)
        self.num_oov_types = args.source_position_markers
        self.num_embeddings = self.num_types - self.num_oov_types

        ### Add extra attention head for pointer layer (1 attention head)
        self.copy_attn_layer = MultiheadAttention(
            args.encoder_embed_dim, args.copy_attention_heads, dropout=args.copy_attention_dropout)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (EncoderOut, optional): output from the encoder, used
                for encoder-side attention
            incremental_state (dict, optional): dictionary used for storing
                state during :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False)
            alignment_layer (int, optional): 0-based index of the layer to be
                used for pointing (default: 0)
            alignment_heads (int, optional): number of attention heads to be
                used for pointing (default: 1)

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        # The normal Transformer model doesn't pass the alignment_layer and
        # alignment_heads parameters correctly. We use our local variables.
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            copy_encoder_embed=self.copy_encoder_embed,
        )

        if not features_only:
            # Embedding the tokens again for generation probability prediction,
            # so that we don't have to reimplement the whole extract_features()
            # method.
            x = self.output_layer(x, extra["attn"][0], encoder_out["src_tokens"][0])
        return x, extra

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        copy_encoder_embed: Optional[bool] = False
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        if copy_encoder_embed:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens, encoder_out)
        else:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        _, attn = self.copy_attn_layer(
            query=x,
            key=encoder_out['encoder_out'][0],
            value=encoder_out['encoder_out'][0],
            key_padding_mask=encoder_out['encoder_padding_mask'][0],
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=True,
            need_head_weights=True
        )
        attn = attn.mean(dim=0)
        print (attn)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features, attn, src_tokens, **kwargs):
        """
        Project features to the vocabulary size and mix with the attention
        distributions.
        """

        # project back to size of vocabulary
        logits = super().output_layer(features, **kwargs)

        batch_size = logits.shape[0]
        output_length = logits.shape[1]
        assert logits.shape[2] == self.num_embeddings
        assert src_tokens.shape[0] == batch_size
        src_length = src_tokens.shape[1]

        # MASK BOS AND EOS since we will never attend to it

        # concatenate unnormalized logits and attention weights
        output_logits = torch.cat((logits,attn), 2)

        remaining = output_logits.new_zeros((batch_size, output_length, self.num_types-output_logits.shape[-1]))
        remaining.fill_(float("-inf"))

        output_logits = torch.cat((output_logits, remaining), 2)

        assert (output_logits.shape[2] == self.num_types)

        return F.softmax(output_logits, -1)

    def get_normalized_probs(self, net_output, log_probs, sample):
        """
        Get normalized probabilities (or log probs) from a net's output.
        Pointer-generator network output is already normalized.
        """
        probs = net_output[0]
        # Make sure the probabilities are greater than zero when returning log
        # probabilities.
        return probs.clamp(1e-10, 1.0).log() if log_probs else probs

class CopyEmbedding(nn.Embedding):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.
    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings. This subclass differs from the standard PyTorch Embedding class by
    allowing additional vocabulary entries that will be mapped to the unknown token
    embedding. Also adds functionality which copies the encoder embeddings for pointer indices.
    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int): Pads the output with the embedding vector at :attr:`padding_idx`
                           (initialized to zeros) whenever it encounters the index.
        unk_idx (int): Maps all token indices that are greater than or equal to
                       num_embeddings to this index.
    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`
    Shape:
        - Input: :math:`(*)`, LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`
    .. note::
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's :class:`optim.SGD` (`CUDA` and `CPU`),
        :class:`optim.SparseAdam` (`CUDA` and `CPU`) and :class:`optim.Adagrad` (`CPU`)
    .. note::
        With :attr:`padding_idx` set, the embedding vector at
        :attr:`padding_idx` is initialized to all zeros. However, note that this
        vector can be modified afterwards, e.g., using a customized
        initialization method, and thus changing the vector used to pad the
        output. The gradient for this vector from :class:`~torch.nn.Embedding`
        is always zero.
    """
    __constants__ = ["unk_idx"]

    def __init__(self, num_embeddings, source_position_markers, embedding_dim, padding_idx, unk_idx):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.unk_idx = unk_idx
        self.padding_idx = padding_idx
        nn.init.normal_(self.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(self.weight[padding_idx], 0)

        #self.source_position_markers = source_position_markers + 1 # include pad for it
        self.source_position_markers = source_position_markers

    def forward(self, input, encoder_out):

        mask = input < self.num_embeddings
        input_tokens = torch.where(input >= self.num_embeddings, torch.ones_like(input) * self.padding_idx, input)
        embed_tokens = F.embedding(input_tokens, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

        # source embeddings
        encoder_embed = encoder_out['encoder_embedding'][0] # B x L x embed_dim
        input_ptr = torch.where(input < self.num_embeddings, torch.ones_like(input) * 0, input - self.num_embeddings)
        embed_ptr = torch.gather(encoder_embed, 1,
                     input_ptr.unsqueeze(-1).expand(input_ptr.size(0), input_ptr.size(1), encoder_embed.size(2)))

        mask = mask.unsqueeze(-1).expand(embed_tokens.size()).int()
        embed = mask * embed_tokens + (1-mask) * embed_ptr
        return embed


class Embedding(nn.Embedding):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.
    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings. This subclass differs from the standard PyTorch Embedding class by
    allowing additional vocabulary entries that will be mapped to the unknown token
    embedding.
    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int): Pads the output with the embedding vector at :attr:`padding_idx`
                           (initialized to zeros) whenever it encounters the index.
        unk_idx (int): Maps all token indices that are greater than or equal to
                       num_embeddings to this index.
    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`
    Shape:
        - Input: :math:`(*)`, LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`
    .. note::
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's :class:`optim.SGD` (`CUDA` and `CPU`),
        :class:`optim.SparseAdam` (`CUDA` and `CPU`) and :class:`optim.Adagrad` (`CPU`)
    .. note::
        With :attr:`padding_idx` set, the embedding vector at
        :attr:`padding_idx` is initialized to all zeros. However, note that this
        vector can be modified afterwards, e.g., using a customized
        initialization method, and thus changing the vector used to pad the
        output. The gradient for this vector from :class:`~torch.nn.Embedding`
        is always zero.
    """
    __constants__ = ["unk_idx"]

    def __init__(self, num_embeddings, source_position_markers, embedding_dim, padding_idx, unk_idx):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.unk_idx = unk_idx
        self.padding_idx = padding_idx
        nn.init.normal_(self.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(self.weight[padding_idx], 0)

        #self.source_position_markers = source_position_markers + 1 # include pad for it
        self.source_position_markers = source_position_markers

        self.weight_ptr = Parameter(torch.Tensor(self.source_position_markers, embedding_dim))
        nn.init.normal_(self.weight_ptr, mean=0, std=embedding_dim ** -0.5)

    def forward(self, input):
        mask = input < self.num_embeddings
        input_tokens = torch.where(input >= self.num_embeddings, torch.ones_like(input) * self.padding_idx, input)
        embed_tokens = F.embedding(input_tokens, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

        #input_ptr = torch.where(input < self.num_embeddings, torch.ones_like(input) * 0, input - self.num_embeddings + 1)
        #embed_ptr = F.embedding(input_ptr, self.weight_ptr, 0, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        input_ptr = torch.where(input < self.num_embeddings, torch.ones_like(input) * 0, input - self.num_embeddings)
        embed_ptr = F.embedding(input_ptr, self.weight_ptr, None, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

        mask = mask.unsqueeze(-1).expand(embed_tokens.size()).int()
        embed = mask * embed_tokens + (1-mask) * embed_ptr
        return embed

@register_model_architecture(
    "transformer_pg_roberta", "transformer_pg_roberta_iwslt_deen"
)
def transformer_pg_roberta(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)

    base_architecture(args)

@register_model_architecture(
    "transformer_pg_roberta", "transformer_pg_roberta_iwslt_deen_copysrcembed"
)
def transformer_pg_roberta_copysrcembed(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)

    base_architecture(args)


@register_model_architecture(
    "transformer_pg_roberta", "transformer_pg_roberta_large_iwslt_deen"
)
def transformer_pg_roberta_large(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)

    base_architecture(args)
