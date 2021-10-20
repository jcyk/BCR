# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.tasks.translation import TranslationTask

import numpy as np
import os
import logging
from fairseq import utils
from fairseq.data import data_utils
from fairseq.tasks import LegacyFairseqTask, register_task


from . import register_task

logger = logging.getLogger(__name__)


@register_task("translation_from_pretrained_roberta")
class TranslationFromPretrainedRoBERTaTask(TranslationTask):
    """
    Same as TranslationTask except use the MaskedLMDictionary class so that
    we can load data that was binarized with the MaskedLMDictionary class.

    This task should be used for the entire training pipeline when we want to
    train an NMT model from a pretrained XLM checkpoint: binarizing NMT data,
    training NMT with the pretrained XLM checkpoint, and subsequent evaluation
    of that trained model.
    """

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)
        
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0]
            )
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        ###IMPORTANT CHANGE!!!
        """
        src_dict = MaskedLMDictionary.load(
            os.path.join(paths[0], "dict.{}.txt".format(args.source_lang))
        )
        """
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.source_lang))
        )
        src_dict.add_symbol("<mask>")
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)
