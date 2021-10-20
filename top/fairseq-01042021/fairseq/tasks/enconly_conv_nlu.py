# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
import torch
from fairseq import utils, metrics
from fairseq.data import (
    ConcatSentencesDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    AppendTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    RollDataset,
    SortDataset,
    StripTokenDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.enconly_conv_nlu_generator import EnconlyConvNLUGenerator


logger = logging.getLogger(__name__)

@register_task("enconly_conv_nlu")
class EnconlyConvNLU(LegacyFairseqTask):
    """
    Conversational NLU task with encoder-only model
    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument(
            "--num-classes",
            type=int,
            default=-1,
            help="number of classes or regression targets",
        )
        parser.add_argument(
            "--init-token",
            type=int,
            default=0,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument(
            "--num-tags",
            type=int,
            default=-1,
            help="number of tags (intent/slot) tags"
        )

        parser.add_argument("--no-shuffle", action="store_true", default=False)
        parser.add_argument(
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )
        parser.add_argument(
            "--shorten-data-split-list",
            default="",
            help="comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',
        )

    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary
        assert (args.num_tags != -1)
        self.num_tags = args.num_tags
        if hasattr(args, "max_positions"):
            self._max_positions = args.max_positions
        else:
            args.max_positions = 512
            self._max_positions = args.max_positions
        self.ignore_index = None
        args.tokens_per_sample = self._max_positions

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "dict.input.txt"),
            source=True,
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        # load label dictionary
        label_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "dict.label.txt"),
            source=False,
        )
        logger.info("[label] dictionary: {} types".format(len(label_dict)))
        return cls(args, data_dict, label_dict)

    def valid_step(self, sample, model, criterion, subset=None):
        if "inference" not in subset:
            loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
            return loss, sample_size, logging_output
        else:

            def decode(toks):
                s = self.source_dictionary.string(
                    toks.int().cpu())
                return s

            gen_out = self.inference_step(self.sequence_generator, [model], sample, prefix_tokens=None)
            # calculate exact match
            hyps, refs = [], []
            logging_output = {}
            num_total = 0
            num_exact_matches = 0
            assert (gen_out["tokens"].shape[0] == len(sample["id"]))
            num_total = len(sample["id"])
            for i in range(num_total):
                hyps.append(
                    decode(
                        utils.strip_pad(gen_out["tokens"][i], self.source_dictionary.pad())
                    )
                )

                refs.append(
                    decode(
                        utils.strip_pad(sample["target"][i], self.source_dictionary.pad())
                    )
                )
                if refs[-1] == hyps[-1]:
                    num_exact_matches += 1

            logging_output["exact_match"] = num_exact_matches
            logging_output["instance_count"] = num_total
            logging_output["nsentences"] = len(sample["id"])
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
            loss = torch.tensor(0.0).cuda()
            sample_size = 1
            return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion, subset=None):
        if subset is None or ("inference" not in subset):
            super().reduce_metrics(logging_outputs, criterion)
        else:
            def sum_logs(key):
                out = sum(log.get(key, 0) for log in logging_outputs)
                if type(out) == torch.Tensor:
                    out = out.cpu()
                return out


            exact_match = sum_logs("exact_match")
            instance_count = sum_logs("instance_count")
            if instance_count > 0:
                metrics.log_scalar("_em", np.array(exact_match))
                metrics.log_scalar("_instance_count", np.array(instance_count))

                def compute_instance_count(meters):
                    return meters["_instance_count"].sum

                def compute_em(meters):
                    return float(meters["_em"].sum) / float(meters["_instance_count"].sum)

                metrics.log_derived("instance_count", compute_instance_count)
                metrics.log_derived("em", compute_em)


    def load_dataset(self, split, combine=False, forgen=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        # if inference in split name then the dataset used for generation
        if "inference" in split:
            forgen = True

        src_tokens = data_utils.load_indexed_dataset(
            os.path.join(self.args.data, "{}.input-label.input".format(split)),
            self.source_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        assert src_tokens is not None

        if self.args.init_token is not None:
            src_tokens = PrependTokenDataset(src_tokens, self.args.init_token)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        # don't shorten if you are using model for generation
        if not forgen:
            src_tokens = maybe_shorten_dataset(
                src_tokens,
                split,
                self.args.shorten_data_split_list,
                self.args.shorten_method,
                self.args.max_positions,
                self.args.seed,
            )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        if not forgen:
            label_dataset = data_utils.load_indexed_dataset(
                os.path.join(self.args.data, "{}.input-label.label".format(split)),
                self.label_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )

            assert (label_dataset is not None)
            # remove the final <eos> and offset the pointer ids by (num_tags)+4(n_special)
            offset_column = [1,2]
            dataset.update(
                target=OffsetTokensDataset(
                    StripTokenDataset(
                        label_dataset,
                        id_to_strip=self.label_dictionary.eos(),
                        right_strip_only=True,
                    ),
                    offset=-(self.num_tags+self.label_dictionary.nspecial),
                    offset_column=offset_column,
                )
            )
        else:
            logger.info("gen model for enconly_convnlu_task")
            self.args.no_shuffle = True

            label_tokens = data_utils.load_indexed_dataset(
                os.path.join(self.args.data, "{}.input-label.label".format(split)),
                self.source_dictionary,  # use source dict here because for gen dict is source dict
                self.args.dataset_impl,
                combine=combine,
            )
            assert label_tokens is not None

            if self.args.init_token is not None:
                label_tokens = PrependTokenDataset(label_tokens, self.args.init_token)

            dataset.update(target=RightPadDataset(label_tokens,pad_idx=self.source_dictionary.pad()))

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if self.args.no_shuffle:
            logger.info("not shuffling data")
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))
        self.datasets[split] = dataset
        return self.datasets[split]

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if self.args.init_token is not None:
            src_tokens = PrependTokenDataset(src_tokens, self.args.init_token)

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        return dataset

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None, return_entities=False
    ):
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints, return_entities=return_entities
            )

    def build_model(self, args):
        model = super().build_model(args)

        # TODO create sequence generator for the decoding
        self.sequence_generator = EnconlyConvNLUGenerator(self.source_dictionary, self.label_dictionary, \
                                                          model)

        return model

    def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None):
        ### Doesn't support ensembling
        return EnconlyConvNLUGenerator(self.source_dictionary, self.label_dictionary, \
                                                          models[0])

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
