
from collections import namedtuple

import numpy as np
import copy
import torch
from fairseq import utils


DecoderOut = namedtuple(
    "IterativeRefinementDecoderOut",
    ["output_tokens", "output_scores", "attn", "step", "max_step", "history"],
)


class EnconlyConvNLUGenerator(object):
    def __init__(
        self,
        src_dict,
        label_dict,
        model,
    ):
        """
        Generates conversational semantic parses using the encoder only architecture.

        Args:
            src_dict: source dictionary
            label_dict: label dictionary
            model: model for inference
        """
        self.bos = src_dict.bos()
        self.pad = src_dict.pad()
        self.unk = src_dict.unk()
        self.eos = src_dict.eos()
        self.vocab_size = len(label_dict)
        self.src_dict = src_dict
        self.label_dict = label_dict
        if type(model) == list:
            self.model = model[0]
        else:
            self.model = model

    def generate_batched_itr(
        self,
        data_itr,
        maxlen_a=None,
        maxlen_b=None,
        cuda=False,
        timer=None,
        prefix_size=0,
    ):
        """Iterate over a batched dataset and yield individual semantic parses.

        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
            timer: StopwatchMeter for timing generations.
        """

        for sample in data_itr:
            if "net_input" not in sample:
                continue
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(
                    self.model,
                    sample
                )
            if timer is not None:
                timer.stop(sample["ntokens"])
            for i, id in enumerate(sample["id"]):
                # remove padding
                src = utils.strip_pad(sample["net_input"]["src_tokens"][i, :], self.pad)
                ref = utils.strip_pad(sample["target"][i, :], self.pad)
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, model, sample, prefix_tokens=None, constraints=None, return_entities=False):
        # ensembling is not supported
        if type(model) == list:
            model = model[0]

        model.eval() # turn-off dropout

        # encoder inputs
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        bsz, src_len = src_tokens.size()

        sent_idxs = torch.arange(bsz)
        finalized = [False for _ in range(bsz)]
        max_iter = src_len * 2
        num_iter = 0

        # saved entities
        if return_entities:
            entities = [[] for _ in range(len(src_lengths))]

        for step in range(max_iter + 1):

            if all(finalized) == True:
                break
            else:
                num_iter += 1

            # forward through the model
            model_outputs, _ = model(src_tokens)

            tag_argmax = model_outputs[0].argmax(axis=-1)
            start_pos_argmax = model_outputs[1].argmax(axis=-1)
            end_pos_argmax = model_outputs[2].argmax(axis=-1)

            ### shift tag indices by the GPT_VOCAB size and insert them
            ### implement with numpy/list for now (not efficient)
            src_tokens_list = src_tokens.cpu().numpy().tolist()
            tag_argmax_list = tag_argmax.cpu().numpy().tolist()
            start_pos_argmax_list = start_pos_argmax.cpu().numpy().tolist()
            end_pos_argmax_list = end_pos_argmax.cpu().numpy().tolist()

            new_src_tokens_list = []

            assert (len(src_tokens_list) == len(tag_argmax_list))
            assert (len(src_tokens_list) == len(start_pos_argmax_list))
            assert (len(src_tokens_list) == len(start_pos_argmax_list))

            for b, (src_tokens_b, tag_b, start_pos_b, end_pos_b) in enumerate(zip(src_tokens_list, tag_argmax_list,\
                                                                 start_pos_argmax_list, end_pos_argmax_list)):
                # remove pad
                if self.src_dict.pad_index in src_tokens_b:
                    src_tokens_b = src_tokens_b[:src_tokens_b.index(self.src_dict.pad_index)]

                ### End of tag symbol; stop decoding
                if tag_b == 4:
                    finalized[b] = True
                    if return_entities:
                        entities[b] = sorted(entities[b])

                if finalized[b]:
                    new_src_tokens_list.append(src_tokens_b)
                else:
                    ### shift tag indices by the GPT_VOCAB size and insert them
                    ### keep in mind that indicies are shifted by 4
                    ### because they contain BOS, PAD, EOS, UNK at the beginning
                    tag_b += model.args.actual_src_dict_size - 4

                    # add 1 because we have BOS symbol at the beginning
                    if start_pos_b > len(src_tokens_b):
                        print ("predicted position greater than length")
                    if end_pos_b > len(src_tokens_b):
                        print ("predicted position greater than length")
                    src_tokens_b.insert(start_pos_b + 1, tag_b)
                    src_tokens_b.insert(end_pos_b + 2, tag_b)

                    # save entities if requested
                    if return_entities:
                        tag_name = self.src_dict[tag_b]
                        tag_span = src_tokens_b[start_pos_b+2:end_pos_b+2]

                        # add valid entities to the list
                        if end_pos_b > start_pos_b:
                            failure_ent = False
                            for tag_x in tag_span:
                                if tag_x > model.args.actual_src_dict_size:
                                    failure_ent = True
                            if not failure_ent:
                                entities[b].append([tag_name, tag_span])

                    new_src_tokens_list.append(src_tokens_b)

            max_b_len = max([len(l) for l in new_src_tokens_list])
            # pad them and put on the GPU for next inference step
            num_tokens = 0
            for i in range(len(new_src_tokens_list)):
                num_tokens += len(new_src_tokens_list[i])
                new_src_tokens_list[i] = new_src_tokens_list[i] + \
                                         [self.src_dict.pad_index] * (max_b_len - len(new_src_tokens_list[i]))

            src_tokens = torch.tensor(np.array(new_src_tokens_list, dtype=np.int)).cuda()

        # return entities here potentially
        if return_entities:
            return {
                "steps": num_iter,
                "tokens": src_tokens,
                "num_tokens": num_tokens,
                "entities": entities
            }
        else:
            return {
                "steps": num_iter,
                "tokens": src_tokens,
                "num_tokens": num_tokens,
            }
