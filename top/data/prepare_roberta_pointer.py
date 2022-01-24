import os
import argparse
import numpy as np
import copy
import json
from fairseq.data.encoders.gpt2_bpe import get_encoder
from fairseq.models.roberta import RobertaModel

ROBERTA_PATH = "roberta.base"
ENCODER_JSON_PATH = os.path.join(ROBERTA_PATH, "encoder.json")
VOCAB_BPE_PATH = os.path.join(ROBERTA_PATH, "vocab.bpe")

def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def process(en_in, parse_in, en_out, parse_out, remove_unsupported=False):

    bpe_encoder = get_encoder(ENCODER_JSON_PATH, VOCAB_BPE_PATH)

    after_c = 0
    total_c = 0

    max_source_len = 0
    max_target_len = 0
    line_idx = -1
    lines_we_have = []
    for en_line, parse_line in zip(en_in, parse_in):
        line_idx += 1
        try:
            en_line = en_line.strip()
            parse_line = parse_line.strip()

            total_c += 1
            # if top level intent is unsupported then skip
            #if parse_line.startswith("[IN:UNSUPPORTED"):
            if remove_unsupported and "[IN:UNSUPPORTED" in parse_line:
                continue
            en_line_processed = bpe_encoder.encode(en_line)
            en_line_processed_copy = np.array(copy.deepcopy(en_line_processed))

            # find indices of intent/slot tags
            parse_line_split = parse_line.split(" ")
            tag_inds = []
            new_parse_line = ""
            for i, parse_w in enumerate(parse_line_split):
                if ("IN:" in parse_w or "SL:" in parse_w):
                    tag_inds.append(i)
                    new_parse_line += "{} ".format(parse_w)
                else:
                    if new_parse_line.strip().split(" ")[-1] != "text":
                        new_parse_line += "text "

            new_parse_line = new_parse_line.strip()

            parse_line_processed = []
            pointers = []
            first = True
            for i1, i2 in zip(tag_inds[:-1], tag_inds[1:]):
                parse_line_processed.append(parse_line_split[i1])
                new_en_subline = " ".join(parse_line_split[i1+1:i2]).strip()
                if new_en_subline == "":
                    continue

                if not first:
                    new_en_subline = " " + new_en_subline
                first = False

                # BPE it and find that segment in source en text
                new_en_subline_processed = bpe_encoder.encode(new_en_subline)
                new_en_subline_processed = np.array(new_en_subline_processed)

                en_line_processed_copy = np.array(en_line_processed_copy)
                # find the segment in source text
                bool_indices = np.all(rolling_window(en_line_processed_copy, \
                                      len(new_en_subline_processed)) == new_en_subline_processed, axis=1)
                ind = np.mgrid[0:len(bool_indices)][bool_indices]
                assert (len(ind) >= 1)

                ind = ind[0]
                assert ((en_line_processed_copy[ind:ind+len(new_en_subline_processed)] == new_en_subline_processed).all())

                prev_len = len(pointers)
                for pointer_i in range(ind, ind + len(new_en_subline_processed)):
                    pointers.append(pointer_i+1+prev_len)
                    parse_line_processed.append("<unk-{}>".format(pointer_i+1+prev_len))

                en_line_processed_copy = list(en_line_processed_copy)
                for del_l in range(0, len(new_en_subline_processed)):
                    del en_line_processed_copy[ind]

            parse_line_processed.append(parse_line_split[i2])

            # Double check
            reconstr_en_processed = []
            for pointer_i in pointers:
                reconstr_en_processed.append(en_line_processed[pointer_i-1])
            assert (str(bpe_encoder.decode(reconstr_en_processed)) == str(bpe_encoder.decode(en_line_processed)))
            assert (str(bpe_encoder.decode(reconstr_en_processed)) == str(en_line))
        except:
            continue
        lines_we_have.append(line_idx)
        # write them
        en_out.write(" ".join([str(e) for e in en_line_processed])+"\n")
        parse_out.write(" ".join(parse_line_processed)+"\n")

        max_source_len = max(max_source_len, len(en_line_processed))
        max_target_len = max(max_target_len, len(parse_line_processed))

        after_c += 1
    print ("Max src len {}".format(max_source_len))
    print ("Max trg len {}".format(max_target_len))
    print ("Total count {}".format(total_c))
    print ("After removing unsupp {}".format(after_c))
    return lines_we_have

def main():
    parser = argparse.ArgumentParser(
        description="Format parse as in Don't Parse Generate paper"
    )
    parser.add_argument(
        "--dataset", type=str, help="which dir to use",
         default='orig-modif-data'
    )
    parser.add_argument(
        "--split", type=str, help="which split to use",
        default='train'
    )
    parser.add_argument(
        "--out_dir", type=str, help="which dir to save",
        default='data'
    )
    parser.add_argument(
        "--lines", type=str, default=None
    )


    args = parser.parse_args()

    args.parse_in = "{}/{}.parse".format(args.dataset,args.split)
    args.en_in = "{}/{}.en".format(args.dataset,args.split)

    # create out folder
    os.makedirs(args.out_dir, exist_ok=True)

    args.parse_out = "{}/{}.parse".format(args.out_dir, args.split)
    args.en_out = "{}/{}.en".format(args.out_dir, args.split)

    parse_in = open(args.parse_in, "r", encoding="utf-8")
    en_in = open(args.en_in, "r", encoding="utf-8")

    parse_out = open(args.parse_out, "w", encoding="utf-8")
    en_out = open(args.en_out, "w", encoding="utf-8")

    lines_we_have = process(en_in, parse_in, en_out, parse_out, remove_unsupported=True)

    parse_in.close()
    parse_out.close()
    en_in.close()
    en_out.close()
    if args.lines:
        json.dump(lines_we_have, open(args.lines, "w"))
if __name__ == "__main__":
    main()
