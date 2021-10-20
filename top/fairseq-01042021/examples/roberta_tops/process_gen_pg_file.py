import argparse
import pdb
import os
import numpy as np
import pdb
import re

from fairseq.data.encoders.gpt2_bpe import get_encoder


ROBERTA_PATH = "/efs/dengcai/top/data/roberta.base"
ENCODER_JSON_PATH = os.path.join(ROBERTA_PATH, "encoder.json")
VOCAB_BPE_PATH = os.path.join(ROBERTA_PATH, "vocab.bpe")


parser = argparse.ArgumentParser(description='Process some integers.')

#parser.add_argument('--srcfile', type=str, default='/home/ubuntu/workspace/src/top-dataset-only-01042021/seq2seq-roberta-pointer-fixed/test.en')
#parser.add_argument('--file', type=str, default='/home/ubuntu/checkpoints/TEST-8gpu-drop0.3-attndrop0.1-attmpt2/test_beam1.parse.unprocessed.maxtokens4096')
#parser.add_argument('--outfile', type=str, default='/home/ubuntu/checkpoints/TEST-8gpu-drop0.3-attndrop0.1-attmpt2/test_beam1.parse.maxtokens4096')

#parser.add_argument('--srcfile', type=str, default='/efs/exp/mansimov/code/fastfood-dmr-splits/seq2seq-roberta-pointer-fixed/test.en')
parser.add_argument('--file', type=str, default='/efs/dengcai/top/checkpoints/base.seed123/test.parse.unprocessed')
parser.add_argument('--outfile', type=str, default='/efs/dengcai/top/checkpoints/base.seed123/test.parse')


if __name__ == '__main__':
    args = parser.parse_args()
    bpe_encoder = get_encoder(ENCODER_JSON_PATH, VOCAB_BPE_PATH)

    # open the tsv file
    with open(args.file, 'r') as f:
        f_lines = f.readlines()

    #with open(args.srcfile, 'r') as f:
    #    src_lines = f.readlines()

    src_hyps = {}
    unp_trg_hyps = {}
    unp_gen_hyps = {}
    parse_hyps = {}
    parse_scores = {}


    for l in f_lines:
        if l.startswith("S-"):
            l = l.strip()
            l = l.split("\t")
            h_num = int(l[0].split('-')[1])
            h_out = l[1].strip()
            src_hyps[h_num] = h_out
            #src_hyps[h_num] = src_lines[h_num]
        elif l.startswith("T-"):
            l = l.strip()
            l = l.split("\t")
            h_num = int(l[0].split('-')[1])
            assert h_num in src_hyps
            unp_trg_hyps[h_num] = l[1].strip()

        elif l.startswith("H-"):
            l = l.strip()
            l = l.split("\t")
            h_num = int(l[0].split('-')[1])
            h_score = float(l[1].strip())
            assert h_num in src_hyps
            parse = []
            if h_num not in unp_gen_hyps:
                unp_gen_hyps[h_num] = []
            unp_gen_hyps[h_num].append(l[2].strip())
            
            if h_num not in parse_scores:
                parse_scores[h_num] = []
            parse_scores[h_num].append(h_score) 
            
            stack = []
            #for unprocess_p in unp_trg_hyps[h_num].strip().split(" "):
            for unprocess_p in l[2].strip().split(" "):
                #if "SL:" in unprocess_p or "IN:" in unprocess_p:
                if not "<unk-" in unprocess_p:
                    if len(stack) > 0:
                        parse.append(bpe_encoder.decode(stack).strip())
                        stack = []
                    parse.append(unprocess_p)
                else:
                    unk_id = unprocess_p[1:-1]
                    unk_id = int(unk_id.split("-")[1])
                    _tot_unks = src_hyps[h_num].split(' ')
                    if unk_id-1 >=len(_tot_unks):
                        print ("invalid unk id:", unk_id, len(_tot_unks), _tot_unks)
                    else:
                        stack.append(int(src_hyps[h_num].split(' ')[unk_id - 1]))

            if len(stack) > 0:
                parse.append(bpe_encoder.decode(stack).strip())
                stack = []

            parse = " ".join(parse).strip()
            parse = re.sub('\s+', ' ', parse).strip()
            #print (parse)
            if h_num not in parse_hyps:
                parse_hyps[h_num] = []
            parse_hyps[h_num].append(parse)

    
    exact_match = 0
    print (len(unp_gen_hyps), len(unp_trg_hyps))
    with open(args.outfile, "w") as f_out, \
         open(args.outfile+".preprocessed.en", "w") as f_en, \
         open(args.outfile+".preprocessed.parse", "w") as f_parse, \
         open(args.outfile+".preprocessed.score", "w") as f_score:
        for h_num in range(0,max(list(parse_hyps.keys()))+1):
            scores = np.array(parse_scores[h_num])
            idx = np.argmax(scores)
            if str(unp_gen_hyps[h_num][idx]) == str(unp_trg_hyps[h_num]):
                exact_match += 1
            for idx in np.argsort(-scores):
                f_out.write(parse_hyps[h_num][idx]+"\n")
                f_en.write(src_hyps[h_num]+"\n")
                f_parse.write(unp_gen_hyps[h_num][idx]+"\n")
                f_score.write(f"{scores[idx]}\n")
                
    #print (exact_match)
    #print (h_num+1)
    print ("Exact match: {:.2f}".format(100*exact_match/(h_num+1)))
    #print ("Done")
