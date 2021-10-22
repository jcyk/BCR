"""
Format vocabulary by adding extra position markers for seq2seq pointer model
"""

import argparse


def process(vocab_in, vocab_out):
    # 64 pointer pos
    v_outs = []
    for v_in in vocab_in:
        v_in = v_in.strip()
        if "IN:" in v_in or "SL:" in v_in:
            vocab_out.write(v_in + "\n")
        elif "unk" in v_in:
            v_outs.append(v_in.split(" ")[0])

    k = 0
    for v_out in range(len(v_outs)+5):
        vocab_out.write("<unk-{}>".format(v_out) + " 0" + "\n")
        k += 1
    print (k)


def main():
    parser = argparse.ArgumentParser(
        description="Format parse as in Don't Parse Generate paper"
    )
    parser.add_argument(
        "--vocab_in", type=str, help="which split to use",
        default='data/dict.parse.txt.unprocessed'
    )
    parser.add_argument(
        "--vocab_out", type=str, help="which split to use",
        default='data/dict.parse.txt'
    )
    args = parser.parse_args()

    vocab_in = open(args.vocab_in, "r", encoding="utf-8")
    vocab_out = open(args.vocab_out, "w", encoding="utf-8")

    process(vocab_in, vocab_out)

    vocab_in.close()
    vocab_out.close()


if __name__ == "__main__":
    main()
