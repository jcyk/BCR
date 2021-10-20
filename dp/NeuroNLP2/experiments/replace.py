import sys
fname = sys.argv[1]
gfname = sys.argv[2]
beam = int(sys.argv[3])


def read(fname):
    blocks = []
    block = []
    for line in open(fname).readlines():
        line = line.strip()
        if line:
            block.append(line)
        else:
            if block:
                blocks.append(block)
                block = []
    if block:
        blocks.append(block)
    return blocks

with open(f"{fname}.post", "w") as fo:
    pred = read(f"{fname}")
    gold = read(f"{gfname}")
    assert len(pred) == len(gold) * beam
    for i, g in enumerate(gold):
        pred_i = pred[i*beam:i*beam + beam]
        for p in pred_i:
            assert len(p) == len(g)
            for lg, lp in zip(g, p):
                x = lp.split("\t")
                y = lg.split("\t")
                y[-4] = x[-2]
                y[-3] = x[-1]
                y[-5] = "_"
                y[-2] = "_"
                y[-1] = "_"
                fo.write("\t".join(y)+"\n")
            fo.write("\n")
print (f"{fname}.post")