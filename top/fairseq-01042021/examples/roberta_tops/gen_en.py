with open(f"orig-modif-data/test.en", "w") as fo:
    for line in read(f"orig-modif-data/test.parse"):
        sent = []
        for x in line.split():
            if x.startswith("[IN:") or x.startswith("[SL:") or ( (x.startswith("IN:") or x.startswith("SL:")) and x.endswith(']')):
                continue
            sent.append(x)
        fo.write(' '.join(sent))
        fo.write('\n')