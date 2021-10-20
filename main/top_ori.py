def is_special(token):
    return token.startswith("[IN:") or token.startswith("[SL:") or token == "]"
def extract(token):
    open = True
    if token.startswith("[IN:"):
        type = "IN"
    elif token.startswith("[SL:"):
        type = "SL"
    elif token == "]":
        type = None
        open = False
    else:
        raise TypeError
    return token[4:], open, type

def parse_line(line):
    words = []
    stack = []
    tree = []
    for token in line.strip().split():
        if is_special(token):
            value, is_open, typ = extract(token)
            if is_open:
                node = {
                    "begin":None,
                    "end":None,
                    "open":True,
                    "value": value,
                    "type": typ,
                    "head":None,
                    "children":[]}
                stack.append(node)
            else:
                nodes = []
                while True:
                    assert len(stack) > 0
                    closing_node = stack.pop()
                    nodes.append(closing_node)
                    if "open" in closing_node:
                        break
                nodes = nodes[:-1]
                node = {
                    "begin":min(x["begin"] for x in nodes),
                    "end":max(x["end"] for x in nodes),
                    "value": closing_node["value"],
                    "type": closing_node["type"],
                    "head":None,
                    "children":nodes}
                for x in nodes:
                    x["head"] = node
                stack.append(node)
                tree.append(node)
        else:
            node = {"begin":len(words),
                    "end":len(words)+1,
                    "value": token,
                    "type": "WD",
                    "head":None,
                    "children":[]}
            stack.append(node)
            words.append(token)
            tree.append(node)
    for node in tree:
        node["children"].sort(key=lambda x: (x["begin"], x["end"], x["value"]))
    assert len(stack) == 1
    return {"tree":tree, "words":words, "root":stack[0]}

# transform subtokens to tokens
def transform(fpred, fgold, fout):
    from transformers import AutoTokenizer
    toker= AutoTokenizer.from_pretrained("roberta-base")
    acc = 0
    tot = 1
    with open(fout, "w") as fo:
        for l0, l1 in zip(open(fgold).readlines(), open(fpred).readlines()):
            gold = l0.strip().split()
            pred = l1.strip().split()
            cur = 0
            transformed_pred = []
            for token in gold:
                if is_special(token):
                    continue
                for subtoken in toker.tokenize(token):
                    while pred[cur] != subtoken:
                        assert is_special(pred[cur])
                        transformed_pred.append(pred[cur])
                        cur += 1
                    cur += 1
                transformed_pred.append(token)
            transformed_pred.extend(pred[cur:])
            pred = " ".join(transformed_pred)
            gold = " ".join(gold)
            if pred == gold:
                acc += 1
            tot += 1
            fo.write(pred+'\n')
    print (f"acc:{acc/tot*100:.2f}")
    
# read trees from original top format
def read_trees(fname):
    trees = []
    for line in open(fname).readlines():
        line = line.strip()
        try:
            tree = parse_line(line)
            tree["status"] = "valid"
        except:
            tree = {"status":"invalid"}
        tree["actions"] = line.split()
        trees.append(tree)
    return trees

if __name__ == "__main__":
    import sys
    fpred = sys.argv[1]
    fgold = sys.argv[2]
    fout = sys.argv[3]
    transform(fpred, fgold, fout)