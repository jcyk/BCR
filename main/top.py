from utils import Statistics, merge_statistics

def is_IN(token):
    return token.startswith("[IN:") or (token.startswith("IN:") and token.endswith("]"))
def is_SL(token):
    return token.startswith("[SL:") or (token.startswith("SL:") and token.endswith("]"))
def is_special(token):
    return is_IN(token) or is_SL(token)
def extract(token):
    if is_IN(token):
        type = "IN"
    else:
        type = "SL"
    if token.startswith("["):
        return token[4:], True, type
    return token[3:-1], False, type

def parse_line(line):
    words = []
    stack = []
    tree = []
    id2node = {}
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
                assert closing_node["value"] == value and closing_node["type"] == typ
                nodes = nodes[:-1]
                node = {
                    "begin":min(x["begin"] for x in nodes),
                    "end":max(x["end"] for x in nodes),
                    "value": value,
                    "type": typ,
                    "head":None,
                    "children":nodes}
                for x in nodes:
                    x["head"] = node
                stack.append(node)
                tree.append(node)
                node["id"] = len(tree)
                id2node[len(tree)] = node
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
            node["id"] = len(tree)
            id2node[len(tree)] = node
    for node in tree:
        node["children"].sort(key=lambda x: (x["begin"], x["end"], x["value"]))
    assert len(stack) == 1
    return {"tree":tree, "words":words, "root":stack[0], "id2node":id2node}

# read trees from modified top format
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


# compare sub-trees
def diff(node_a, node_b):
    if node_a["begin"] != node_b["begin"]:
        return True
    if node_a["end"] != node_b["end"]:
        return True
    if node_a["value"] != node_b["value"]:
        return True
    if len(node_a["children"]) != len(node_b["children"]):
        return True
    for child_a, child_b in zip(node_a["children"], node_b["children"]):
        if diff(child_a, child_b):
            return True
    return False


def size_depth_root_linearization(node):
    size, depth, root = 1, 1, node["value"]
    linearization = "( " + node["value"] + " "
    for child in node["children"]:
        _s, _d, _r, _l = size_depth_root_linearization(child)
        size += _s
        depth = max(depth, 1 +_d)
        if _l:
            linearization += _l + " "  
    linearization += ")"
    if node["type"] == "WD":
        linearization = ""
    return size, depth, root, linearization

def get_span2node(g):
    span2node = {}
    for gnode in g["tree"]:
        if gnode["type"] == "WD":
            continue
        span = (gnode["begin"], gnode["end"])
        if span not in span2node:
            while (gnode["head"] is not None) and \
                gnode["head"]["begin"]==gnode["begin"] and \
                gnode["head"]["end"]==gnode["end"]:
                gnode = gnode["head"]
            span2node[span] = gnode
    return span2node

def seq_distance(a, b):
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    if left < right:
        return 0
    return 1 + min(abs(a[1]- b[0]), abs(a[0]- b[1]))

def compute_dis_to_root(a, id2node):
    a_to_root = [a]
    x = a
    while id2node[x]["head"] != None:
        x = id2node[x]["head"]["id"]
        a_to_root.append(x)
    
    dis_from_a_to = {}
    for i, x in enumerate(a_to_root):
        dis_from_a_to[x] = i
    return dis_from_a_to

def tree_distance(a, b, id2node, ignore_subtree=False):
    dis_from_a_to = compute_dis_to_root(a, id2node)
    dis_from_b_to = compute_dis_to_root(b, id2node)
    if (b in dis_from_a_to):
        if ignore_subtree:
            return 0
        return dis_from_a_to[b]
    if (a in dis_from_b_to):
        if ignore_subtree:
            return 0
        return dis_from_b_to[a]
    
    ret = len(id2node) + 100
    for x in dis_from_a_to:
        if x in dis_from_b_to:
            ret = min(ret, dis_from_a_to[x] + dis_from_b_to[x])
    return ret

# compare two models
def _tree_level(gold, old, new, details=False):
    stats = []
    for g, o, n in zip(gold, old, new):
        stat = Statistics()
        #span
        span2node = get_span2node(g)
        ospan2node = {}
        if o["status"] == "valid":
            ospan2node = get_span2node(o)
        nspan2node = {}
        if n["status"] == "valid":
            nspan2node = get_span2node(n)
        span_nflip = {}
        for span in span2node:
            o_yes = (span in ospan2node) and not diff(ospan2node[span], span2node[span])
            n_yes = (span in nspan2node) and not diff(nspan2node[span], span2node[span])
            span_length = span[1]-span[0]
            stat.update(f"atom", o_yes, n_yes)
            stat.update(f"span#{span_length}", o_yes, n_yes)
            span_nflip[span] = (o_yes and not n_yes)
        if details:
            for xw in span_nflip:
                for yw in span_nflip:
                    if xw == yw:
                        continue
                    seq_dis = seq_distance(xw, yw)
                    stat.update(f"pair_span_seqdis#{seq_dis}", span_nflip[xw], span_nflip[yw])
        
        #complete tree
        o_yes = o["status"]=="valid" and not diff(g["root"], o["root"])
        n_yes = n["status"]=="valid" and not diff(g["root"], n["root"])
        stat.update("whole", o_yes, n_yes)
        # sub-tree
        if details:
            subtree_nflip = {}
            for idx in g["id2node"]:
                gnode = g["id2node"][idx]
                if gnode["type"] == "WD":
                    continue
                size, depth, root, linearization = size_depth_root_linearization(gnode)

                o_yes = False
                if o["status"]=="valid":
                    for onode in o["tree"]:
                        if not diff(onode, gnode):
                            o_yes = True

                n_yes = False
                if n["status"]=="valid":
                    for nnode in n["tree"]:
                        if not diff(nnode, gnode):
                            n_yes = True
                span_length = gnode["end"] - gnode["begin"]
                stat.update(f"tspan#{span_length}", o_yes, n_yes)
                stat.update(f"size#{size}", o_yes, n_yes)
                stat.update(f"depth#{depth}", o_yes, n_yes)
                stat.update(f"{gnode['type']}#{root}", o_yes, n_yes)
                stat.update(f"linearization#{linearization}", o_yes, n_yes)
                subtree_nflip[idx] = (o_yes and not n_yes)
            for xw in subtree_nflip:
                for yw in subtree_nflip:
                    if xw == yw:
                        continue
                    tree_dis = tree_distance(xw, yw, g["id2node"], ignore_subtree=True)
                    stat.update(f"pair_subtree_treedis#{tree_dis}", subtree_nflip[xw], subtree_nflip[yw])
        stats.append(stat)
    return stats

def tree_level(gold, old, new, details=False):
    if isinstance(gold, str):
        gold = read_trees(gold)
    if isinstance(old, str):
        old = read_trees(old)
    if isinstance(new, str):
        new = read_trees(new)
    res = _tree_level(gold, old, new, details=details)
    return res