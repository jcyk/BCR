from utils import Statistics
# read trees from conllx format
def read_trees(fname):
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
    
    trees = []
    for block in blocks:
        tree = {}
        root = None
        for line in block:
            info = line.split()
            idx, word, pos, head, rel = info[0], info[1], info[4], info[6], info[7]
            tree[idx] = {"word":word,
                        "pos":pos,
                        "head":head,
                        "rel":rel,
                        "labeled_head":(rel, head),
                        "children":[],
                        "labeled_children":[]}
        for nidx in tree:
            node = tree[nidx]
            hidx = node["head"]
            if hidx == "0":
                if root is None:
                    root = nidx
                else:
                    root = "MULTIPLE ROOTS"
                continue
            try:
                tree[hidx]["children"].append(nidx)
            except:
                print (block)
                
            tree[hidx]["labeled_children"].append((node["rel"], nidx))
        
        for nidx in tree:
            tree[nidx]["children"].sort(key=lambda x: int(x)) # sort by order in the sentence
            tree[nidx]["labeled_children"].sort(key=lambda x: int(x[1])) # sort by order in the sentence
        words = []
        tags = []
        for i in range(1, 1+len(tree)):
            str_i = f"{i}"
            if str_i in tree:
                words.append(tree[str_i]["word"])
                tags.append(tree[str_i]["pos"])
            else:
                break
        meta = {"tree": tree, "root": root, "words": words, "tags": tags}
        trees.append(meta)
    return trees

def print_tree(nidx, tree):
    head = tree[nidx]['word']
    added_root = False
    linearization = "( "
    for child in tree[nidx]["children"]:
        _l = print_tree(child, tree)
        if int(child) > int(nidx) and not added_root:
            linearization += tree[nidx]['word'] + " "
            added_root = True
        linearization += _l + " "
    if not added_root:
        linearization += tree[nidx]['word'] + " "
    linearization += ")"
    return linearization

# compare sub-trees
def diff_u(nidx, tree_a, tree_b):
    if tree_a[nidx]["children"] != tree_b[nidx]["children"]:
        return True
    for child in tree_a[nidx]["children"]:
        if diff_u(child, tree_a, tree_b):
            return True
    return False

def diff_l(nidx, tree_a, tree_b):
    if tree_a[nidx]["labeled_children"] != tree_b[nidx]["labeled_children"]:
        return True
    for child in tree_a[nidx]["children"]:
        if diff_l(child, tree_a, tree_b):
            return True
    return False

def size_depth_pos_linearization_u(nidx, tree):
    size, depth, pos = 1, 1, tree[nidx]['pos']
    added_root = False
    linearization = "( "
    for child in tree[nidx]["children"]:
        _s, _d, _p, _l = size_depth_pos_linearization_u(child, tree)
        size += _s
        depth = max(depth, 1 +_d)
        if int(child) > int(nidx) and not added_root:
            linearization += tree[nidx]['pos'] + " "
            added_root = True
        linearization += _l + " "
    if not added_root:
        linearization += tree[nidx]['pos'] + " "
    linearization += ")"
    return size, depth, pos, linearization

def size_depth_pos_linearization_l(nidx, tree):
    size, depth, pos = 1, 1, tree[nidx]['pos']
    added_root = False
    linearization = "( "
    for rel, child in tree[nidx]["labeled_children"]:
        _s, _d, _p, _l = size_depth_pos_linearization_l(child, tree)
        size += _s
        depth = max(depth, 1 +_d)
        if int(child) > int(nidx) and not added_root:
            linearization += tree[nidx]['pos'] + " "
            added_root = True
        linearization += "(:"+rel + _l[1:] + " "
    if not added_root:
        linearization += tree[nidx]['pos'] + " "
    linearization += ")"
    return size, depth, pos, linearization

def compute_dis_to_root(a, tree):
    a_to_root = [a]
    x = a
    while tree[x]["head"] != "0":
        x = tree[x]["head"]
        a_to_root.append(x)
    
    dis_from_a_to = {}
    for i, x in enumerate(a_to_root):
        dis_from_a_to[x] = i
    return dis_from_a_to

def tree_distance(a, b, tree, ignore_subtree=False):
    dis_from_a_to = compute_dis_to_root(a, tree)
    dis_from_b_to = compute_dis_to_root(b, tree)
    if (b in dis_from_a_to):
        if ignore_subtree:
            return 0
        return dis_from_a_to[b]
    if (a in dis_from_b_to):
        if ignore_subtree:
            return 0
        return dis_from_b_to[a]
    
    ret = len(tree) + 100
    for x in dis_from_a_to:
        if x in dis_from_b_to:
            ret = min(ret, dis_from_a_to[x] + dis_from_b_to[x])
    return ret

# compare two models
def _tree_level(gold, old, new, label=True, details=False):
    diff = diff_l if label else diff_u
    size_depth_pos_linearization = size_depth_pos_linearization_l if label else size_depth_pos_linearization_u
    stats = []
    for g, o, n in zip(gold, old, new):
        stat = Statistics()
        #word-level
        assert len(g["tree"]) == len(o["tree"]) == len(n["tree"])
        
        word_nflip = {}
        for i in range(1, 1+len(g["tree"])):
            str_i = f"{i}"
            pos = g["tree"][str_i]["pos"]
            if pos == "SYM" or pos == "PUNCT":
                continue
            str_head = "labeled_head" if label else "head"
            o_yes = g["tree"][str_i][str_head] ==  o["tree"][str_i][str_head]
            n_yes = g["tree"][str_i][str_head] ==  n["tree"][str_i][str_head]
            dis = abs(int(str_i)-int(g["tree"][str_i]["head"]))
            rel = g["tree"][str_i]["rel"]
            head = g["tree"][str_i]["head"]
            if head == "0":
                headpos = "ROOT"
            else:
                headpos = g["tree"][head]["pos"]
            stat.update(f"word", o_yes, n_yes)
            stat.update(f"pos#{pos}", o_yes, n_yes)
            stat.update(f"dis#{dis}",o_yes, n_yes)
            stat.update(f"rel#{rel}", o_yes, n_yes)
            stat.update(f"headpos#{headpos}", o_yes, n_yes)
            # negative flip happens
            word_nflip[str_i] = (o_yes and not n_yes)
        if details:
            for xw in word_nflip:
                for yw in word_nflip:
                    if xw == yw:
                        continue
                    seq_dis = abs(int(xw)-int(yw))
                    tree_dis = tree_distance(xw, yw, g["tree"])
                    if tree_dis == 1:
                        assert g["tree"][xw]["head"] == yw or g["tree"][yw]["head"] == xw
                        if g["tree"][xw]["head"] == yw:
                            rel = g["tree"][xw]["rel"]
                        else:
                             rel = g["tree"][yw]["rel"]
                        stat.update(f"pair_token_dep#{rel}", word_nflip[xw], word_nflip[yw])
                    stat.update(f"pair_token_seqdis#{seq_dis}", word_nflip[xw], word_nflip[yw])
                    stat.update(f"pair_token_treedis#{tree_dis}", word_nflip[xw], word_nflip[yw])
                
                
        #complete tree
        o_yes = not diff(g["root"], g["tree"], o["tree"])
        n_yes = not diff(g["root"], g["tree"], n["tree"])
        stat.update("whole", o_yes, n_yes)
        # sub-tree
        if details:
            subtree_nflip = {}
            for nidx in g["tree"]:
                if nidx == "root":
                    continue
                size, depth, pos, linearization = size_depth_pos_linearization(nidx, g["tree"])
                if size == 1:
                    continue
                o_yes = not diff(nidx, g["tree"], o["tree"])
                n_yes = not diff(nidx, g["tree"], n["tree"])
                stat.update(f"size#{size}", o_yes, n_yes)
                stat.update(f"depth#{depth}", o_yes, n_yes)
                stat.update(f"rootpos#{pos}", o_yes, n_yes)
                stat.update(f"linearization#{linearization}", o_yes, n_yes)
                subtree_nflip[nidx] = (o_yes and not n_yes)

            for xw in subtree_nflip:
                for yw in subtree_nflip:
                    if xw == yw:
                        continue
                    tree_dis = tree_distance(xw, yw, g["tree"], ignore_subtree=True)
                    stat.update(f"pair_subtree_treedis#{tree_dis}", subtree_nflip[xw], subtree_nflip[yw])
        stats.append(stat)
    return stats

def tree_level(gold, old, new, scheme='unlabeled', details=False):
    assert scheme in ['labeled', 'unlabeled', 'both']
    if isinstance(gold, str):
        gold = read_trees(gold)
    if isinstance(old, str):
        old = read_trees(old)
    if isinstance(new, str):
        new = read_trees(new)
    if scheme in ['labeled', 'both']:
        res_l = _tree_level(gold, old, new, details=details)
    if scheme in ['unlabeled', 'both']:
        res_u = _tree_level(gold, old, new, label=False, details=details)
        
    if scheme == 'both':
        return res_l, res_u
    elif scheme == 'labeled':
        return res_l
    elif scheme == 'unlabeled':
        return res_u
        