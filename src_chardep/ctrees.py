import collections.abc
import gzip

class TreebankNode(object):
    pass

class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children
        self.children = tuple(children)

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, index=0, nocache=False):
        tree = self
        sublabels = [self.label]

        while len(tree.children) == 1 and isinstance(
                tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label)

        children = []
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right

        return InternalParseNode(tuple(sublabels), children, nocache=nocache)

class LeafTreebankNode(TreebankNode):
    def __init__(self, tag, word):
        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word)

class ParseNode(object):
    pass

class InternalParseNode(ParseNode):
    def __init__(self, label, children, nocache=False):
        assert isinstance(label, tuple)
        assert all(isinstance(sublabel, str) for sublabel in label)
        assert label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, ParseNode) for child in children)
        assert children
        assert len(children) > 1 or isinstance(children[0], LeafParseNode)
        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        self.left = children[0].left
        self.right = children[-1].right

        self.nocache = nocache

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    def enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    def oracle_label(self, left, right):
        enclosing = self.enclosing(left, right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        return ()

    def oracle_splits(self, left, right):
        return [
            child.left
            for child in self.enclosing(left, right).children
            if left < child.left < right
        ]

class LeafParseNode(ParseNode):
    def __init__(self, index, tag, word):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def leaves(self):
        yield self

    def convert(self):
        return LeafTreebankNode(self.tag, self.word)

flag = []

def rmove_c(cur, childs, mod_list):
    child = childs[cur]
    flag[cur] = 1

    for ch in child:
        if flag[ch]==1:
            mod_list.append((cur, ch))
        else:
            rmove_c(ch, childs, mod_list)

    return


def dfs(cur, fa, label, childs, pos_list, words):
    child = childs[cur]
    flag[cur] = 1
    if len(child)== 0:
        return LeafTreebankNode(pos_list[cur - 1], words[cur-1])
    tree_child = []
    f = 0
    for ch in child:
        if flag[ch] == 1:
            continue
        if ch > cur and f == 0:
            f = 1
            tree_child.append(LeafTreebankNode(pos_list[cur - 1], words[cur-1]))
        tree_child.append(dfs(ch, cur, label, childs, pos_list, words))
    if f == 0:
        tree_child.append(LeafTreebankNode(pos_list[cur - 1], words[cur - 1]))

    if fa == 0:
        return InternalTreebankNode(label + '-#', tree_child)
    return InternalTreebankNode('iw#', tree_child)

def load_trees(path, word_dict, char_const_dict, strip_top=True, strip_spmrl_features=True):
    with open(path) as infile:
        treebank = infile.read()

    # Features bounded by `##` may contain spaces, so if we strip the features
    # we need to do so prior to tokenization
    if strip_spmrl_features:
        treebank = "".join(treebank.split("##")[::2])

    tokens = treebank.replace("(", " ( ").replace(")", " ) ").split()

    # XXX(nikita): this should really be passed as an argument
    if 'Hebrew' in path or 'Hungarian' in path or 'Arabic' in path:
        strip_top = False

    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                if len(children) > 0 :
                    trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                if label != '-NONE-':

                    if len(word) > 1 and (label != 'PU' or word.isalnum()):
                        global flag

                        dep_children = [[] for _ in word]
                        dep_children.append([])  # start from 1
                        root = 0
                        if word in word_dict:
                            pos_list = []
                            for idx, (head, charpos)in enumerate(zip(word_dict[word].heads, word_dict[word].postags)):
                                if head == 0 and root == 0:
                                    root_pos = label + "-" + word_dict[word].postags[idx]
                                    pos_list.append(root_pos)
                                    root = idx + 1
                                else:

                                    pos_list.append(charpos)
                                    if head == 0:
                                        dep_children[root].append(idx + 1)
                                    else:
                                        dep_children[head].append(idx + 1)

                            flag = [0 for _ in dep_children]
                            mod_list = []
                            rmove_c(root, dep_children, mod_list)
                            for i in range(1,len(word) + 1):
                                if flag[i] == 0:
                                    rmove_c(i, dep_children, mod_list)
                            for (fa, son) in mod_list:
                                dep_children[root].append(son)

                            dep_children[root].sort()

                        else:
                            for idx in range(2,len(word)+1):
                                dep_children[idx].append(idx-1)
                            root = len(word)
                            pos_list = [ "n" for _ in word[:-1]]
                            root_pos = label + "-" + "n"
                            pos_list.append(root_pos)
                        assert root != 0

                        flag = [0 for _ in dep_children]
                        word_node = dfs(root, 0, label, dep_children, pos_list, word)
                        for i in range(1, len(word) + 1):
                            if flag[i] ==0:
                                print(word_dict[word].heads)
                                print(dep_children)
                            assert flag[i] == 1
                        # if word == '进一步':
                        #     print(word_dict[word].heads)
                        #     print(dep_children)
                        #     print(word_node.linearize())
                        if word not in char_const_dict:
                            char_const_dict[word] = word_node.linearize()
                        trees.append(word_node)
                        # word_children = []
                        # for pos, char in zip(pos_list, word):
                        #     word_children.append(LeafTreebankNode(pos, char))
                        # # trees.append(InternalTreebankNode(root_pos, word_children))
                        # trees.append(InternalTreebankNode(label + "-#", word_children))
                    else:
                        trees.append(LeafTreebankNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)



    # XXX(nikita): this behavior should really be controlled by an argument
    if 'German' in path:
        # Utterances where the root is a terminal symbol break our parser's
        # assumptions, so insert a dummy root node.
        for i, tree in enumerate(trees):
            if isinstance(tree, LeafTreebankNode):
                trees[i] = InternalTreebankNode("VROOT", [tree])

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label in ("TOP", "ROOT"):
                assert len(tree.children) == 1
                trees[i] = tree.children[0]

    def process_NONE(tree):

        if isinstance(tree, LeafTreebankNode):
            label = tree.tag
            if label == '-NONE-':
                return None
            else:
                return tree

        tr = []
        label = tree.label
        if label == '-NONE-':
            return None
        for node in tree.children:
            new_node = process_NONE(node)
            if new_node is not None:
                tr.append(new_node)
        if tr == []:
            return None
        else:
            return InternalTreebankNode(label, tr)

    new_trees = []
    for i, tree in enumerate(trees):
        new_tree = process_NONE(tree)
        new_trees.append(new_tree)

    return new_trees

def load_silver_trees_single(path):
    with gzip.open(path, mode='rt') as f:
        linenum = 0
        for line in f:
            linenum += 1
            tokens = line.replace("(", " ( ").replace(")", " ) ").split()

            def helper(index):
                trees = []

                while index < len(tokens) and tokens[index] == "(":
                    paren_count = 0
                    while tokens[index] == "(":
                        index += 1
                        paren_count += 1

                    label = tokens[index]
                    index += 1

                    if tokens[index] == "(":
                        children, index = helper(index)
                        trees.append(InternalTreebankNode(label, children))
                    else:
                        word = tokens[index]
                        index += 1
                        trees.append(LeafTreebankNode(label, word))

                    while paren_count > 0:
                        assert tokens[index] == ")"
                        index += 1
                        paren_count -= 1

                return trees, index

            trees, index = helper(0)
            assert index == len(tokens)

            assert len(trees) == 1
            tree = trees[0]

            # Strip the root S1 node
            assert tree.label == "S1"
            assert len(tree.children) == 1
            tree = tree.children[0]

            yield tree

def load_silver_trees(path, batch_size):
    batch = []
    for tree in load_silver_trees_single(path):
        batch.append(tree)
        if len(batch) == batch_size:
            yield batch
            batch = []
