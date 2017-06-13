from itertools import groupby
import os

#########
#### Functions used to obtain bracketed parse
#########

def furthest_descendant(i, head, direction):
    if direction == 'left':
        if i not in head[:i]:  # no child to the left
            return i
        else:
            lc = head.index(i)  # leftmost child
            return furthest_descendant(lc, head, 'left')
    else:
        if i not in head[i+1:]:  # no child to the right
            return i
        else:
            rc = len(head) - 1 - head[::-1].index(i)  # rightmost child
            return furthest_descendant(rc, head, 'right')


def bracket(toks, head):
    _toks = toks[:]
    for root in [i for i, x in enumerate(head) if x == 0]:
        _bracket(root, _toks, head)
    return _toks
        

def _bracket(i, toks, head):
    if i not in head:  # token has no dependent
        toks[i] = '[ ' + toks[i] + ' ]'
    else:
        for dep in [idx for idx, x in enumerate(head) if x == i]:
            _bracket(dep, toks, head)
        lmost_dep = furthest_descendant(i, head, direction='left')
        rmost_dep = furthest_descendant(i, head, direction='right')
        toks[lmost_dep] = '[ ' + toks[lmost_dep]
        toks[rmost_dep] = toks[rmost_dep] + ' ]'


#########
#### Functions used to obtain shift-reduce parse
#########

def transparse(toks, heads):
    """
    toks is a list of token forms
    heads is a list giving head index for each position in toks
    """
    parse = []
    _buffer = range(len(toks))
    s = []  # The working stack
    for i in _buffer:
        if heads[i] is not None and heads[i] < 0:  
            # Negative head index means not part of the parse
            continue
        parse.append(toks[i])
        s.append(i)
        _reduce(i, s, parse, heads)
    return parse


def _reduce(i, s, parse, heads):
    """
    i is an int representing how far we are in the buffer
    s is the working stack
    parse is the output string so far
    heads is a list giving head index for each position in toks
    """
    while len(s) > 1:
        # Second condition added so we don't pop a token from stack until
        # all its dependent have been connected to it
        if s[-1] == heads[s[-2]] and s[-2] not in heads[i+1:]:
            parse.append('<<')
            del s[-2]
        elif s[-2] == heads[s[-1]] and s[-1] not in heads[i+1:]:
            parse.append('>>')
            del s[-1]
        else:
            return


#########
#### Different versions of the Tweebank preprocessing script.
#### Differences are pointed out in the doc strings.
#########

def preprocess_tweebank1(in_fp, out_dir):
    """
    Builds the source and target files that OpenNMT expects for training.
    This preprocessing generates training data with the following formats:

    INPUT
    lowercase_token|POS|SPECIAL|BROWN_CLUSTER_4|BROWN_CLUSTER_6|PENN_TB_SCORE

    OUTPUT
    bracketed parse of token forms
    """
    print('loading dataset')
    with open(in_fp) as f:
        train_raw = [x.split('\t') for x in f.read().split('\n')]

    # Get each tweet in the dataset in its own dataframe
    print('building tweet list')
    group_key = lambda l: len(l) > 1
    train = [list(group) for k, group in groupby(train_raw, group_key) if k]

    # aliasing useful indices in tweets
    FORM, POS, HEAD, SPECIAL, BC4, BC6, PTB = 1, 3, 6, 7, 10, 11, 12  
    src_strings = []
    tgt_strings = []
    i = 0

    print('generating input/output sequences')
    for tweet in train:
        i += 1
        print('{}/{} tweets'.format(i, len(train)), end='\r')
        words = []
        # Build the word|feat1|feat2|feat3|... strings
        for tok in tweet: 
            tok_feats = '|'.join([tok[FORM].lower(), tok[POS],  tok[SPECIAL], 
                                  tok[BC4], tok[BC6], tok[PTB]])
            words.append(tok_feats)

        input_str = ' '.join(words)

        # tokens are 1-indexed in the dataset, 
        # adding '/////' root node simplifies everything
        toks = ['/////'] + [x[FORM].lower() for x in tweet]
        head = [None] + [int(x[HEAD]) for x in tweet]
        output_str = ' '.join(bracket(toks, head)[1:])
        src_strings.append(input_str)
        tgt_strings.append(output_str)
    print('\n')

    with open(os.path.join(out_dir, 'src-all'), 'w') as src:
        src.write('\n'.join(src_strings))
    with open(os.path.join(out_dir, 'tgt-all'), 'w') as tgt:
        tgt.write('\n'.join(tgt_strings))


def preprocess_tweebank2(in_fp, out_dir):
    """
    Builds the source and target files that OpenNMT expects for training.
    This preprocessing generates training data with the following formats:

    INPUT
    lowercase_token|POS|SPECIAL

    OUTPUT
    bracketed parse of token forms
    """
    print('loading dataset')
    with open(in_fp) as f:
        train_raw = [x.split('\t') for x in f.read().split('\n')]

    # Get each tweet in the dataset in its own dataframe
    print('building tweet list')
    group_key = lambda l: len(l) > 1
    train = [list(group) for k, group in groupby(train_raw, group_key) if k]

    # aliasing useful indices in tweets
    FORM, POS, HEAD, SPECIAL, BC4, BC6, PTB = 1, 3, 6, 7, 10, 11, 12  
    src_strings = []
    tgt_strings = []
    i = 0

    print('generating input/output sequences')
    for tweet in train:
        i += 1
        print('{}/{} tweets'.format(i, len(train)), end='\r')
        words = []
        # Build the word|feat1|feat2|feat3|... strings
        for tok in tweet: 
            tok_feats = '|'.join([tok[FORM].lower(), tok[POS],  tok[SPECIAL]])
            words.append(tok_feats)

        input_str = ' '.join(words)

        # tokens are 1-indexed in the dataset, 
        # adding '/////' root node simplifies everything
        toks = ['/////'] + [x[FORM].lower() for x in tweet]
        head = [None] + [int(x[HEAD]) for x in tweet]
        output_str = ' '.join(bracket(toks, head)[1:])
        src_strings.append(input_str)
        tgt_strings.append(output_str)
    print('\n')

    with open(os.path.join(out_dir, 'src-all'), 'w') as src:
        src.write('\n'.join(src_strings))
    with open(os.path.join(out_dir, 'tgt-all'), 'w') as tgt:
        tgt.write('\n'.join(tgt_strings))


def preprocess_tweebank3(in_fp, out_dir):
    """
    Builds the source and target files that OpenNMT expects for training.
    This preprocessing generates training data with the following formats:

    INPUT
    lowercase_token|POS

    OUTPUT
    bracketed parse of token POS
    """
    print('loading dataset')
    with open(in_fp) as f:
        train_raw = [x.split('\t') for x in f.read().split('\n')]

    # Get each tweet in the dataset in its own dataframe
    print('building tweet list')
    group_key = lambda l: len(l) > 1
    train = [list(group) for k, group in groupby(train_raw, group_key) if k]

    # aliasing useful indices in tweets
    FORM, POS, HEAD, SPECIAL, BC4, BC6, PTB = 1, 3, 6, 7, 10, 11, 12  
    src_strings = []
    tgt_strings = []
    i = 0

    print('generating input/output sequences')
    for tweet in train:
        i += 1
        print('{}/{} tweets'.format(i, len(train)), end='\r')
        words = []
        # Build the word|feat1|feat2|feat3|... strings
        for tok in tweet: 
            tok_feats = '|'.join([tok[FORM].lower(), tok[POS]])
            words.append(tok_feats)

        input_str = ' '.join(words)

        # tokens are 1-indexed in the dataset, 
        # adding '/////' root node simplifies everything
        toks = ['/////'] + [x[POS] for x in tweet]
        head = [None] + [int(x[HEAD]) for x in tweet]
        output_str = ' '.join(bracket(toks, head)[1:])
        src_strings.append(input_str)
        tgt_strings.append(output_str)
    print('\n')

    with open(os.path.join(out_dir, 'src-all'), 'w') as src:
        src.write('\n'.join(src_strings))
    with open(os.path.join(out_dir, 'tgt-all'), 'w') as tgt:
        tgt.write('\n'.join(tgt_strings))


def preprocess_tweebank4(in_fp, out_dir):
    """
    Builds the source and target files that OpenNMT expects for training.
    This preprocessing generates training data with the following formats:

    INPUT
    lowercase_token|POS

    OUTPUT
    shift-reduce parse of token forms
    """
    print('loading dataset')
    with open(in_fp) as f:
        train_raw = [x.split('\t') for x in f.read().split('\n')]

    # Get each tweet in the dataset in its own dataframe
    print('building tweet list')
    group_key = lambda l: len(l) > 1
    train = [list(group) for k, group in groupby(train_raw, group_key) if k]

    # aliasing useful indices in tweets
    FORM, POS, HEAD, SPECIAL, BC4, BC6, PTB = 1, 3, 6, 7, 10, 11, 12  
    src_strings = []
    tgt_strings = []
    i = 0

    print('generating input/output sequences')
    for tweet in train:
        i += 1
        print('{}/{} tweets'.format(i, len(train)), end='\r')
        words = []
        # Build the word|feat1|feat2|feat3|... strings
        for tok in tweet: 
            tok_feats = '|'.join([tok[FORM].lower(), tok[POS]])
            words.append(tok_feats)

        input_str = ' '.join(words)

        # tokens are 1-indexed in the dataset, 
        # adding '/////' root node simplifies everything
        toks = ['/////'] + [x[FORM].lower() for x in tweet]
        head = [None] + [int(x[HEAD]) for x in tweet]
        output_str = ' '.join(transparse(toks, head))  # Here we use shift-reduce, and include LEFTWALL
        src_strings.append(input_str)
        tgt_strings.append(output_str)
    print('\n')

    with open(os.path.join(out_dir, 'src-all'), 'w') as src:
        src.write('\n'.join(src_strings))
    with open(os.path.join(out_dir, 'tgt-all'), 'w') as tgt:
        tgt.write('\n'.join(tgt_strings))


if __name__ == '__main__':
    input_fp = '/data/hugo/tweets/eng_tweets_min40/all_eng_tweets_min40.conll'
    output_dir = '/data/hugo/tweets/variations/lowercase_br'
    preprocess_tweebank1(input_fp, output_dir)

    input_fp = '/data/hugo/tweets/eng_tweets_min40/all_eng_tweets_min40.conll'
    output_dir = '/data/hugo/tweets/variations/lowercase_pos_br'
    preprocess_tweebank2(input_fp, output_dir)

    input_fp = '/data/hugo/tweets/eng_tweets_min40/all_eng_tweets_min40.conll'
    output_dir = '/data/hugo/tweets/variations/lowercase_pos_br_out_pos'
    preprocess_tweebank3(input_fp, output_dir)

    input_fp = '/data/hugo/tweets/eng_tweets_min40/all_eng_tweets_min40.conll'
    output_dir = '/data/hugo/tweets/variations/lowercase_sr'
    preprocess_tweebank4(input_fp, output_dir)
