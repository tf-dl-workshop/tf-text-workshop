from functools import *
from data.data_creator import *
import itertools
import glob
import os


def get_skip_gram(tokens, k1, k2):
    n_tokens = len(tokens)
    target_list = []
    context_list = []
    for i in range(n_tokens):
        if (i < k1) | ((i + k2 + 1) > len(tokens)):
            continue
        target = tokens[i]
        context = tokens[i - k1:i + k2 + 1]
        context.remove(target)
        for c in context:
            target_list.append(target)
            context_list.append(c)
    return target_list, context_list


if __name__ == "__main__":

    files = glob.glob(os.path.join("data/_BEST/novel", '*.txt'))
    filtered_words = ['', '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
                      ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_']
    novels = []
    for file in files:
        words = []
        lines = open(file, 'r', encoding='utf-8')
        for line in lines:
            line = reduce(lambda a, kv: a.replace(*kv), list(MARKS.items()) + [('\n', ''), ('+', ''), (' ', '')], line)
            word = list(filter(lambda x: x not in filtered_words, line.split("|")))
            words.extend(word)
        novels.append(words)

    import pickle
    from collections import Counter

    all_words = list(itertools.chain.from_iterable(novels))
    word_count_dict = Counter(all_words)
    word_count = [(-c, w) for w, c in word_count_dict.items()]
    word_count.sort()
    all_words = [w for c, w in word_count if word_count_dict[w] > 5]

    indexer = {v: i for i, v in enumerate(all_words)}

    target_list = []
    context_list = []
    for n in novels:
        target, context = get_skip_gram([indexer[w] for w in n if word_count_dict[w] > 5], 4, 4)
        target_list.extend(target)
        context_list.extend(context)

    with open('target_list_novel', 'wb') as fp:
        pickle.dump(target_list, fp)
    with open('context_list_novel', 'wb') as fp:
        pickle.dump(context_list, fp)
    with open('indexer_novel', 'wb') as fp:
        pickle.dump(indexer, fp)
