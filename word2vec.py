import math
import pickle
import random
import sys
from collections import Counter, OrderedDict
from enum import Enum
from typing import List, Optional, Generator, Dict, Tuple

import numpy as np
import torchtext
from dataclasses import dataclass
from torchtext.vocab import Vocab
from tqdm import tqdm


@dataclass
class VocabWord:
    count: int
    word: str
    codelen: int
    point: List[int]
    code: List[int]


class PredictMode(Enum):
    HierarchicalSoftmax = 0
    NegativeSampling = 1


class ModelVariant(Enum):
    CBOW = 0
    SkipGram = 1


def read_tokens_from_file(file) -> Generator[str, None, None]:
    token = ''
    char = ' '
    with open(file, 'r') as f:
        while char:
            char = f.read(1)
            if char == ' ':
                if token != '':
                    yield token
                    token = ''
            else:
                token += char


def read_word_indices_from_file(file, vocab: Vocab) -> Generator[Optional[int], None, None]:
    vocab_dict = vocab.get_stoi()
    for word in read_tokens_from_file(file):
        word_index = vocab_dict.get(word, -1)
        yield word_index
    yield None


def build_vocab(train_file, min_word_freq, special_tokens) -> Tuple[Vocab, List[VocabWord], Dict[str, int]]:
    counter = Counter()

    for token in read_tokens_from_file(train_file):
        counter.update([token])

    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    vocab_words = [(word, count) for word, count in sorted_items if count >= min_word_freq]

    for s in reversed(special_tokens):
        vocab_words.insert(0, (s, min_word_freq))

    vocab_words = [VocabWord(count, word, 0, [], []) for word, count in vocab_words]
    word_indices = {word: i for i, (word, count) in enumerate(vocab_words)}

    ordered_dict = OrderedDict(vocab_words)

    vocab = torchtext.vocab.vocab(ordered_dict, min_freq=min_word_freq)

    return vocab, vocab_words, word_indices


def init_unigram_table(vocab_words: List[VocabWord], vocab_size: int) -> np.ndarray:
    table_size = int(1e8)
    table = np.zeros(table_size, dtype=int)
    train_words_pow = 0
    power = 0.75

    # compute denominator
    for entry in vocab_words:
        train_words_pow += pow(entry.count, power)

    i = 0
    d1 = pow(vocab_words[i].count, power) / train_words_pow

    # build table of word indices, where one word index occupies a number of entries which is proportional
    # to its probability
    for a in range(table_size):
        # store pointer to i-th word
        table[a] = i

        if a / table_size > d1:
            i += 1
            d1 += pow(vocab_words[i].count, power) / train_words_pow

        if i >= vocab_size:
            raise ValueError(f'{i} >= {vocab_size}')

    return table


def prepare_data(min_word_freq: int, special_tokens: List[str], train_file: str, vocab_file: str, vocab_words_file: str,
                 word_indices_file: str, unigram_table_file: str):
    vocab, vocab_words, word_indices = build_vocab(train_file, min_word_freq, special_tokens)
    vocab_size = len(vocab)

    assert len(vocab_words) == vocab_size
    assert len(word_indices) == vocab_size

    unigram_table = init_unigram_table(vocab_words, vocab_size)
    create_binary_tree(vocab_size, vocab_words)

    pickle.dump(vocab, open(vocab_file, 'wb'))
    pickle.dump(vocab_words, open(vocab_words_file, 'wb'))
    pickle.dump(word_indices, open(word_indices_file, 'wb'))
    pickle.dump(unigram_table, open(unigram_table_file, 'wb'))


def load_data(vocab_file: str, vocab_words_file: str, word_indices_file: str, unigram_table_file: str) -> Tuple[
    Vocab, List[VocabWord], Dict[str, int], np.ndarray]:
    vocab = pickle.load(open(vocab_file, 'rb'))
    vocab_words = pickle.load(open(vocab_words_file, 'rb'))
    word_indices = pickle.load(open(word_indices_file, 'rb'))
    unigram_table = pickle.load(open(unigram_table_file, 'rb'))

    return vocab, vocab_words, word_indices, unigram_table


def create_binary_tree(vocab_size: int, vocab_words: List[VocabWord]):
    count = np.zeros(vocab_size * 2 + 1, dtype=int)
    binary = np.zeros(vocab_size * 2 + 1, dtype=int)
    parent_node = np.zeros(vocab_size * 2 + 1, dtype=int)

    # set word counts for leaf nodes
    for a in range(vocab_size):
        count[a] = vocab_words[a].count

    # set word counts for inner nodes
    for a in range(vocab_size, vocab_size * 2):
        count[a] = 1e15

    # pointer into leaf nodes
    pos1 = vocab_size - 1

    # pointer into inner nodes
    pos2 = vocab_size

    # pos1 points to the smallest leaf node
    # pos2 points to the smallest inner node
    # with this the Huffman tree is constructed greedily

    for a in range(vocab_size - 1):
        # find two smallest nodes min1 and min2
        if pos1 >= 0:
            if count[pos1] < count[pos2]:
                min1 = pos1
                pos1 -= 1
            else:
                min1 = pos2
                pos2 += 1
        else:
            min1 = pos2
            pos2 += 1

        if pos1 >= 0:
            if count[pos1] < count[pos2]:
                min2 = pos1
                pos1 -= 1
            else:
                min2 = pos2
                pos2 += 1
        else:
            min2 = pos2
            pos2 += 1

        # create inner node
        new_node = vocab_size + a
        count[new_node] = count[min1] + count[min2]
        parent_node[min1] = new_node
        parent_node[min2] = new_node
        # binary[min1] = 0
        binary[min2] = 1

    # for each vocabulary word:
    # - assign binary code
    # - assign pointers to output table
    for a in range(vocab_size):
        b = a
        i = 0

        # binary code of word
        code = []

        # output indices of path nodes
        point = []

        while b != vocab_size * 2 - 2:
            code.append(binary[b])
            if i > 0:
                # append only inner nodes
                point.append(b - vocab_size)
            i += 1
            b = parent_node[b]

        # append root node
        # there are vocab_size - 1 combinations
        # root node is at vocab_size * 2 - 2 in the count/binary/parent_node matrix
        # root node is at vocab_size - 2 of the output matrix
        point.append(vocab_size - 2)

        # set codelen
        vocab_words[a].codelen = i

        vocab_words[a].code = list(reversed(code))
        vocab_words[a].point = list(reversed(point))


def build_unigram_sampler(vocab_words, vocab_size, special_tokens=None):
    total = sum([e.count for e in vocab_words])
    weights = [e.count / total for e in vocab_words]
    num_special_tokens = len(special_tokens) if special_tokens is not None else None

    def sample():
        word_index = np.random.choice(vocab_size, p=weights)

        # don't sample special tokens
        if num_special_tokens is not None:
            if word_index < num_special_tokens:
                word_index += num_special_tokens

        return word_index

    return sample


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def write_vectors(file: str, vectors: np.ndarray, vocab: Vocab):
    print(f'Writing vectors to {file}')

    with open(file, 'w') as f:
        num_examples, embedding_size = vectors.shape
        f.write(f'{num_examples} {embedding_size}\n')

        for i, w in enumerate(vocab.get_itos()):
            vec = vectors[i, :]
            f.write(w)
            for v in vec:
                f.write(f' {v:.6f}')
            f.write('\n')


# file locations
prefix = '/data/text8'
train_file = f'{prefix}/text8'
vocab_file = f'{prefix}/vocab'
vocab_words_file = f'{prefix}/vocab_words'
word_indices_file = f'{prefix}/word_indices'
unigram_table_file = f'{prefix}/unigram_table'
output_file = f'{prefix}/vectors.txt'

# number of epochs to train
num_epochs = 3

# size of the word vectors
embedding_size = 300

# include only words which appear at least min_word_freq times
min_word_freq = 5

special_tokens = ['</s>']

# maximum length of sentence to load at once
max_sentence_length = 1000

# max skip length between words
window_size = 5

# model variant
# - CBOW: predict middle word using continuous bag-of-words vector derived from context words
# - SkipGram: predict context word using word vector of middle word
model_variant = ModelVariant.CBOW
# model_variant = ModelVariant.SkipGram

# prediction mode
# predict_mode = PredictMode.HierarchicalSoftmax
predict_mode = PredictMode.NegativeSampling

# number of negative samples for NegativeSampling
num_negative_samples = 5

# sub-sampling of frequent words threshold
sub_sampling = 1e-4

# starting learning rate
learning_rate = 0.05 if model_variant == ModelVariant.CBOW else 0.025
starting_learning_rate = learning_rate

# data needs to prepared only once
#
# prepare_data(min_word_freq, special_tokens, train_file, vocab_file, vocab_words_file, word_indices_file,
#              unigram_table_file)
# exit(0)

vocab, vocab_words, word_indices, unigram_table = load_data(vocab_file, vocab_words_file, word_indices_file,
                                                            unigram_table_file)

vocab_size = len(vocab)
eos_idx = vocab['</s>']
unigram_table_size = unigram_table.shape[0]

# number of total training words
num_train_words = sum([e.count for e in vocab_words])

# number of total training words over all epochs
num_train_words_all_epochs = num_epochs * num_train_words

# word counters
num_words_actual = 0
num_words = 0
last_num_words = 0

# epoch counter
actual_num_epochs = 0

# word embedding matrices
input_word_vectors: np.ndarray = (np.random.rand(vocab_size, embedding_size) - 0.5) / embedding_size
output_word_vectors = np.zeros((vocab_size, embedding_size))

# helper vectors
hidden_vector = np.zeros(embedding_size)
eh_vector = np.zeros(embedding_size)

# the current sentence
sentence = []
sentence_length = 0
sentence_position = 0

print(f'Number of words: {vocab_size}')
print(f'</s> index: {eos_idx}')
print(f'Number of train words: {num_train_words}')
print(f'Max sentence length: {max_sentence_length}')
print(f'Number of epochs: {num_epochs}')
print(f'Learning rate: {learning_rate}')
print(f'Subsampling: {sub_sampling}')
print(f'Window size: {window_size}')
print(f'Embedding size: {embedding_size}')
print(f'Model variant: {model_variant}')
print(f'Predict mode: {predict_mode}')
print(f'Number of negative samples: {num_negative_samples}')
print(f'Input embeddings shape: {input_word_vectors.shape}')
print(f'Output embeddings shape: {output_word_vectors.shape}')

train_file_iter = read_word_indices_from_file(train_file, vocab)
eof = False

pbar = tqdm(total=num_train_words_all_epochs, desc='training.....', file=sys.stdout)

while True:
    new_words = num_words - last_num_words

    if new_words > 10000:
        # update number of processed words
        num_words_actual += new_words
        last_num_words = num_words

        # update progress
        pbar.update(new_words)
        pbar.set_description(f'epoch={actual_num_epochs + 1}/{num_epochs}.....')
        pbar.set_postfix(lr=learning_rate)

        # update learning rate
        learning_rate = starting_learning_rate * (1 - (num_words_actual / num_train_words_all_epochs))
        if learning_rate < starting_learning_rate * 0.0001:
            learning_rate = starting_learning_rate * 0.0001

    # read in next sentence
    if sentence_position >= sentence_length:
        sentence = []
        sentence_length = 0
        sentence_position = 0

        while sentence_length < max_sentence_length:
            word_index = next(train_file_iter)

            # skip out-of-vocabulary word
            if word_index == -1:
                continue

            # eof
            elif word_index is None:
                eof = True
                break

            num_words += 1

            # skip end of sentence </s> token
            if word_index == eos_idx:
                break

            # sub-sampling of frequent words
            if sub_sampling > 0:
                freq_w = vocab_words[word_index].count / num_train_words
                p_w = (math.sqrt(freq_w / sub_sampling) + 1) * sub_sampling / freq_w
                next_random = random.random()

                # p_w is the probability a word is kept
                # if p_w is lower than next_random, then the word is dropped
                if p_w < next_random:
                    continue

            sentence.append(word_index)
            sentence_length += 1

    # start next epoch or stop on last epoch
    # TODO last batch of words is not computed
    if eof:
        num_words_actual += new_words
        actual_num_epochs += 1

        if actual_num_epochs == num_epochs:
            break

        num_words = 0
        last_num_words = 0
        sentence = []
        sentence_length = 0
        sentence_position = 0
        train_file_iter = read_word_indices_from_file(train_file, vocab)
        eof = False

        continue

    # current word is used as output word in CBOW/SkipGram
    output_word_index = sentence[sentence_position]
    output_vocab_word = vocab_words[output_word_index]

    # reset hidden and EH vectors
    hidden_vector[:] = 0
    eh_vector[:] = 0

    # window size is randomized
    window_offset = np.random.randint(0, window_size)

    # train CBOW/Skip-Gram
    if model_variant == ModelVariant.CBOW:
        # in -> hidden
        # build input vector
        # this is a CBOW vector composed of all the context words
        context_size = 0
        for a in range(window_offset, 2 * window_size + 1 - window_offset):
            # skip middle word (output word)
            if a == window_size:
                continue

            input_word_position = sentence_position - window_size + a

            # skip positions lower 0 or larger than sentence length
            if input_word_position < 0 or input_word_position >= sentence_length:
                continue

            input_word_index = sentence[input_word_position]

            # add input word vector to CBOW vector
            hidden_vector += input_word_vectors[input_word_index, :]
            context_size += 1

        # CBOW vector is the mean of the context vectors
        hidden_vector /= context_size

        if predict_mode == PredictMode.HierarchicalSoftmax:
            # traverse the huffman tree top to down
            for d in range(output_vocab_word.codelen):
                # index from inner node
                inner_node_index = output_vocab_word.point[d]
                inner_node_vector = output_word_vectors[inner_node_index, :]

                # Propagate hidden -> output
                # Compute probability to choose the left path
                f = np.dot(hidden_vector, inner_node_vector)
                f = sigmoid(f)

                # label = 1: left path should be chosen, positive example, f should be close to 1
                # label = 0: right path should be chosen, negative example, f should be close to 0
                label = 1 - output_vocab_word.code[d]

                # 'g' is the error multiplied by the learning rate
                # in order to have low error:
                # - positive example: f should be close to 1
                #   hidden_vector and inner_node_vector should be similar
                # - negative example: f should be close to 0
                #   hidden_vector and inner_node_vector should be different
                g = (label - f) * learning_rate

                # Propagate errors output -> hidden
                # Aggregate EH
                eh_vector += g * inner_node_vector

                # Learn weights hidden -> output
                # Update the inner node vectors
                output_word_vectors[inner_node_index, :] += g * hidden_vector

        elif predict_mode == PredictMode.NegativeSampling:
            for d in range(0, num_negative_samples + 1):
                # output word is positive example
                if d == 0:
                    target = output_word_index
                    label = 1

                # sample random word is negative example
                else:
                    unigram_index = np.random.randint(0, unigram_table_size)
                    target = unigram_table[unigram_index]

                    # output word should not be a negative sample
                    if target == output_word_index:
                        continue

                    # if target is eos token, sample random other word
                    if target == 0:
                        target = np.random.randint(1, vocab_size)

                    label = 0

                output_word_vector = output_word_vectors[target, :]

                f = np.dot(hidden_vector, output_word_vector)
                f = sigmoid(f)

                g = (label - f) * learning_rate

                eh_vector += g * output_word_vector
                output_word_vectors[target, :] += g * hidden_vector

        # hidden -> in
        # Update the input word vectors for all context words
        for a in range(window_offset, 2 * window_size + 1 - window_offset):
            # skip middle output word
            if a == window_size:
                continue

            input_word_position = sentence_position - window_size + a

            # skip positions lower 0 or larger than sentence length
            if input_word_position < 0 or input_word_position >= sentence_length:
                continue

            input_word_index = sentence[input_word_position]

            # Update the input word vectors
            # TODO does not normalize by dividing with context size
            input_word_vectors[input_word_index, :] += eh_vector

    elif model_variant == ModelVariant.SkipGram:
        for a in range(window_offset, window_size * 2 + 1 - window_offset):
            # skip middle output word
            if a == window_size:
                continue

            input_word_position = sentence_position - window_size + a

            # skip positions lower 0 or larger than sentence length
            if input_word_position < 0 or input_word_position >= sentence_length:
                continue

            # input word from context should predict the current output word
            input_word_index = sentence[input_word_position]
            input_word_vector = input_word_vectors[input_word_index, :]

            # reset EH
            eh_vector[:] = 0

            if predict_mode == PredictMode.HierarchicalSoftmax:
                for d in range(output_vocab_word.codelen):
                    inner_node_index = output_vocab_word.point[d]
                    inner_node_vector = output_word_vectors[inner_node_index, :]

                    # Propagate hidden -> output
                    # Compute probability to choose the left path
                    f = np.dot(input_word_vector, inner_node_vector)
                    f = sigmoid(f)

                    label = 1 - output_vocab_word.code[d]

                    g = (label - f) * learning_rate

                    # Propagate errors output -> hidden
                    # Aggregate EH
                    eh_vector += g * inner_node_vector

                    # Learn weights hidden -> output
                    # Update the inner node vectors
                    output_word_vectors[inner_node_index, :] += g * input_word_vector

            elif predict_mode == PredictMode.NegativeSampling:
                for d in range(0, num_negative_samples + 1):
                    # output word is positive example
                    if d == 0:
                        target = output_word_index
                        label = 1

                    # sample random word is negative example
                    else:
                        unigram_index = np.random.randint(0, unigram_table_size)
                        target = unigram_table[unigram_index]

                        if target == output_word_index:
                            continue

                        # if target is eos token, sample random other word
                        if target == 0:
                            target = np.random.randint(1, vocab_size)

                        label = 0

                    output_word_vector = output_word_vectors[target, :]

                    f = np.dot(input_word_vector, output_word_vector)
                    f = sigmoid(f)

                    g = (label - f) * learning_rate

                    eh_vector += g * output_word_vector
                    output_word_vectors[target, :] += g * input_word_vector

            # Update the input word vectors
            # TODO does not normalize by dividing with context size
            input_word_vectors[input_word_index, :] += eh_vector

    sentence_position += 1

pbar.close()

write_vectors(output_file, input_word_vectors, vocab)
