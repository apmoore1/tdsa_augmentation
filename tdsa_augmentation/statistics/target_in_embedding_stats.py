import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

#from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

from tdsa_augmentation.helpers.general_helper import multi_word_targets

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def target_length_count(targets: List[str], 
                        string_delimiter: str = '_') -> Dict[int, int]:
    '''
    :param targets: A list of target strings.
    :param string_delimiter: The value used to split the target into multiple 
                             words.
    :returns: A dictionary where keys are the length of a target in tokens based 
              on the string_delimiter and the values are the number of targets 
              of said token length.
    '''
    length_count: Dict[int, int] = defaultdict(lambda: 0)
    for target in targets:
        length_count[len(target.split(string_delimiter))] += 1
    return dict(length_count)

def cumulate_length_count(length_count: Dict[int, int], max_length: int
                           ) -> Dict[str, int]:
    '''
    :param length_count: Keys are the length of the target in tokens and the 
                         values are the number of targets that have this 
                         length.
    :param max_length: The maximum length of a target in tokens before they 
                       all merge into the same max length target bucket/key.
    :returns: A dictionary of keys being the target token length and the value 
              being the number of target of said length. Given a max_length 
              argument the largest key value can be max length where anything 
              bigger than this will be counted in this key's value.
    '''
    length_count = sorted(length_count.items(), key=lambda x: x[0])
    if length_count[0][0] == 0:
        raise ValueError('The lowest length cannot be zero')
    temp_length_count = defaultdict(lambda: 0)
    for length, count in length_count:
        if length >= max_length:
            temp_length_count[f'{max_length}+'] += count
        else:
            temp_length_count[f'{length}'] += count
    return dict(temp_length_count)

def fraction_words_in_target(multi_word_targets: List[str], 
                             embedding: KeyedVectors,
                             fraction: float, 
                             string_delimiter: str = '_') -> List[str]:
    '''
    :param multi_word_targets: A list of targets that are multi word targets.
    :param embedding: The embedding you want statistics on
    :param fraction: The fraction of target words within one target that is 
                     also in the embedding to declare the embedding can be 
                     used for this multi word target
    :param string_delimiter: The string to be used to split the target up into 
                             multiple words
    :returns: The list of targets that are multi word targets, that contain 
              enough words within each target that are within the embedding 
              to state that the embedding contains enough information 
              to represent these multi word targets. The `fraction` argument 
              states what are enough words in the embedding that are also 
              in the multi word target for it to be in this returned list.
    '''
    filtered_mwe_targets = set()
    for mwe_target in multi_word_targets:
        words = mwe_target.split(string_delimiter) 
        words_in_embedding = [word for word in words if word in embedding.wv]
        if len(words) <= 1:
            raise ValueError('Number of words in MWE target has to be greater '
                             f'than 1. MWE Target {mwe_target}')
        frac_words = len(words_in_embedding) / len(words)
        if frac_words >= fraction:
            filtered_mwe_targets.add(mwe_target)
    return list(filtered_mwe_targets)

if __name__ == '__main__':
    fraction_help = "The fraction fo words within a target that the embedding "\
                    "must have to state that the embedding can represent "\
                    "this word"
    parser = argparse.ArgumentParser()
    parser.add_argument("target_list_data_fp", type=parse_path, 
                        help='File path to the list of targets')
    parser.add_argument("embedding_fp", type=parse_path, 
                        help='File path to the embedding that we want to know'\
                             ' how many of the targets are in')
    parser.add_argument("--fraction", type=float, help=fraction_help)
    args = parser.parse_args()
    fraction = args.fraction
    if fraction is None:
        fraction = 1.0

    targets = set()
    with args.target_list_data_fp.open('r') as target_list_file:
        for line in target_list_file:
            line = line.strip().lower()
            if not line:
                continue
            targets.add(line)
    targets = list(targets)
    multi_targets = list(multi_word_targets(targets, lower=True).values())

    embedding = KeyedVectors.load(str(args.embedding_fp))
    # Targets that are in the embedding
    embedding_multi_targets = [target for target in multi_targets 
                               if target in embedding.wv]
    # Targets that multi word word targets but not in the embeddings as a 
    # multi word target
    multi_target_not_in_embedding = [target for target in multi_targets if '_' in target]
    multi_target_not_in_embedding = [target for target in multi_target_not_in_embedding 
                                     if target not in embedding.wv]
    multi_target_embedding_can_rep = fraction_words_in_target(multi_target_not_in_embedding, 
                                                              embedding, fraction)
    # Statistics
    # Calculate the length of targets as normal
    normal_target_length_count = target_length_count(multi_targets)
    cumulate_normal_length_count = cumulate_length_count(normal_target_length_count, 3)
    # Calculate the length of targets that are in the embedding including those 
    # that represent MWE as one embedding
    filtered_target_length_count = target_length_count(embedding_multi_targets)
    cumulate_filtered_length_count = cumulate_length_count(filtered_target_length_count, 3)
    # Calculate the number of targets that are multi word expressions but 
    # can be represented as an embedding if the number of targets in the embedding 
    # is at least 50% of the words.
    multi_not_in_embedding_length_count = target_length_count(multi_target_embedding_can_rep)
    cumulate_multi_not_in_embedding_length_count = cumulate_length_count(multi_not_in_embedding_length_count, 3)

    print(f'Number of targets (lowered): {len(targets)}')
    print(f'Number of targets directly in the embedding: {len(embedding_multi_targets)}')
    print('Number of targets based on tokenised length count:'
          f' {cumulate_normal_length_count}')
    print('Number of targets directly in the embedding based on tokenised length count:'
          f' {cumulate_filtered_length_count}')
    print(f'Number of multi word targets where at least {fraction * 100}% '
          'of the target words within the multi word targets are in the embedding'
          f' {cumulate_multi_not_in_embedding_length_count}.\nThese numbers do '
          'not contain those multi word targets that are already in the embedding')
    # Need to re-calculate the number of words in the embedding after doing the average thing.
    all_rep_words = list(set(embedding_multi_targets + multi_target_embedding_can_rep))
    total_embedding_length_count = target_length_count(all_rep_words)
    cumulate_total_embedding_length_count = cumulate_length_count(total_embedding_length_count, 3)
    print('Number of targets directly and can be represented through '
          'averaging based on tokenised length count: '
          f'{cumulate_total_embedding_length_count}')
    total_number_words_in_embeddding = len(all_rep_words)
    assert total_number_words_in_embeddding == sum(cumulate_total_embedding_length_count.values())
    print(f'Total number of targets in the embeddings {total_number_words_in_embeddding}')
    coverage_percent = round((total_number_words_in_embeddding / len(targets)) * 100, 2)
    print(f'Percentage of target words covered {coverage_percent}%')