import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from gensim.models.word2vec import Word2Vec

from tdsa_augmentation.helpers.general_helper import multi_word_targets

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def target_length_count(targets: List[str]) -> Dict[int, int]:
    length_count: Dict[int, int] = defaultdict(lambda: 0)
    for target in targets:
        length_count[len(target.split('_'))] += 1
    return dict(length_count)

def cumulate_length_count(length_count: Dict[int, int], max_length: int
                           ) -> Dict[str, int]:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("target_list_data_fp", type=parse_path, 
                        help='File path to the list of targets')
    parser.add_argument("embedding_fp", type=parse_path, 
                        help='File path to the embedding that we want to know'\
                             ' how many of the targets are in')
    args = parser.parse_args()

    targets = set()
    with args.target_list_data_fp.open('r') as target_list_file:
        for line in target_list_file:
            line = line.strip()
            if not line:
                continue
            targets.add(line)
    targets = list(targets)
    targets = multi_word_targets(targets, lower=True)

    embedding = Word2Vec.load(str(args.embedding_fp))
    # Targets that are in the embedding
    embedding_targets = [target for target in targets 
                         if target in embedding.wv]
    # Basic stats
    normal_target_length_count = target_length_count(targets)
    cumulate_normal_length_count = cumulate_length_count(normal_target_length_count, 3)
    
    filtered_target_length_count = target_length_count(embedding_targets)
    cumulate_filtered_length_count = cumulate_length_count(filtered_target_length_count, 3)

    print(f'Number of targets (lowered): {len(targets)}')
    print(f'Number of targets in the embedding: {len(embedding_targets)}')

    print('Number of targets based on tokenised length count:'
          f' {cumulate_normal_length_count}')
    print('Number of targets in the embedding based on tokenised length count:'
          f' {cumulate_filtered_length_count}')