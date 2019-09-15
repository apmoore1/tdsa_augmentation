import argparse
import json
from pathlib import Path
from typing import Dict, List

from gensim.models.word2vec import Word2Vec
from target_extraction.data_types import TargetTextCollection
import numpy as np

from tdsa_augmentation.helpers.general_helper import multi_word_targets

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__=='__main__':
    save_fp_help = 'File Path to save the targets and there top N most similar'\
                   ' words as a json dictionary of {`target`: List[similar targets]}'
    parser = argparse.ArgumentParser()
    parser.add_argument("target_list_data_fp", type=parse_path, 
                        help='File path to the list of targets')
    parser.add_argument("train_dataset_fp", type=parse_path, help="Training dataset")
    parser.add_argument("embedding_fp", type=parse_path, 
                        help='File path to the embedding that will expand the targets')
    parser.add_argument("N", type=int, 
                        help='Top N similar targets to expand each target')
    parser.add_argument("save_fp", type=parse_path, help=save_fp_help)
    args = parser.parse_args()

    embedding = Word2Vec.load(str(args.embedding_fp))

    targets = set()
    with args.target_list_data_fp.open('r') as target_list_file:
        for line in target_list_file:
            line = line.strip()
            if not line:
                continue
            targets.add(line.lower())
    targets = list(targets)
    target_mwe: Dict[str, str] = multi_word_targets(targets, lower=True)
    target_mwe = {target: mwe for target, mwe in target_mwe.items() if mwe in embedding.wv}
    target_mwe_list = list(set(target_mwe.values()))
    # Mapper to map targets from that contain `_` from the `multi_word_targets`
    # function back to normally perfectly without tokenization error. Cannot 
    # be done for all
    mwe_target = {mwe: target for target, mwe in target_mwe.items()}

    train_targets = list(TargetTextCollection.load_json(args.train_dataset_fp).target_count(lower=True).keys())
    train_target_mwe  = multi_word_targets(train_targets, lower=True)

    
    # Targets that are in the embedding and in the training dataset
    embedding_targets_mwe = {target: mwe for target, mwe in train_target_mwe.items()
                             if mwe in embedding.wv}

    target_similar_targets: Dict[str, List[str]] = {}
    error_count = 0
    for target, mwe in embedding_targets_mwe.items():
        similarity_scores = embedding.wv.distances(mwe, target_mwe_list)
        highest_n_indexs = np.argsort(similarity_scores)[-50:]
        highest_scoring_words = [target_mwe_list[index] for index in highest_n_indexs]
        word_sim_value = embedding.wv.most_similar(positive=[mwe], topn=200 + 1)
        in_the_list = [word for word, sim in word_sim_value if word in target_mwe_list]
        import pdb
        pdb.set_trace()
        temp_similar_targets = sorted(word_sim_value, key=lambda x: x[1])
        is_mwe_in_sim_targets = False
        for sim_target, sim in temp_similar_targets:
            if mwe == sim_target or sim_target == target:
                is_mwe_in_sim_targets = True
        if not is_mwe_in_sim_targets:
            temp_similar_targets = temp_similar_targets[1:]
        temp_similar_targets = [word for word, sim in temp_similar_targets]
        similar_targets = []
        for sim_target in temp_similar_targets:
            if sim_target in mwe_target:
                sim_target = mwe_target[sim_target].strip()
            else:
                sim_target = ' '.join(sim_target.split('_')).strip()
            similar_targets.append(sim_target)
        try:
            assert len(set(similar_targets)) == len(similar_targets), f'{similar_targets}'
        except:
            error_count +=1
            continue
        assert args.N == len(similar_targets), print(len(similar_targets))
        target_similar_targets[target] = similar_targets
    print(f'Error count {error_count}')
    args.save_fp.parent.mkdir(parents=True, exist_ok=True)
    with args.save_fp.open('w+') as save_file:
        json.dump(target_similar_targets, save_file)

    