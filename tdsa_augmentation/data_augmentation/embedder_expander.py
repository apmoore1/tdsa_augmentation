import argparse
import json
from pathlib import Path
from typing import Dict, List

from gensim.models import KeyedVectors
import numpy as np

from tdsa_augmentation.helpers.general_helper import multi_word_targets, extract_multi_word_targets
from tdsa_augmentation.helpers.general_helper import get_embedding

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__=='__main__':
    save_fp_help = 'File Path to save the targets and there top N most similar'\
                   ' words as a json dictionary of {`target`: List[similar targets]}'
    fraction_help = "The fraction of words within a target that the embedding "\
                    "must have to state that the embedding can represent "\
                    "this target"
    parser = argparse.ArgumentParser()
    parser.add_argument("target_list_data_fp", type=parse_path, 
                        help='File path to the list of targets')
    parser.add_argument("embedding_fp", type=parse_path, 
                        help='File path to the embedding that will expand the targets')
    parser.add_argument("N", type=int, 
                        help='Top N similar targets to expand each target')
    parser.add_argument("save_fp", type=parse_path, help=save_fp_help)
    parser.add_argument("fraction", type=float, help=fraction_help)
    args = parser.parse_args()

    targets = set()
    with args.target_list_data_fp.open('r') as target_list_file:
        for line in target_list_file:
            line = line.strip()
            if not line:
                continue
            targets.add(line.lower())
    targets = list(targets)
    # Mapper to map targets from that contain `_` from the `multi_word_targets`
    # function back to normally perfectly without tokenization error.
    target_mw: Dict[str, str] = multi_word_targets(targets, lower=True)
    mw_target_mapper = {mw: target for target, mw in target_mw.items()}
    # Get all the targets the embedding can represent through averaging the 
    # words within the Multi Word Targets
    embedding = KeyedVectors.load(str(args.embedding_fp))
    fraction = args.fraction
    extract_multi_word_targets_args = {'targets': targets, 'fraction': fraction,
                                       'embedding': embedding, 'mw_targets_ignore': None,
                                       'string_delimiter': '_', 
                                       'remove_single_word_targets': True,
                                       'remove_existing_mw_targets_in_embedding': True,
                                       'raise_value_error_on_single_target': True}
    multi_target_embedding_can_rep = extract_multi_word_targets(**extract_multi_word_targets_args)
    targets_direct_in_embedding = [mw for target, mw in target_mw.items() if mw in embedding.wv]
    
    direct_targets_embedding = get_embedding(embedding, targets_direct_in_embedding)
    rep_targets_embedding = get_embedding(embedding, multi_target_embedding_can_rep,
                                          average=True, string_delimiter='_')
    mw_targets_embedding = {**direct_targets_embedding, **rep_targets_embedding}
    mw_targets = []
    mw_embeddings = []
    for mw_target, mw_embedding in mw_targets_embedding.items():
        mw_targets.append(mw_target)
        mw_embeddings.append(mw_embedding)
    mw_embeddings = np.array(mw_embeddings)

    target_similar_targets: Dict[str, List[str]] = {}
    for mw_index, mw_target in enumerate(mw_targets):
        mw_target_embedding = mw_embeddings[mw_index]
        target_similarities = embedding.wv.cosine_similarities(mw_target_embedding, 
                                                               mw_embeddings)
        # sort similarties
        similarity_indexs = np.argsort(target_similarities).tolist()
        similarity_indexs.reverse()
        # Remove the index of the current mw target as we do not want similar
        # targets that are the same target
        similarity_indexs.remove(mw_index)
        similar_targets = [mw_target_mapper[mw_targets[sim_index]] 
                           for sim_index in similarity_indexs[:args.N]]
        target = mw_target_mapper[mw_target]
        if target in similar_targets:
            print(f'The target {target} is in similar targets {similar_targets}'
                  ' thus removing the target')
            similar_targets.remove(target)
        target_similar_targets[target] = similar_targets
    args.save_fp.parent.mkdir(parents=True, exist_ok=True)
    with args.save_fp.open('w+') as save_file:
        json.dump(target_similar_targets, save_file)