import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from gensim.models import KeyedVectors
import numpy as np

from tdsa_augmentation.helpers.general_helper import multi_word_targets, extract_multi_word_targets
from tdsa_augmentation.helpers.general_helper import get_embedding

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def get_target_data(target_fp: Path, embedding: KeyedVectors, fraction: float
                    ) -> Tuple[List[str], np.ndarray, Dict[str, str]]:
    '''
    :param target_fp: The path to a file that contains a list of targets
    :param embedding: The embedding to be used to find similar targets from the 
                      target file
    :param fraction: The fraction of words within a multi word target that must 
                     exist in the embedding for the target to represented.
    :returns: A Tuple containing three elements: 1. The list of targets that 
              can be represented by the embedding. 2. The embedding vector 
              for each target it can represent. 3. Dictionary mapping 
              the multi word targets in the first element in the tuple to 
              their actual non-tokenised target word(s). 
    '''
    targets = set()
    with target_fp.open('r') as target_list_file:
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

    return mw_targets, mw_embeddings, mw_target_mapper

if __name__=='__main__':
    save_fp_help = 'File Path to save the targets and there top N most similar'\
                   ' words as a json dictionary of {`target`: List[similar targets]}'
    fraction_help = "The fraction of words within a target that the embedding "\
                    "must have to state that the embedding can represent "\
                    "this target"
    additional_train_help = "When the target list data fp argument is a list"\
                            " of predicted targets this should be the file "\
                            "path to the list of training targets that are "\
                            "used to find similar predicted targets for"
    parser = argparse.ArgumentParser()
    parser.add_argument("target_list_data_fp", type=parse_path, 
                        help='File path to the list of targets')
    parser.add_argument("embedding_fp", type=parse_path, 
                        help='File path to the embedding that will expand the targets')
    parser.add_argument("N", type=int, 
                        help='Top N similar targets to expand each target')
    parser.add_argument("save_fp", type=parse_path, help=save_fp_help)
    parser.add_argument("fraction", type=float, help=fraction_help)
    parser.add_argument("--additional_train_targets", type=parse_path, 
                        help=additional_train_help)
    args = parser.parse_args()
    embedding = KeyedVectors.load(str(args.embedding_fp))
    mw_targets, mw_embeddings, mw_target_mapper = get_target_data(args.target_list_data_fp, embedding, args.fraction)
    search_mw_targets = mw_targets
    search_mw_embeddings = mw_embeddings
    search_mw_target_mapper = mw_target_mapper
    if args.additional_train_targets is not None:
        search_mw_targets, search_mw_embeddings, search_mw_target_mapper = get_target_data(args.additional_train_targets, embedding, args.fraction)

    target_similar_targets: Dict[str, List[str]] = {}
    for search_mw_index, search_mw_target in enumerate(search_mw_targets):
        search_mw_target_embedding = search_mw_embeddings[search_mw_index]
        target_similarities = embedding.wv.cosine_similarities(search_mw_target_embedding, 
                                                               mw_embeddings)
        # sort similarties
        similarity_indexs = np.argsort(target_similarities).tolist()
        similarity_indexs.reverse()
        if args.additional_train_targets is None:
            # Remove the index of the current mw target as we do not want similar
            # targets that are the same target
            similarity_indexs.remove(search_mw_index)
        similar_targets = [mw_target_mapper[mw_targets[sim_index]] 
                           for sim_index in similarity_indexs[:args.N]]
        target = search_mw_target_mapper[search_mw_target]
        if target in similar_targets:
            print(f'The target {target} is in similar targets {similar_targets}'
                  ' thus removing the target')
            similar_targets.remove(target)
        target_similar_targets[target] = similar_targets
    args.save_fp.parent.mkdir(parents=True, exist_ok=True)
    with args.save_fp.open('w+') as save_file:
        json.dump(target_similar_targets, save_file)