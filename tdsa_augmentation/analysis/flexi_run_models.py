import argparse
from pathlib import Path
import tempfile

from allennlp.common.params import Params

from tdsa_augmentation.analysis.run_model import run_model

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def add_elmo_indexer(config_params: Params) -> Params:
    elmo_indexer = {"elmo": {"type": "elmo_characters", "token_min_padding_length": 1}}
    if 'token_indexers' not in config_params['dataset_reader']:
        config_params['dataset_reader']['token_indexers'] = elmo_indexer
    else:
        config_params['dataset_reader']['token_indexers']['elmo'] = elmo_indexer['elmo']
    return config_params

def add_token_indexer(config_params: Params) -> Params:
    token_indexer = {"tokens": {"type": "single_id", "lowercase_tokens": True,
                                "token_min_padding_length": 1}}
    if 'token_indexers' not in config_params['dataset_reader']:
        config_params['dataset_reader']['token_indexers'] = token_indexer
    else:
        config_params['dataset_reader']['token_indexers']['tokens'] = token_indexer['tokens']
    return config_params 

def add_word_embedding(config_params: Params, pre_trained_fp: Path) -> Params:
    pre_trained_path = str(pre_trained_fp.resolve())
    embedding_dict = {"type": "embedding", "embedding_dim": 300, 
                      "pretrained_file": f"{pre_trained_path}",
                      "trainable": False}
    if 'context_field_embedder' not in config_params['model']:
        config_params['model']['context_field_embedder'] = {'tokens': embedding_dict}
    else:
        config_params['model']['context_field_embedder']['tokens'] = embedding_dict
    return config_params

def add_elmo_embedding(config_params: Params, pre_trained_fp: Path) -> Params:
    pre_trained_path = str(pre_trained_fp.resolve())
    embedding_dict = {"type": "bidirectional_lm_token_embedder",
                      "archive_file": f"{pre_trained_path}",
                      "bos_eos_tokens": ["<S>", "</S>"],
                      "remove_bos_eos": True,
                      "requires_grad": False}
    if 'context_field_embedder' not in config_params['model']:
        config_params['model']['context_field_embedder'] = {'elmo': embedding_dict}
    else:
        config_params['model']['context_field_embedder']['elmo'] = embedding_dict
    return config_params

def model_specific_rep_params(config_params: Params, model_name: str, 
                              word_rep_dim: int) -> Params:
    if model_name == 'ian':
        config_params['model']['context_encoder']['input_size'] = word_rep_dim
        config_params['model']['target_encoder']['input_size'] = word_rep_dim
    if model_name == 'tdlstm':
        config_params['model']['left_text_encoder']['input_size'] = word_rep_dim
        config_params['model']['right_text_encoder']['input_size'] = word_rep_dim
    if model_name == 'tclstm':
        config_params['model']['left_text_encoder']['input_size'] = word_rep_dim * 2
        config_params['model']['right_text_encoder']['input_size'] = word_rep_dim * 2
        config_params['model']['target_encoder']['embedding_dim'] = word_rep_dim
    if model_name == 'atae':
        config_params['model']['context_encoder']['input_size'] = word_rep_dim * 2
        config_params['model']['target_encoder']['embedding_dim'] = word_rep_dim
    if model_name == 'at':
        config_params['model']['context_encoder']['input_size'] = word_rep_dim
        config_params['model']['target_encoder']['embedding_dim'] = word_rep_dim
    if model_name == 'interae':
        config_params['model']['context_encoder']['input_size'] = word_rep_dim * 2
        config_params['model']['target_encoder']['input_size'] = word_rep_dim
    return config_params

def encoder_name(is_elmo: bool, is_elmo_ds: bool, is_word_embedding: bool, 
                 is_word_embedding_ds: bool) -> str:
    name = ''
    if is_elmo:
        name = 'Elmo'
    if is_elmo_ds:
        name = 'DS_Elmo'
    if is_word_embedding:
        if name != '':
            name = f'{name}_WE'
        else:
            name = 'WE'
    if is_word_embedding_ds:
        if name != '':
            name = f'{name}_DS_WE'
        else:
            name = 'DS_WE'
    return name

if __name__=='__main__':
    model_save_fp_help = "File Path to save the model. NOTE if N>1 then the "\
                         "first model will be saved. This needs to be a directory"\
                         " that does not exist currently."
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=parse_path, 
                        help='File Path to directory that contains the train, val, and test data')
    parser.add_argument("config_fp", type=parse_path, 
                        help='File Path to the models config file')
    parser.add_argument("N", type=int, 
                        help='Number of times to run the model')
    parser.add_argument("domain", type=str, choices=['laptop', 'restaurant', 'election'])
    parser.add_argument("model_name", type=str, choices=['ian', 'tdlstm', 'tclstm', 'atae', 'at', 'interae'])
    parser.add_argument("save_dir", type=parse_path, 
                        help='Top level directory to save all of the test and validation data')
    parser.add_argument("--elmo", action='store_true')
    parser.add_argument("--elmo_ds", action='store_true')
    parser.add_argument("--word_embedding", action='store_true')
    parser.add_argument("--word_embedding_ds", action='store_true')
    args = parser.parse_args()

    config_params = Params.from_file(args.config_fp)

    if not args.elmo and not args.elmo_ds and not args.word_embedding and not args.word_embedding_ds:
        raise ValueError('Require at least one of the following arguments to be true'
                         '`elmo`, `elmo_ds`, `word_embedding`, `word_embedding_ds`')
    # Cannot have two of the same types
    if args.elmo and args.elmo_ds:
        raise ValueError('Cannot have both `elmo` and `elmo_ds`')
    if args.word_embedding and args.word_embedding_ds:
        raise ValueError('Cannot have both `word_embedding` and `word_embedding_ds`')
    # Add relevant token indexers
    if args.elmo or args.elmo_ds:
        add_elmo_indexer(config_params)
    if args.word_embedding or args.word_embedding_ds:
        add_token_indexer(config_params)
    # Add relevant word embedding contexts
    word_embedding_dir = Path('.', 'resources', 'word_embeddings').resolve()
    if args.word_embedding:
        glove_embedding_fp = Path(word_embedding_dir, 'glove.840B.300d.txt')
        add_word_embedding(config_params, glove_embedding_fp)
    if args.word_embedding_ds:
        ds_embedding_fp = Path(word_embedding_dir, f'{args.domain}_glove.txt')
        add_word_embedding(config_params, ds_embedding_fp)
    # Add relevant elmo embeddings contexts
    elmo_dir = Path('.', 'resources', 'language_models').resolve()
    if args.elmo:
        elmo_fp = Path(elmo_dir, 'transformer-elmo-2019.01.10.tar.gz')
        add_elmo_embedding(config_params, elmo_fp)
    if args.elmo_ds:
        ds_elmo_fp = Path(elmo_dir, f'{args.domain}_model.tar.gz')
        add_elmo_embedding(config_params, ds_elmo_fp)
    # Find the context encoder size
    word_rep_dim = 0
    if args.elmo:
        word_rep_dim += 1024
    if args.elmo_ds:
        word_rep_dim += 1024
    if args.word_embedding:
        word_rep_dim += 300
    if args.word_embedding_ds:
        word_rep_dim += 300
    # Need to change model specific parameters based on the word representation
    # dimensions
    model_specific_rep_params(config_params, args.model_name, word_rep_dim)
    # save directory
    encoder = encoder_name(args.elmo, args.elmo_ds, args.word_embedding, 
                           args.word_embedding_ds)
    print(encoder)
    save_dir = Path(args.save_dir, args.model_name, args.domain, encoder).resolve() 
    save_dir.mkdir(parents=True, exist_ok=True)
    test_save_fp = Path(save_dir, 'pred_test.json')
    val_save_fp = Path(save_dir, 'pred_val.json')
    if test_save_fp.exists() and val_save_fp.exists():
        print(f'Predictions have already been made at the following directory {save_dir}') 
    else:
        dataset_dir = args.dataset_dir
        train_fp = Path(dataset_dir, 'train.json')
        val_fp = Path(dataset_dir, 'val.json')
        test_fp = Path(dataset_dir, 'test.json')
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            config_params.to_file(temp_file.name)

            run_model(train_fp, val_fp, test_fp, Path(temp_file.name), test_save_fp, 
                      args.N, None, val_save_fp)