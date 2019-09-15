import argparse
from pathlib import Path
import random as rand

from sklearn.model_selection import train_test_split

from target_extraction.data_types import TargetTextCollection
from target_extraction.dataset_parsers import semeval_2014, semeval_2016
from target_extraction.dataset_parsers import wang_2017_election_twitter_train
from target_extraction.dataset_parsers import wang_2017_election_twitter_test
from target_extraction import tokenizers, pos_taggers

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dataset_fp", type=parse_path,
                        help="File path to the training dataset (JSON)")
    parser.add_argument("val_dataset_fp", type=parse_path,
                        help="File path to the validation dataset (JSON)")
    parser.add_argument("test_dataset_fp", type=parse_path,
                        help='File path to the test dataset (JSON)')
    parser.add_argument("save_train_fp", type=parse_path, 
                        help='File Path to save the new training dataset to')
    parser.add_argument("save_val_fp", type=parse_path, 
                        help='File Path to save the new validation dataset to')
    parser.add_argument("save_test_fp", type=parse_path, 
                        help='File Path to save the new test dataset to')
    args = parser.parse_args()
    
    train = TargetTextCollection.load_json(args.train_fp)
    val = TargetTextCollection.load_json(args.val_fp)
    test = TargetTextCollection.load_json(args.test_fp)

    train.force_targets()
    # The main thing I will have to do now is similar to the force targets 
    # but instead of doing it for the whole text create one for each of the
    # targets I think! This will also get around the problem of multiple 
    # targets over the same token in the Election Twitter dataset.
    datasets = [train, val, test]
    pos_tagger = pos_taggers.spacy_tagger()
    for dataset in datasets:
        dataset: TargetTextCollection
        if args.task == 'sentiment':
            dataset.pos_text(pos_tagger)
        else:
            dataset.pos_text(pos_tagger)
            errors = dataset.sequence_labels(return_errors=True)
            if errors and not args.remove_errors:
                raise ValueError('While creating the sequence labels the '
                                f'following sequence labels have occured {errors}')
            elif errors:
                print(f'{len(errors)} number of sequence labels errors have occured'
                    ' and will be removed from the dataset')
                for error in errors:
                    del dataset[error['text_id']]
    print(f'Length of train, val and test:')
    print([dataset_length(args.task, dataset) 
           for dataset in [train_dataset, val_dataset, test_dataset]])

    args.save_train_fp.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving the JSON training dataset to {args.save_train_fp}')
    train_dataset.to_json_file(args.save_train_fp)
    print(f'Saving the JSON training dataset to {args.save_val_fp}')
    val_dataset.to_json_file(args.save_val_fp)
    print(f'Saving the JSON training dataset to {args.save_test_fp}')
    test_dataset.to_json_file(args.save_test_fp)