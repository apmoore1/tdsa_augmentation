import argparse
import copy
from pathlib import Path

from target_extraction.data_types import TargetText, TargetTextCollection

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_augmented_fp", type=parse_path, 
                        help='File path to the training only targets augmented dataset')
    parser.add_argument("predicted_augmented_fp", type=parse_path, 
                        help='File path to the predicted only targets augmented dataset')
    parser.add_argument("save_fp", type=parse_path, 
                        help='File path to save the new combined dataset')
    args = parser.parse_args()

    augmented_train_dataset = TargetTextCollection.load_json(args.train_augmented_fp)
    augmented_pred_dataset = TargetTextCollection.load_json(args.predicted_augmented_fp)
    save_fp = args.save_fp

    assert len(augmented_train_dataset) == len(augmented_pred_dataset)
    assert augmented_train_dataset == augmented_pred_dataset
    assert augmented_pred_dataset == augmented_train_dataset

    combined_target_objects = []
    for text_id, train_target_text in augmented_train_dataset.items():
        pred_target_text = augmented_pred_dataset[text_id]

        train_targets = train_target_text['targets']
        pred_targets = pred_target_text['targets']
        target_not_same_err = ('The target lists have to be the same within '
                               f'train {train_target_text} and predicted '
                               f'{pred_target_text}')
        assert train_targets == pred_targets, target_not_same_err
        number_targets = len(train_targets)
        combined_target_object = copy.deepcopy(train_target_text)
        for target_index in range(0, number_targets):
            eq_key = f'target {target_index}'
            train_equivalent_targets = train_target_text[eq_key]
            pred_equivalent_targets = pred_target_text[eq_key]
            combined_equivalent_targets = train_equivalent_targets + pred_equivalent_targets
            combined_equivalent_targets = list(set(combined_equivalent_targets))
            # Creating the new combined version
            del combined_target_object[eq_key]
            combined_target_object[eq_key] = combined_equivalent_targets
        combined_target_objects.append(combined_target_object)
    
    combined_dataset = TargetTextCollection(combined_target_objects)
    combined_dataset.to_json_file(save_fp)