import argparse
import copy
from typing import List
from pathlib import Path

from target_extraction.data_types import TargetText, TargetTextCollection
from target_extraction.data_types_util import OverLappingTargetsError

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def add_augmented_targets(target_object: TargetText, 
                          remove_repeats: bool = True) -> List[TargetText]:
    '''
    :param target_object: Target Text that contains all of the augmented targets
    :param remove_repeats: If True for Target sentences that contain more than 
                           one target, for the first target the original targets 
                           will be kept for all other targets in the sentence 
                           the original target will not be kept. This is to 
                           avoid repeats of the original targets in the target
                           sequence.
    :returns: A list of TargetText object each one with a different augmented 
              target apart from the one which contains the original targets.
    :Note: The returned TargetText objects will not contain the list of 
           augmented targets to save memory.
    '''
    target_copy = copy.deepcopy(target_object)
    # remove all of the augmented targets so that that the target copy can then 
    # be further used as the replace targets object (this is done to save 
    # memory)
    number_targets = len(target_copy['targets'])
    for target_index in range(0, number_targets):
        del target_copy[f'target {target_index}']
    if 'pos_tags' in target_copy:
        del target_copy['pos_tags']
    if 'tokenized_text' in target_copy:
        del target_copy['tokenized_text']
    all_targets = []
    for target_index in range(0, number_targets):
        un_augmented_target = target_object['targets'][target_index]
        augmented_targets = target_object[f'target {target_index}']
        original_target_err = (f'The original target {un_augmented_target}, '
                               f'should be in the list with the augmented '
                               f'targets: {augmented_targets}\nTarget object '
                               f'that errored: {target_object}')
        assert un_augmented_target in augmented_targets, original_target_err
        if target_index > 0 and remove_repeats:
            augmented_targets.remove(un_augmented_target)
        for augmented_index, augmented_target in enumerate(augmented_targets):
            try:
                aug_target_object = target_copy.replace_target(target_index, 
                                                            augmented_target)
                text_id = aug_target_object['text_id']
                new_text_id = f'{text_id}:{target_index}:{augmented_index}'
                aug_target_object['original_text_id'] = text_id
                aug_target_object._storage['text_id'] = new_text_id
                all_targets.append(aug_target_object)
            except OverLappingTargetsError:
                # This needs to be skipped as when targets overlap it is very 
                # difficult to easily calculate all possible span offsets 
                # for all other targets. Furthermore there are only 3 
                # occasion this happens so it is a very rare occurrence.
                continue
    return all_targets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("augmented_dataset", type=parse_path, 
                        help='File path the augmented dataset')
    parser.add_argument("save_fp", type=parse_path, 
                        help='File path to save the new re-formated augmented dataset')
    args = parser.parse_args()

    augmented_data_fp = args.augmented_dataset
    save_fp = args.save_fp

    augmented_dataset = TargetTextCollection.load_json(augmented_data_fp)
    new_dataset = []

    for target_object in augmented_dataset.values():
        augmented_targets = add_augmented_targets(target_object, 
                                                  remove_repeats=True)
        new_dataset.extend(augmented_targets)
    new_dataset = TargetTextCollection(new_dataset)
    number_samples = new_dataset.number_targets()
    print(f'The number of samples in the dataset {number_samples}')
    new_dataset.to_json_file(save_fp)