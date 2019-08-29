import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Dict

from target_extraction.data_types import TargetText

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("augmented_training_dataset", type=parse_path, 
                        help='File path to the augmented training dataset')
    parser.add_argument("expanded_targets_fp", type=parse_path, 
                        help='File path to the expanded targets json file')
    args = parser.parse_args()

    with args.expanded_targets_fp.open('r') as expanded_targets_file:
        targets_equivalents: Dict[str, str] = json.load(expanded_targets_file)
    assert len(targets_equivalents) > 1

    expanded_target_counts = Counter()
    number_training_samples = 0
    number_targets_expanded = 0
    with args.augmented_training_dataset.open('r') as training_file:
        for line in training_file:
            training_sample = TargetText.from_json(line)
            number_targets = len(training_sample['targets'])
            number_training_samples += number_targets
            for target_index in range(number_targets):
                original_target = training_sample['targets'][target_index]
                if original_target not in targets_equivalents:
                    continue 
                number_targets_expanded += 1

                expanded_target_key = f'target {target_index}'
                expanded_targets = training_sample[expanded_target_key]
                assert original_target in expanded_targets
                number_expanded_targets = len(expanded_targets) - 1
                expanded_target_counts.update([number_expanded_targets])

    total_more_samples = 0
    number_targets_can_be_expanded = 0
    for number_expanded, count in expanded_target_counts.items():
        total_more_samples += (number_expanded * count)
        if number_expanded > 0:
            number_targets_can_be_expanded += count
    print(f'Number of training samples {number_training_samples}')
    print(f'Number of training samples that had targets that can be expanded {number_targets_expanded}')
    print(f'Number of samples that can be expanded {number_targets_can_be_expanded}')
    print(expanded_target_counts)
    print(f'Total more training samples from augmentation {total_more_samples}')