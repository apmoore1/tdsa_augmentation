import argparse
import json
from pathlib import Path
from typing import List

from target_extraction.data_types import TargetText, TargetTextCollection

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    train_fp_help = 'File path to the split training dataset (should be in json format)'
    confidence_score_help = "The confidence that the predicted target is a "\
                            "target based on the model's confidence scores"
    save_fp_help = 'File to save the predicted targets to, each target will be '\
                   'saved on a new line'
    parser = argparse.ArgumentParser()
    parser.add_argument("predicted_target_data_fp", type=parse_path, 
                        help='File path to the predicted targets')
    parser.add_argument("train_fp", type=parse_path, 
                        help=train_fp_help)
    parser.add_argument('confidence_score', type=float, 
                        help=confidence_score_help)
    parser.add_argument('save_fp', type=parse_path, help=save_fp_help)
    args = parser.parse_args()

    # Setting the data up
    train_data = TargetTextCollection.load_json(args.train_fp)
    train_targets = set(list(train_data.target_count(lower=True).keys()))


    acceptable_confidence = args.confidence_score
    all_targets: List[TargetText] = []
    with args.predicted_target_data_fp.open('r') as predicted_file:
        for index, line in enumerate(predicted_file):
            target_data = json.loads(line)
            target_id = str(index)
            target_data_dict = {'text': target_data['text'], 'text_id': target_id,
                                'confidences': target_data['confidence'], 
                                'sequence_labels': target_data['sequence_labels'], 
                                'tokenized_text': target_data['tokens']}
            target_data = TargetText.target_text_from_prediction(**target_data_dict, 
                                                                confidence=acceptable_confidence)
            if target_data['targets']:
                all_targets.append(target_data)
    print(len(all_targets))
    all_targets: TargetTextCollection = TargetTextCollection(all_targets)
    pred_target_dict = all_targets.target_count(lower=True)
    pred_targets = set(list(pred_target_dict.keys()))

    print(f'Number of unique targets in training dataset {len(train_targets)}')
    print(f'Number of unique predicted targets {len(pred_targets)}')
    pred_difference = pred_targets.difference(train_targets)
    print(f'Number of unique targets in predicted but not in training {len(pred_difference)}')
    train_difference = train_targets.difference(pred_targets)
    print(f'Number of unique targets in training but not in predicting {len(train_difference)}')
    overlap = train_targets.intersection(pred_targets)
    print(f'Number of unique targets that intersect predicting and training {len(overlap)}')
    
    # Saving only the predicted targets that are not in the training data
    with args.save_fp.open('w+') as save_file:
        for target_index, target in enumerate(pred_difference):
            target = target.strip()
            if not target:
                continue
            if target_index == 0:
                save_file.write(target)
            else:
                save_file.write(f'\n{target}')