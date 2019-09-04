import argparse
from typing import Dict, Any, Iterable
from pathlib import Path

from target_extraction.allen.allennlp_model import AllenNLPModel
from target_extraction.data_types import TargetTextCollection, TargetText

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def target_iter(data: TargetTextCollection) -> Iterable[Dict[str, Any]]:
    for target_text in data.values():
        yield dict(target_text)

if __name__=='__main__':
    model_save_fp_help = "File Path to save the model. NOTE if N>1 then the "\
                         "first model will be saved. This needs to be a directory"\
                         " that does not exist currently."
    parser = argparse.ArgumentParser()
    parser.add_argument("train_fp", type=parse_path, 
                        help='File Path to the json training data')
    parser.add_argument("val_fp", type=parse_path, 
                        help='File Path to the json validation data')
    parser.add_argument("test_fp", type=parse_path, 
                        help='File Path to the json test data')
    parser.add_argument("config_fp", type=parse_path, 
                        help='File path to the model config')
    parser.add_argument("N", type=int, 
                        help='Number of times to run the model')
    parser.add_argument("save_fp", type=parse_path, 
                        help="File Path to the file to save the JSON data to")
    parser.add_argument("--model_save_dir", type=parse_path, 
                        help=model_save_fp_help)
    args = parser.parse_args()
    
    train_data = TargetTextCollection.load_json(args.train_fp)
    val_data = TargetTextCollection.load_json(args.val_fp)
    test_data = TargetTextCollection.load_json(args.test_fp)

    prediction_data = list(test_data.dict_iterator())

    number_runs = args.N
    for run in range(number_runs):
        if run == 0 and args.model_save_dir:
            model = AllenNLPModel('model', args.config_fp, 
                                  predictor_name='target-sentiment',
                                  save_dir=args.model_save_dir)
        else:
            model = AllenNLPModel('model', args.config_fp, 
                                  predictor_name='target-sentiment')
        model.fit(train_data, val_data, test_data)
        for pred_index, prediction in enumerate(model._predict_iter(prediction_data)):
            pred_target = prediction_data[pred_index]
            text_id = pred_target['text_id']
            test_target = test_data[text_id]
            if 'predicted_target_sentiments' not in test_target:
                test_target['predicted_target_sentiments'] = []
            test_target['predicted_target_sentiments'].append(prediction['sentiments'])
    test_data.to_json_file(args.save_fp)