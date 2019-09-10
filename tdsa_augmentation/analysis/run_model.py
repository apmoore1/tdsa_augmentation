import argparse
from typing import Dict, Any, Iterable, Optional
from pathlib import Path

from allennlp.models import Model
from target_extraction.allen.allennlp_model import AllenNLPModel
from target_extraction.data_types import TargetTextCollection, TargetText

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def predict_on(model: Model, prediction_data: Dict[str, Any], 
               prediction_target_collection: TargetTextCollection) -> None:
    '''
    :param model: Model that has been trained and will generate the predictions
    :param prediction_data: The data to feed to the model to be predicted on
    :param prediction_target_collection: The original format of the data being 
                                         predicted on that will have the predicted
                                         label added to
    '''
    for pred_index, prediction in enumerate(model._predict_iter(prediction_data)):
        pred_target = prediction_data[pred_index]
        text_id = pred_target['text_id']
        target_object = prediction_target_collection[text_id]
        if 'predicted_target_sentiments' not in target_object:
            target_object['predicted_target_sentiments'] = []
        target_object['predicted_target_sentiments'].append(prediction['sentiments'])

def run_model(train_fp: Path, val_fp: Path, test_fp: Path, 
              config_fp: Path, save_test_fp: Path, number_runs: int, 
              model_save_dir: Optional[Path] = None, 
              save_val_fp: Optional[Path] = None) -> None:
    '''
    :param train_fp: Path to file that contains JSON formatted training data
    :param val_fp: Path to file that contains JSON formatted validation data
    :param test_fp: Path to file that contains JSON formatted testing data
    :param config_fp: Path to file that contains the models configuration
    :param save_test_fp: Path to save the test data results
    :param number_runs: Number of times to run the model
    :param model_save_dir: Path to save the first trained model (optional)
    :param save_val_fp: Path to save the validation data results (optional)
    '''
    train_data = TargetTextCollection.load_json(train_fp)
    val_data = TargetTextCollection.load_json(val_fp)
    test_data = TargetTextCollection.load_json(test_fp)

    test_prediction_data = list(test_data.dict_iterator())
    if save_val_fp:
        val_prediction_data = list(val_data.dict_iterator())

    for run in range(number_runs):
        if run == 0 and model_save_dir:
            model = AllenNLPModel('model', config_fp, 
                                  predictor_name='target-sentiment',
                                  save_dir=model_save_dir)
        else:
            model = AllenNLPModel('model', config_fp, 
                                  predictor_name='target-sentiment')
        model.fit(train_data, val_data, test_data)
        predict_on(model, test_prediction_data, test_data)
        if save_val_fp:
            predict_on(model, val_prediction_data, val_data)
    test_data.to_json_file(save_test_fp)
    if save_val_fp:
        val_data.to_json_file(save_val_fp)

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
    parser.add_argument("save_test_fp", type=parse_path, 
                        help="File Path to save the test JSON prediction data to")
    parser.add_argument("--save_val_fp", type=parse_path, 
                        help="File Path to save the validation JSON prediction data to")
    parser.add_argument("--model_save_dir", type=parse_path, 
                        help=model_save_fp_help)
    args = parser.parse_args()
    run_model(args.train_fp, args.val_fp, args.test_fp, args.config_fp, 
              args.save_test_fp, args.N, args.model_save_dir, args.save_val_fp)