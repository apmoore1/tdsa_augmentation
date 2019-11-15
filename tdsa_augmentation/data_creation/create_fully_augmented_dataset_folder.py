import argparse
from pathlib import Path
import shutil

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    save_help = 'File path to store the augmented training data and the '\
                'original validation and test data'
    parser = argparse.ArgumentParser()
    parser.add_argument("augmented_data_fp", type=parse_path, 
                        help='File path to the augmented training data')
    parser.add_argument("test_val_folder_fp", type=parse_path, 
                        help='File path to the folder containing the validation and test data')
    parser.add_argument("save_folder", type=parse_path, help=save_help)
    args = parser.parse_args()

    save_folder = args.save_folder
    save_folder.mkdir(parents=True, exist_ok=True)
    new_train_fp = Path(save_folder, 'train.json')
    shutil.copy(args.augmented_data_fp, new_train_fp)
    val_fp = Path(args.test_val_folder_fp, 'val.json')
    new_val_fp = Path(save_folder, 'val.json')
    shutil.copy(val_fp, new_val_fp)
    test_fp = Path(args.test_val_folder_fp, 'test.json')
    new_test_fp = Path(save_folder, 'test.json')
    shutil.copy(test_fp, new_test_fp)

