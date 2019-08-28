import argparse
import json
from pathlib import Path

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    save_fp_help = 'File to save the predicted targets to, each target will be '\
                   'saved on a new line'
    parser = argparse.ArgumentParser()
    parser.add_argument("json_fp", type=parse_path, 
                        help='File path to json data')
    parser.add_argument("save_fp", type=parse_path, help=save_fp_help)
    args = parser.parse_args()

    all_targets = []
    with args.json_fp.open('r') as json_file:
        for line in json_file:
            targets = json.loads(line)['targets']
            if targets is None:
                continue
            all_targets.extend(targets)
    with args.save_fp.open('w+') as save_file:
        for index, target in enumerate(all_targets):
            if index == 0:
                save_file.write(f'{target}')
            else:
                save_file.write(f'\n{target}')
