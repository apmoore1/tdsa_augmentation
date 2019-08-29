import argparse
from collections import Counter
import json
from pathlib import Path
from typing import List, Dict

from target_extraction.data_types import TargetTextCollection

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("target_list_data_fp", type=parse_path, 
                        help='File path to the list of targets')
    parser.add_argument("train_fp", type=parse_path, help='Train dataset')
    parser.add_argument("expanded_targets_fp", type=parse_path, 
                        help='File path to the expanded targets json file')
    args = parser.parse_args()

    with args.expanded_targets_fp.open('r') as expanded_targets_file:
        targets_equivalents: Dict[str, List[str]] = json.load(expanded_targets_file)

    train_targets = set(list(TargetTextCollection.load_json(args.train_fp).target_count(lower=True).keys()))
    
    targets = set()
    with args.target_list_data_fp.open('r') as target_list_file:
        for line in target_list_file:
            line = line.strip()
            if not line:
                continue
            targets.add(line)

    counts = Counter()
    train_counts = Counter()
    for target, equivalents in targets_equivalents.items():
        if target not in train_targets:
            continue
        count = 0
        train_count = 0
        for equivalent in equivalents:
            if equivalent in targets:
                count += 1
            if equivalent in targets and equivalent in train_targets:
                raise ValueError('something')
            if equivalent in train_targets:
                train_count += 1
        train_counts.update([train_count])
        counts.update([count])
        try:
            assert 15 == (train_count + count) 
        except:
            print(equivalents)
            import pdb
            pdb.set_trace()
    print(counts)
    print(train_counts)
    total = 0
    for i in range(16):
        if i in counts:
            total += i * counts[i]
        if i in train_counts:
            total += i * train_counts[i]
    print(total)
