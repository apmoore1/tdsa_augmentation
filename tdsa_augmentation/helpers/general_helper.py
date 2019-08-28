from typing import Dict, List

from target_extraction.tokenizers import spacy_tokenizer

def multi_word_targets(targets: List[str], 
                       lower: bool = True) -> Dict[str, str]:
    '''
    :param targets: A list of targets where multi word targets will have 
                    there whitespace replaced with `_` to create a single 
                    word target. Spacy tokenizer determines multi word targets. 
    :param lower: if to lower case the target words.
    :returns: A dictionary of the original target and their multi words targets
              whitespace replacement version where the whitepsace is replaced 
              with `_` e.g. {`tesco supermarket`: `tesco_supermarket`}
    '''
    tokenizer = spacy_tokenizer()
    target_mapper = {}
    unique_targets = set()
    for target in targets:
        if lower:
            target = target.lower()
        unique_targets.add(target)
        tokenized_target = tokenizer(target)
        target_mapper[target] = '_'.join(tokenized_target)
    assert_err = 'The length of the multi word targets is not the same '\
                 'as the non-multi-word targets'
    assert len(unique_targets) == len(target_mapper), assert_err
    return target_mapper