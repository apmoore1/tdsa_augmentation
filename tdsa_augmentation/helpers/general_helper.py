from typing import List

from target_extraction.tokenizers import spacy_tokenizer

def multi_word_targets(targets: List[str], 
                       lower: bool = True) -> List[str]:
    '''
    :param targets: A list of targets where multi word targets will have 
                    there whitespace replaced with `_` to create a single 
                    word target. Spacy tokenizer determines multi word targets. 
    :param lower: if to lower case the target words.
    :returns: A list of the same legnth but multi word targets whitespace 
              will be replaced by `_` e.g. `tesco supermarket` would be 
              `tesco_supermarket`
    '''
    tokenizer = spacy_tokenizer()
    temp_targets = []
    for target in targets:
        if lower:
            target = target.lower()
        tokenized_target = tokenizer(target)
        temp_targets.append('_'.join(tokenized_target))
    assert_err = 'The length of the multi word targets is not the same '\
                 'as the non-multi-word targets'
    assert len(targets) == len(temp_targets), assert_err
    return temp_targets