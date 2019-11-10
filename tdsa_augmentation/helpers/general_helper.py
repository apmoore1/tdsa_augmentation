from typing import Dict, List, Optional

from gensim.models import KeyedVectors
from target_extraction.tokenizers import spacy_tokenizer

def multi_word_targets(targets: List[str], lower: bool = True,
                       string_delimiter: str = '_') -> Dict[str, str]:
    '''
    :param targets: A list of targets where multi word targets will have 
                    there whitespace replaced with `_` to create a single 
                    word target. Spacy tokenizer determines multi word targets. 
    :param lower: if to lower case the target words.
    :param string_delimiter: The string to be used to join the target words 
                             together after they have been tokenised by the 
                             spacy tokeniser.
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
        target_mapper[target] = string_delimiter.join(tokenized_target)
    assert_err = 'The length of the multi word targets is not the same '\
                 'as the non-multi-word targets'
    assert len(unique_targets) == len(target_mapper), assert_err
    return target_mapper

def extract_multi_word_targets(targets: List[str], 
                               fraction: float, embedding: KeyedVectors,
                               mw_targets_ignore: Optional[List[str]] = None, 
                               string_delimiter: str = '_', 
                               remove_single_word_targets: bool = False,
                               remove_existing_mw_targets_in_embedding: bool = False,
                               raise_value_error_on_single_target: bool = False
                               ) -> List[str]:
    '''
    :param targets: A list of targets.
    :param embedding: The embedding you want statistics on
    :param fraction: The fraction of target words within one target that is 
                     also in the embedding to declare the embedding can be 
                     used for this multi word target
    :param mw_targets_ignore: Any targets that should not be returned.
    :param string_delimiter: The string to state when a token is a token in 
                             a multi word target. The target words will be 
                             tokenised first using the spacy tokeniser through
                             the :py:meth:`tdsa_augmentation.general_helper.multi_word_targets`
    :param remove_single_word_targets: Removes the single word targets based on 
                                       the `string_delimiter`. This is done 
                                       after removing `mw_targets_ignore` and 
                                       before the `raise_value_error_on_single_target`
                                       check.
    :param remove_words_in_embedding: Removes multi word targets that already 
                                      exist in the embedding. This is done 
                                      after `remove_single_word_targets` and 
                                      `mw_targets_ignore` but before
                                      `raise_value_error_on_single_target` check.
                                      These multi-word targets that exist in 
                                      the embedding as a single vector.
    :param raise_value_error_on_single_target: Will raise a ValueError if 
                                               single word targets exist in the 
                                               list of targets after `mw_targets_ignore`
                                               have been removed.
    :returns: A list of targets that contain more than one word. Furthermore 
              these multi word targets must contain at least `fraction` amount 
              of words that are also within the embedding. Lastly any targets 
              that are within the `mw_targets_ignore` will not be returned.
    :raises ValueError: If `raise_value_error_on_single_target` True it will 
                        raise a ValueError if single word targets exist in the 
                        list of targets after `mw_targets_ignore` have been 
                        removed. 
    '''
    multi_targets = list(multi_word_targets(targets, lower=True, 
                                            string_delimiter=string_delimiter)\
                                            .values())
    if mw_targets_ignore is not None:
        # filter the targets
        for ignore_target in mw_targets_ignore:
            if ignore_target not in multi_targets:
                multi_targets.remove(ignore_target)
    if remove_single_word_targets:
        temp_multi_targets = []
        # Removes the single word targets
        for target in multi_targets:
            if string_delimiter in target:
                temp_multi_targets.append(target)
        multi_targets = temp_multi_targets

    if remove_existing_mw_targets_in_embedding:
        temp_multi_targets = []
        # Remove all multi word targets that are in the embedding:
        for target in multi_targets:
            if target not in embedding.wv:
                temp_multi_targets.append(target)
        multi_targets = temp_multi_targets

    filtered_mw_targets = set()
    for mw_target in multi_targets:
        words = mw_target.split(string_delimiter) 
        words_in_embedding = [word for word in words if word in embedding.wv]
        if raise_value_error_on_single_target and len(words) <= 1:
            raise ValueError(f'This target is not a multi word target {mw_target}')
        frac_words = len(words_in_embedding) / len(words)
        if frac_words >= fraction:
            filtered_mw_targets.add(mw_target)
    return list(filtered_mw_targets)

