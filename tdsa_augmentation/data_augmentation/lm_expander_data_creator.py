import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any, Callable, Set, Union

from allennlp.data.dataset_readers import SimpleLanguageModelingDatasetReader, DatasetReader
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import load_archive, Model, BidirectionalLanguageModel
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.nn.util import get_text_field_mask
from allennlp.data.dataset import Batch
from allennlp.nn import util
import numpy as np
from target_extraction.data_types import TargetText
from target_extraction.tokenizers import spacy_tokenizer
from target_extraction.data_types_util import OverLappingTargetsError
import torch

# This is form from allennlp.models import BidirectionalLanguageModel
def _forward(self,  # type: ignore
            source: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
    """
    Computes the averaged forward (and backward, if language model is bidirectional)
    LM loss from the batch.
    Parameters
    ----------
    source: ``Dict[str, torch.LongTensor]``, required.
        The output of ``Batch.as_tensor_dict()`` for a batch of sentences. By convention,
        it's required to have at least a ``"tokens"`` entry that's the output of a
        ``SingleIdTokenIndexer``, which is used to compute the language model targets.
    Returns
    -------
    Dict with keys:
    ``'loss'``: ``torch.Tensor``
        forward negative log likelihood, or the average of forward/backward
        if language model is bidirectional
    ``'forward_loss'``: ``torch.Tensor``
        forward direction negative log likelihood
    ``'backward_loss'``: ``torch.Tensor`` or ``None``
        backward direction negative log likelihood. If language model is not
        bidirectional, this is ``None``.
    ``'lm_embeddings'``: ``Union[torch.Tensor, List[torch.Tensor]]``
        (batch_size, timesteps, embed_dim) tensor of top layer contextual representations or
        list of all layers. No dropout applied.
    ``'noncontextual_token_embeddings'``: ``torch.Tensor``
        (batch_size, timesteps, token_embed_dim) tensor of bottom layer noncontextual
        representations
    ``'mask'``: ``torch.Tensor``
        (batch_size, timesteps) mask for the embeddings
    """
    # pylint: disable=arguments-differ
    mask = get_text_field_mask(source)

    # shape (batch_size, timesteps, embedding_size)
    embeddings = self._text_field_embedder(source)

    # Either the top layer or all layers.
    contextual_embeddings: Union[torch.Tensor, List[torch.Tensor]] = self._contextualizer(
            embeddings, mask
    )

    return_dict = {}

    # If we have target tokens, calculate the loss.
    token_ids = source.get("tokens")
    if token_ids is not None:
        assert isinstance(contextual_embeddings, torch.Tensor)

        # Use token_ids to compute targets
        forward_targets = torch.zeros_like(token_ids)
        forward_targets[:, 0:-1] = token_ids[:, 1:]

        if self._bidirectional:
            backward_targets = torch.zeros_like(token_ids)
            # This is only the same if all of the sentences are the same length
            # else the shorter sentences will have a </S> Special tokens
            backward_targets[:, 1:] = token_ids[:, 0:-1]
        else:
            backward_targets = None

        # add dropout
        contextual_embeddings_with_dropout = self._dropout(contextual_embeddings)

        # compute softmax loss
        forward_loss, backward_loss = self._compute_loss(contextual_embeddings_with_dropout,
                                                            embeddings,
                                                            forward_targets,
                                                            backward_targets)
        
        num_targets = torch.sum((forward_targets > 0).long())

        if num_targets > 0:
            if self._bidirectional:
                if getattr(self, 'batch_loss', None) is None:
                    average_loss = 0.5 * (forward_loss + backward_loss) / num_targets.float()
                else:
                    average_loss = 0.5 * (forward_loss + backward_loss)
            else:
                if getattr(self, 'batch_loss', None) is None:
                    average_loss = forward_loss / num_targets.float()
                else:
                    average_loss = forward_loss
        else:
            average_loss = torch.tensor(0.0).to(forward_targets.device)  # pylint: disable=not-callable

        if getattr(self, 'batch_loss', None) is None:
            self._last_average_loss[0] = average_loss.detach().item()

        if num_targets > 0:
            return_dict.update({
                    'loss': average_loss,
                    'forward_loss': forward_loss / num_targets.float(),
                    'backward_loss': (backward_loss / num_targets.float()
                                        if backward_loss is not None else None),
                    'batch_weight': num_targets.float()
            })
        else:
            # average_loss zero tensor, return it for all
            return_dict.update({
                    'loss': average_loss,
                    'forward_loss': average_loss,
                    'backward_loss': average_loss if backward_loss is not None else None
            })

    return_dict.update({
            # Note: These embeddings do not have dropout applied.
            'lm_embeddings': contextual_embeddings,
            'noncontextual_token_embeddings': embeddings,
            'mask': mask
    })

    return return_dict

# This is from from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
def _forward_eval(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # pylint: disable=invalid-name
    # evaluation mode, use full softmax
    if self.sparse:
        w = self.softmax_w.weight
        b = self.softmax_b.weight.squeeze(1)
    else:
        w = self.softmax_w
        b = self.softmax_b

    log_softmax = torch.nn.functional.log_softmax(torch.matmul(embeddings, w.t()) + b, dim=-1)
    if self.tie_embeddings and not self.use_character_inputs:
        targets_ = targets + 1
    else:
        targets_ = targets
    batch_loss = getattr(self, 'batch_loss', None)
    if batch_loss:
        return torch.nn.functional.nll_loss(log_softmax, targets_.long(), reduction='none')
    else:
        return torch.nn.functional.nll_loss(log_softmax, targets_.long(), reduction="sum")

def _loss_helper(self,  
                    direction: int,
                    direction_embeddings: torch.Tensor,
                    direction_targets: torch.Tensor,
                    token_embeddings: torch.Tensor) -> Tuple[int, int]:
    mask = direction_targets > 0
    # we need to subtract 1 to undo the padding id since the softmax
    # does not include a padding dimension

    # shape (batch_size * timesteps, )
    non_masked_targets = direction_targets.masked_select(mask) - 1

    # shape (batch_size * timesteps, embedding_dim)
    non_masked_embeddings = direction_embeddings.masked_select(
            mask.unsqueeze(-1)
    ).view(-1, self._forward_dim)
    # note: need to return average loss across forward and backward
    # directions, but total sum loss across all batches.
    # Assuming batches include full sentences, forward and backward
    # directions have the same number of samples, so sum up loss
    # here then divide by 2 just below
    if not self._softmax_loss.tie_embeddings or not self._use_character_inputs:
        loss = self._softmax_loss(non_masked_embeddings, non_masked_targets)
        batch_loss = getattr(self, 'batch_loss', None)
        if batch_loss:
            loss_reshaped = torch.zeros_like(direction_targets, 
                                             dtype=torch.float)
            loss_reshaped[mask] = loss
            loss = loss_reshaped.sum(1) / mask.sum(1, dtype=torch.float)
        return loss
    else:
        # we also need the token embeddings corresponding to the
        # the targets
        raise NotImplementedError("This requires SampledSoftmaxLoss, which isn't implemented yet.")
        # pylint: disable=unreachable
        non_masked_token_embeddings = self._get_target_token_embeddings(token_embeddings, mask, direction)
        return self._softmax(non_masked_embeddings,
                                non_masked_targets,
                                non_masked_token_embeddings)

def _compute_loss(self,
                      lm_embeddings: torch.Tensor,
                      token_embeddings: torch.Tensor,
                      forward_targets: torch.Tensor,
                      backward_targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

    # If bidirectional, lm_embeddings is shape (batch_size, timesteps, dim * 2)
    # If unidirectional, lm_embeddings is shape (batch_size, timesteps, dim)
    # forward_targets, backward_targets (None in the unidirectional case) are
    # self (batch_size, timesteps) masked with 0
    if self._bidirectional:
        forward_embeddings, backward_embeddings = lm_embeddings.chunk(2, -1)
        backward_loss = self._loss_helper(1, backward_embeddings, backward_targets, token_embeddings)
    else:
        forward_embeddings = lm_embeddings
        backward_loss = None

    forward_loss = self._loss_helper(0, forward_embeddings, forward_targets, token_embeddings)
    return forward_loss, backward_loss

def sentence_perplexitys(model: Model, 
                         dataset_reader: SimpleLanguageModelingDatasetReader, 
                         sentences: List[str]) -> List[float]:
    '''
    :param model: A loaded language model
    :param dataset_reader: Reader that can convert list of list of tokens 
                           into batches that can be fed into the language 
                           model.
    :param sentences: A pre-tokenized list of sentences where each sentence is 
                      still a String but has been tokenized.
    :returns: A list of perplexity scores where the the list represents the 
              perplexity score for each sentence given.
    '''
    sentence_instances = []
    for sentence in sentences:
        sentence_instances.append(dataset_reader.text_to_instance(sentence))
    results = model.forward_on_instances(sentence_instances)
    result_perplexitys = [math.exp(result['loss']) for result in results]
    return result_perplexitys

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__=='__main__':
    save_fp_help = 'File Path to save the expanded training dataset'
    maximum_perplexity_help = 'Maximum perplexity a sentence can have before'\
                              ' the target is denoted as not semantically equivalent'
    batch_size_help = "Batch size. The higher this is the more GPU memory "\
                      "required. Default 15."
    parser = argparse.ArgumentParser()
    parser.add_argument("expanded_targets_fp", type=parse_path, 
                        help='File path to the expanded targets json file')
    parser.add_argument("lm_fp", type=parse_path, 
                        help='File path to the language model')
    parser.add_argument("train_json_dataset_fp", type=parse_path, 
                        help="File path to the json training dataset")
    parser.add_argument("save_fp", type=parse_path, help=save_fp_help)
    parser.add_argument("--cuda", action="store_true", 
                        help="Whether to load the language model on to a GPU")
    parser.add_argument("--batch_size", type=int, 
                        help=batch_size_help)
    args = parser.parse_args()

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 15
    print(batch_size)

    # Loading the language model
    BidirectionalLanguageModel._loss_helper = _loss_helper
    BidirectionalLanguageModel._compute_loss = _compute_loss
    BidirectionalLanguageModel.forward = _forward
    SampledSoftmaxLoss._forward_eval = _forward_eval
    archive = load_archive(args.lm_fp)
    transformer_model = archive.model
    if args.cuda:
        transformer_model.cuda()
    else:
        transformer_model.cpu()
    transformer_model.eval()
    transformer_model.batch_loss = True
    transformer_model._softmax_loss.batch_loss = True
    # Load the dataset reader that came with the transformer model and ensure 
    # that the max sequence length is set to infinte so that we can analysis 
    # any length sentence (problem can occur with Memory (GPU espically))
    # if a sentence is really long).
    config = archive.config
    dict_config = config.as_dict(quiet=True)
    dataset_reader_config = config.get("dataset_reader")
    if dataset_reader_config.get("type") == "multiprocess":
        dataset_reader_config = dataset_reader_config.get("base_reader")
        if 'max_sequence_length' in dataset_reader_config:
            dataset_reader_config['max_sequence_length'] = None
    dataset_reader = DatasetReader.from_params(dataset_reader_config)
    dataset_reader.lazy = False

    with args.expanded_targets_fp.open('r') as expanded_targets_file:
        targets_equivalents: Dict[str, str] = json.load(expanded_targets_file)
    assert len(targets_equivalents) > 1

    args.save_fp.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = spacy_tokenizer()
    count = 0
    with args.save_fp.open('w+') as save_file:
        with args.train_json_dataset_fp.open('r') as train_dataset:
            for line in train_dataset:
                train_sample = TargetText.from_json(line)
                targets = train_sample['targets']
                # Ensure that each sample in the training set has targets sainty check
                raise_target_error = False
                if targets is None:
                    raise_target_error = True
                elif len(targets) == 0:
                    raise_target_error = True
                if raise_target_error:
                    raise ValueError(f'Training sample contains no targets {train_sample}')
                # Get the perplexity score that replacement targets have to beat 
                # which is defined by the perplexity score of the original sentence 
                original_sentence = ' '.join(tokenizer(train_sample['text']))
                max_perplexity = sentence_perplexitys(transformer_model, dataset_reader, 
                                                      [original_sentence])
                assert len(max_perplexity) == 1
                max_perplexity = max_perplexity[0] 

                index_targets: Dict[int, List[str]] = {}
                for index, target in enumerate(targets):
                    if target in targets_equivalents:
                        equivalent_targets = targets_equivalents[target]
                        new_target_sentences: List[str] = []
                        new_target_sentence_lengths: List[int] = []
                        for equivalent_target in equivalent_targets:
                            try:
                                new_target_sentence = train_sample.replace_target(index, equivalent_target)['text']
                            except OverLappingTargetsError:
                                # Reason why we can have not `new_target_sentences`
                                index_targets[index] = [target]
                                continue
                            new_tokenized_sentence = tokenizer(new_target_sentence)
                            new_target_sentence_lengths.append(len(new_tokenized_sentence))
                            new_target_sentence = ' '.join(new_tokenized_sentence)
                            new_target_sentences.append(new_target_sentence)
                        # To handle the case of errors caused by overlapping 
                        # targets which happens in the Election Twitter dataset
                        if new_target_sentences == []:
                            continue
                        # Sort the sentences by sentence length for batching reasons.
                        # The reason we want all the sentences to be of the same 
                        # length when calculating the loss is due to the fact that 
                        # the LM NN model adds a special character for sentences 
                        # that are longer than the shortest sentence.
                        sorted_target_indexs = np.argsort(new_target_sentence_lengths).tolist()
                        sorted_sentences = [(new_target_sentences[sort_index], new_target_sentence_lengths[sort_index]) 
                                            for sort_index in sorted_target_indexs]
                        # Create batches where all of the batches are of the 
                        # same length and no larger than batch_size
                        sentence_batchs: List[List[str]] = []
                        min_sentence_length = sorted_sentences[0][-1]
                        a_batch = []
                        first_sentence = True
                        for sentence, sentence_length in sorted_sentences:
                            if not first_sentence:
                                if (sentence_length > min_sentence_length or 
                                    batch_size == len(a_batch)):
                                    sentence_batchs.append(a_batch)
                                    a_batch = [] 
                            a_batch.append(sentence)
                            first_sentence = False
                            min_sentence_length = sentence_length
                        assert len(a_batch) > 0
                        sentence_batchs.append(a_batch)
                        # Calculate the perplexity scores
                        perplexities = []
                        for batch in sentence_batchs:
                            batch_perplexity = sentence_perplexitys(transformer_model, 
                                                                    dataset_reader, 
                                                                    batch)
                            perplexities.extend(batch_perplexity)
                        # sort the perplexity scores back to the original 
                        # sentence ordering
                        temp_perplexities = [0] * len(sorted_target_indexs)
                        for value_index, sort_index in enumerate(sorted_target_indexs):
                            temp_perplexities[sort_index] = perplexities[value_index]
                        perplexities = temp_perplexities
                        # Filter the targets so that only those that have a lower 
                        # perplexity than the original target are kept
                        filtered_equivalent_targets = [target]
                        for perplexity_index, perplexity in enumerate(perplexities):
                            if perplexity <= max_perplexity:
                                equivalent_target = equivalent_targets[perplexity_index]
                                filtered_equivalent_targets.append(equivalent_target)
                        index_targets[index] = filtered_equivalent_targets
                    else:
                        index_targets[index] = [target]
                train_sample_dict = copy.deepcopy(dict(train_sample))
                index_targets =  {f'target {index}': targets 
                                  for index, targets in index_targets.items()}
                new_train_sample = {**index_targets, **train_sample_dict}
                new_train_sample = TargetText(**new_train_sample)
                if count == 0:
                    save_file.write(new_train_sample.to_json())
                else:
                    save_file.write(f'\n{new_train_sample.to_json()}')
                count += 1

    