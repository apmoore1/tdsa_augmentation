# tdsa_augmentation


## Word Vectors and Languages models
This repository assumes that you have followed the steps within the following Github repository to create Domain Specific (DS) pre-trained ELMo Transformer Language Models (LM) and pre-trained DS word embeddings: [repo](https://github.com/apmoore1/language-model). 

### Word Vectors
The Word vectors created from this repository should be stored in the following folder: `./resources/word_embeddings` along with the Glove 840 billion token 300 dimension word vector (Glove 840) and the 200 dimension Twitter Glove vector (Glove Twitter), both can be downloaded from [here](https://nlp.stanford.edu/projects/glove/).

### Language Models (LM)
The following LMs should be stored in this folder: `./resources/language_models`

1. The non-domain specific ELMo Transformer from the following [paper](https://www.aclweb.org/anthology/D18-1179) that can be found [here](https://s3-us-west-2.amazonaws.com/allennlp/models/transformer-elmo-2019.01.10.tar.gz) and stored at `./resources/language_models/transformer-elmo-2019.01.10.tar.gz`
2. The domain specific ELMo Transformers created from the [repo](https://github.com/apmoore1/language-model) described above. There should be three LMs and each saved within `./resources/language_models/` under;
    * `restaurant_model.tar.gz` for Yelp (Restaurant domain). 
    * `laptop_model.tar.gz` Amazon (Laptop domain).
    * `election_model.tar.gz` MP/Election.

A **NOTE** on the domain specific ELMo Transformers ensure that within the `tar.gz` file that the `config.json` does not still have the `initializer` field containing a link to the pre-trained weights, if so remove that field else the LM will not work.

## TDSA datasets and creating training, validation, and test datasets.
The datasets we are going to look at are the following of which follow the relevant instructions on downloading and where to download them too: 
1. [SemEval Restaurant and Laptop 2014](https://www.aclweb.org/anthology/S14-2004/) [train](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-train-data-v20-annotation-guidelines/683b709298b811e3a0e2842b2b6a04d7c7a19307f18a4940beef6a6143f937f0/) and [test] and put the `Laptop_Train_v2.xml` and `Restaurants_Train_v2.xml` training files into the following directory `./data` and do the same for the test files (`Laptops_Test_Gold.xml` and `Restaurants_Test_Gold.xml`)
2. [Twitter Election dataset](https://www.aclweb.org/anthology/E17-1046/). No need to download this as this is done automatically in the next step, where it will be downloaded into the `./data` folder.

To create the training, validation and testing datasets for the Sentiment prediction task of TDSA run the following command:
```
./tdsa_augmentation/data_creation/create_sentiment_splits.sh
```
The validation and test sets are roughly equal in size.

## Augmenting the datasets through existing targets
### Extract Targets
In this approach we are only going to use the targets that have occurred in the training dataset to augment the current training datasets. First we need to extract out all of the targets in the training datasets and store each of them in `./resources/target_words` as: 
* `restaurant_train.txt` 
* `laptop_train.txt`
* `election_train.txt`:
``` bash
./tdsa_augmentation/data_creation/extract_targets_from_train_json.sh 
```
### Target Embedding/Vectors statistics 
#### DS Word Embedding statistics
The starting point of our data augmentation first starts by finding candidate targets for each target where the candidates are semantically equivalent targets for each target. To do this we are going to use our DS word embeddings to find those equivalent targets. To do so we first want to know how many of the targets (lower cased) can be represented by our DS embeddings:
``` bash
./tdsa_augmentation/statistics/target_in_embedding_stats.sh 1.0 DS
```
The argument 1.0 determines the fraction of the number of words that have to be in a multi word target for the target to be represented by the embedding through averaging the words vectors within the target. The `DS` argument states that we want to run this for the DS word embeddings.

| Dataset    | Num Targets (In Embedding)  | TL 1        | TL 2       | TL 3+     |
|------------|-----------------------------|-------------|------------|-----------|
| Laptop     | 739 (730)                   | 280 (275)   | 296 (292)  | 163 (163) |
| Restaurant | 914 (888)                   | 379 (368)   | 326 (314)  | 209 (206) |
| Election   | 1496 (1221)                 | 936 (746)   | 429 (362)  | 131 (113) |

Where TL stands for Target Length e.g. TL 2 is a multi word target that is made up of 2 tokens e.g. "camera lens". However TL 3+ contains targets that contain at least 3 tokens thus it can contain targets that have 10 tokens. 

Across all of the dataset we have a very high TC 98.78%, 97.16%, and 81.62% for the Laptop, Restaurant, and Election datasets respectively. The Election dataset has the worse coverage which is not un-expected as it comes from Twitter which will have a high lexical diversity. The Multi-Word (MW) targets have a high coverage due to the embedding containing a lot of the individual words within the target and thus can represent the target as an average of it's words embeddings. However the number MW targets that the embedding contains without having to average the words is actually very low as shown in the table below:

| Dataset    | TL 2       | TL 3+     |
|------------|------------|-----------|
| Laptop     | 296 (98)   | 163 (0)   |
| Restaurant | 326 (66)   | 209 (0)   |
| Election   | 429 (166)  | 131 (3)   |

#### Glove Embedding statistics
For comparison we are also going to explore how many words the general Glove Vectors can represent for these datasets. First for computational and practical reasons we are going to shrink the Glove Vectors so that they only contain words that are in the targets for each of the three datasets, and then store these words and there vectors in Word2Vec format.
``` bash
./tdsa_augmentation/data_creation/shrink_glove_to_word2vec.sh
```
Once this has ran the shrunk Glove Vectors can be found in the following directory `./resources/word_embeddings/shrunk_target_vectors/`, where we have a Word2Vec for each dataset and 2 for the Election dataset as we have shrunk both Glove 300 and Glove Twitter where as for the others only the Glove 300 has been shrunk. 

Now we can find the number of targets within the general Glove embeddings:
``` bash
./tdsa_augmentation/statistics/target_in_embedding_stats.sh 1.0 GLOVE
```

| Dataset                     | Num Targets (In Embedding)  | TL 1        | TL 2       | TL 3+     |
|-----------------------------|-----------------------------|-------------|------------|-----------|
| Laptop                      | 739 (736)                   | 280 (278)   | 296 (295)  | 163 (163) |
| Restaurant                  | 914 (882)                   | 379 (364)   | 326 (314)  | 209 (204) |
| Election                    | 1496 (1064)                 | 936 (636)   | 429 (324)  | 131 (104) |
| Election (Twitter Vectors)  | 1496 (1090)                 | 936 (651)   | 429 (331)  | 131 (108) |

Coverage difference:

| Dataset    | DS Coverage | Glove Coverage | DS - Glove |
| -----------|-------------|----------------|------------|
| Laptop     | 98.78%      | 99.59%         | -0.81%     |
| Restaurant | 97.16%      | 96.5%          | 0.66%      |
| Election   | 81.62%      | 72.86%         | 8.76%      |

We can see the Glove vectors and DS are very similar in TC for the review datasets but for the Twitter the DS has a much higher coverage. Having a high coverage will mean that more samples will be able to be augmented and more likely to be able to find more semantically equivalent targets to augment with. Unlike the DS embedding the Glove embeddings do not contain MW targets without averaging.

### Finding semantically similar targets through word embeddings.
#### Using only the original data
##### Finding similar targets through the DS word embedding
We thus want to use these embeddings to find targets that are semantically similar, therefore for each target within the TC find the top *N* most similar target words within the TC. In our case we use *N=15* (15 is just an arbitrary number but the higher this number is the more targets that the LM will have to consider in the next step, which is the more computationally expensive step. Therefore the lower *N* is the quicker the augmentation.):
``` bash
./tdsa_augmentation/data_augmentation/train_targets_embedding_expander.sh
```
All of the expanded target words can be found at `./resources/data_augmentation/target_words` in the following files: 
* `laptop_train_expanded.json`
* `restaurant_train_expanded.json`
* `election_train_expanded.json`

##### Narrowing the similar targets through a LM
Now that we have some strong semantic equivalent candidate words, we can shrink these candidates down further based on the context the original target originates from. To do this for each target and for each context the target appears in through out the training dataset the target will be replaced with one of the semantic equivalent target words. Each time a target is replaced in the training sentence if the LMs perplexity score for that sentence is less than (less in perplexity is better) the perplexity score of the original sentence then the target will be added to that training instance. **NOTE** as we are using the LMs per training instance, each target can have a different semantic equivalent target lists per training instance as the LM will have a different score for each target based on how well it fits into the sentence.

The LMs we will use are the DS LMs, not the non-DS LM. This form of data augmentation is similar to combining the following two papers [1](https://www.aclweb.org/anthology/D15-1306/), [2](https://www.aclweb.org/anthology/P19-1328/) the former performs data augmentation based on top N similar words from a word embedding and the latter shows that using a BERT model's similarity between the original and the word substitution sentences are useful for evaluating lexical substitution.* 

Thus we are now going to create the fully expanded training dataset for each of the domains/datasets. This dataset will be exactly the same as the original training datasets that are currently json files, however there will be extra fields denoted by the index of the target where it will contain a list of semantically equivalent targets for that target at that index. An example of one sentence that contains multiple targets and it's equivalent targets is shown below: 
``` json
{"text": "It appears that many lefties are unable to tell the difference between tax evasion and tax avoidance #bbcqt", "text_id": "68544488139964416", "targets": ["tax avoidance", "tax evasion"], "spans": [[87, 100], [71, 82]], "target_sentiments": ["neutral", "neutral"], "target 0": ["tax avoidance", "tax evasion"], "target 1": ["tax evasion", "libor", "tax avoidance"]}
```
To create these expanded training datasets run the following bash script which will produce expanded datasets for the laptop, restaurant and election domain at the following paths respectively: 
* `./data/augmented_train/laptop.json`
* `./data/augmented_train/restaurant.json`
* `./data/augmented_train/election.json`

``` bash
./tdsa_augmentation/data_augmentation/augmented_lm_train_dataset.sh
```

Now that we have the augmented training dataset we shall see how many new training samples we have (to get this table data run the command above the table and below this):

``` bash
./tdsa_augmentation/statistics/training_num_additional_targets.sh
```

| Dataset    | Num samples  | Can be expanded | Expanded | More samples |
|------------|--------------|-----------------|----------|--------------|
| Laptop     | 1661         | 1040            |    528   | 1338         |
| Restaurant | 2490         | 1829            |    675   | 1774         |
| Election   | 6811         | 2370            |    1794  | 7184         |

We can see from the table above the limited number of targets in the word embedding reduces the number of targets that can be expanded by up to 66% shown by the `Can be expanded` column, further more the language model reduces this further as shown by the `Expanded` column. However in the case of the Election dataset the language model finds out of the *15* candidate targets (15 comes from the fact the embedder expanded each target by 15 (the *N* parameter)) a large proportion of them are good candidates. Lastly the fact that the language model finds a lot of these candidates to be good candidates it still only expands the dataset by just over 100% in the best case and 71% in the worse case. This expansion is fairly small considering it can be between 510% and 1095% of the training dataset if all 15 candidate targets are accepted by the language model. Furthermore the larger the choice of different semanticaly equivalent targets per target the more flexible the training dataset expansion can be.

**What could be a good idea is to plot the target expander distribution and show that**

To create a potentially larger training dataset and to induce more unseen targets into the dataset we are going to find new targets through Semi-Supervision.

## Finding new Targets by through semi-supervision
In this section we will train a state of the art Target Extraction system to extract targets from large un-labeled text corpora. The state of the art Target Extraction system is simply a Bi-Directional LSTM with 50 hidden units that has been given two word representations:
1. The 840 billion token 300 dimension Glove vector.
2. The output of a domain specific Bi-Directional transformer language model, which creates the contextualized word representation.
This architecture is very similar to... the only difference being the addition of the contextualized word representation as input.

### Large un-labeled text corpora
As the main point of this paper is to create more data by using already existing Target training data but changing the target from the original to a semantically equivalent target, where this target could come from a target that we already know from the Target training data or an unknown target that has been predicted on a large unlabelled dataset with high confidence. We thus need a large unlabelled dataset within the domain of the target data. Therefore we are going to use the filtered training tokenized language model training dataset that were used to create the domain specific language models (the details to create this dataset are at the following [repo](https://github.com/apmoore1/language-model)). We expect the filtered language model training datasets for the following domains Restaurant, Laptop, and Twitter Election at the respective paths `yelp_filtered_split_train.txt`, `amazon_filtered_split_train.txt`, `election_filtered_split_train.txt` within the `./resources/language_model_datasets` directory.

### Training and Predicting on the large un-labeled text corpora
Here for each of the domains; 1. Restaurant, 2. Laptop, 3. Election we are going to: 
1. Train a Target Extraction system 
2. Predict on the relevant large un-labeled text corpora
3. Extract out only the targets we have a 90% confidence are targets, where all of the target word(s) have at least 90% confidence they are that label.

All trained Target Extraction systems are saved within the `./resources/data_augmentation/target_extraction_models` directory under their own directory; 1. `restaurant`, 2. `laptop`, 3. `election`. After training the respective Extraction systems we get the following F1 scores respectively:
1. 88.47, current best on this dataset is [85.61](https://www.ijcai.org/proceedings/2018/0583.pdf). (Took around 70 minutes to predict on the relevant large un-labeled text corpora).
2. 86.59, current best on this dataset is [84.26](https://www.aclweb.org/anthology/N19-1242). (Took around 82 minutes to predict on the relevant large un-labeled text corpora).
3. 89.02, no paper to compare this score to. (Took around 103 minutes to predict on the relevant large un-labeled text corpora).

To re-create these results and create the predicted target extraction sequences from the large un-labeled text corpora which can be found in `./resources/predicted_targets` and thus steps 1 and 2 from above run the following command:
``` bash
./tdsa_augmentation/data_augmentation/run_train_predict.sh
```

As the Target Extraction systems have been trained and have outputted there predictions on the `relevant large un-labeled text corpora` we can now extract out the 90% confident targets for each dataset and store them in `./resources/target_words` as `restaurant_predicted.txt`, `laptop_predicted.txt`, and `election_predicted.txt`. The targets that have been stored are not all of the targets just those that do not appear in the relevant training datasets:
``` bash
./tdsa_augmentation/data_augmentation/extract_predicted_targets.sh
```

All of the statistics below are number of unique lower cased targets from their relevant sources.

| Dataset    | Num in Train (T) | Num Predicted (P) | P\T     | T\P | $P\capT$ |
|------------|------------------|-------------------|---------|-----|----------|
| Laptop     | 739              | 22,923            | 22,483  | 299 | 440      |
| Restaurant | 914              | 50,697            | 50,129  | 346 | 568      |
| Election   | 1496             | 148,179           | 147,390 | 707 | 789      |

As we can the majority of the predicted targets have never been seen before in the training set, further more there are some targets (T\P) that have only ever been seen in the training, which is great as this means the training targets are still useful and high quality.

Next we are going to call
``` bash
./tdsa_augmentation/statistics/predicted_target_in_embedding_stats.sh
```

| Dataset    | Num Targets (In Embedding)  | TL 1             | TL 2           | TL 3        |
|------------|-----------------------------|------------------|----------------|-------------|
| Laptop     | 22,483 (5,922)              | 4,971 (4,131)    | 12,358 (1,791) | 5,154 (0)   |
| Restaurant | 50,129 (8,597)              | 5,661 (5,087)    | 24,600 (3,510) | 19,868 (0)  |
| Election   | 147,390 (51,572)            | 123,248 (44,426) | 21,929 (6,532) | 2,213 (614) |

As we can see even though we have extracted a lot of targets that have never been seen only a subset of them can be used due to the embedding not containing all of the targets.

We now need to combine the targets from the training with the predicted targets so that we can expand the training dataset using both the predicted and training targets.
``` bash
./tdsa_augmentation/combine_target_words.sh
```
This creates three new target word list within the following folder `./resources/target_words` for the laptop, restaurant, and election datasets respectively with these file names `all_laptop.txt`, `all_restaurant.txt`, `all_election.txt`

Like before we are now going to use these predicted and training targets to find similar targets using the embedder for the training samples:
``` bash
./tdsa_augmentation/data_augmentation/predicted_train_targets_embedding_expander.sh
```
All of the expanded target words can be found at `./resources/data_augmentation/target_words` in the following files; 1. `laptop_predicted_train_expanded.json`, 2. `restaurant_predicted_train_expanded.json`, 3. `election_predicted_train_expanded.json`

We want to find out the number of targets that came from the predicted targets that are in the *N* most similar targets.
``` bash

```

To create these expanded training datasets run the following bash script which will produce expanded datasets for the laptop, restaurant and election domain at the following paths respectively `./data/augmented_train/laptop_predicted.json`, `./data/augmented_train/restaurant_predicted.json`, `./data/augmented_train/election_predicted.json`
``` bash
./tdsa_augmentation/data_augmentation/augmented_lm_predicted_train_dataset.sh
```


python tdsa_augmentation/statistics/number_additional_targets.py ./data/augmented_train/laptop_predicted.json ./resources/data_augmentation/target_words/laptop_predicted_train_expanded.json

python tdsa_augmentation/statistics/number_target_in_similar.py ./resources/target_words/laptop_predicted.txt ./data/original_laptop_sentiment/train.json ./resources/data_augmentation/target_words/laptop_predicted_train_expanded.json

## Baseline Scores
First we need to convert our domain specific Word2Vec models () into Glove `.txt` format to do so run the following script:

``` bash
./tdsa_augmentation/analysis/convert_ds_embeddings.sh
```

This will then create three new embedding files in the Glove `.txt` format that are within `./resource/word_embeddings` directory. For the election, laptop, and restaurant embeddings it converts the following `mp_300`, `amazon_300`, `yelp_300` to respectively `election_glove.txt`, `laptop_glove.txt`, `restaurant_glove.txt`.

The models we are experimenting on throughout are the following:
1. [IAN](https://www.ijcai.org/proceedings/2017/0568.pdf)
2. [TDLSTM](https://www.aclweb.org/anthology/C16-1311)
3. [InterAE](https://www.aclweb.org/anthology/N18-2043)

Of which each model is going to be ran with a different word representations as input:
1. 300D Glove embeddings
2. Domain Sepcific (DS) word2vec embeddings
3. Elmo with 300D Glove embeddings
4. DS Elmo with 300D Glove embeddings
5. DS Elmo with DS word2vec embeddings

Each model is trained 5 times with the 5 different word representations stated above across the 3 different datasets using the following script.
``` bash
./tdsa_augmentation/analysis/run_baselines.sh $(which python) ./save_data/original/ 8 ./training_configs/
```
The test and validation prediction for each of these permutations is stored within the `./save_data/original` folder where the folders are structured as follows:

`model_name/dataset_name/word_representation`

Then within the `word_representation` folder are two json files `pred_test.json` and `pred_val.json`.

## Cross domain performance
Each of the models in the baselines can be trained on one dataset but then applied to the other datasets to test the cross domain performance of these models.




with N = 15
Number of training samples 1661
Number of training samples that had targets that can be expanded 1040
Number of samples that can be expanded 537
Counter({0: 503, 1: 190, 2: 114, 3: 71, 4: 51, 5: 40, 6: 18, 7: 12, 10: 10, 8: 10, 9: 6, 12: 4, 15: 4, 14: 3, 13: 2, 11: 2})
Total more training samples from augmentation 1659


Number of training samples 1661
Number of training samples that had targets that can be expanded 1040
Number of samples that can be expanded 664
Counter({0: 376, 1: 222, 2: 109, 3: 80, 4: 55, 5: 43, 7: 27, 6: 27, 8: 25, 9: 14, 11: 8, 13: 7, 10: 7, 20: 5, 14: 4, 18: 4, 17: 4, 12: 4, 19: 3, 16: 3, 23: 2, 27: 2, 15: 2, 30: 2, 25: 2, 24: 1, 26: 1, 21: 1})
Total more training samples from augmentation 2801


Number of training samples 1661
Number of training samples that had targets that can be expanded 1040
Number of samples that can be expanded 641
Counter({0: 399, 1: 216, 2: 100, 3: 79, 4: 55, 5: 41, 6: 27, 7: 26, 8: 24, 9: 13, 11: 9, 10: 7, 20: 5, 14: 4, 13: 4, 18: 4, 17: 4, 12: 4, 19: 3, 16: 3, 23: 2, 27: 2, 15: 2, 30: 2, 25: 2, 22: 1, 26: 1, 21: 1})
Total more training samples from augmentation 2710