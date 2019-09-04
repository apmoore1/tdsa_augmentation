# tdsa_augmentation


## Word Vectors and Languages models
This repository assumes that you have followed the steps within the following Github repository to create domain specific pre-trained ELMo Transformer language models and pre-trained domain specific word embeddings: [repo](https://github.com/apmoore1/language-model). 

### Word Vectors
The Word vectors created from this repository should be stored in the following folder: `./resources/word_embeddings` along with the Glove 840 billion token 300 dimension word vector that can be downloaded from [here](https://nlp.stanford.edu/projects/glove/).

### Language Models
The following language models should be stored in this folder: `./resources/language_models`

1. The non-domain specific ELMo Transformer from the following [paper](https://www.aclweb.org/anthology/D18-1179) that can be found [here](https://s3-us-west-2.amazonaws.com/allennlp/models/transformer-elmo-2019.01.10.tar.gz) and stored at `./resources/language_models/transformer-elmo-2019.01.10.tar.gz`
2. The domain specific ELMo Transformers created from the [repo](https://github.com/apmoore1/language-model) described above. There should be three language models here; 1. Yelp (Restaurant domain), 2. Amazon (Laptop domain), 3. MP/Election, all of these should be saved in the `./resources/language_models/` folder under the following names respectively; `yelp_model.tar.gz`, `amazon_model.tar.gz`, and `election_model.tar.gz`.

A **NOTE** on the domain domain specific ELMo Transformers ensure that within the `tar.gz` file that the `config.json` does not still have the `initializer` field containing a link to the pre-trained weights, if so remove that field else the language model will not work.

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
In this approach we are only going to use the targets that have occurred in the training dataset to augment the current training datasets. First we need to extract out all of the targets in the training datasets and store each of them in `./resources/target_words` as `restaurant_train.txt`, `laptop_train.txt`, and `election_train.txt`:
``` bash
./tdsa_augmentation/data_creation/extract_targets_from_train_json.sh 
```

The starting point of our data augmentation first starts by finding candidate targets for each target where the candidates are semantically equivalent targets for each target. To do this we are going to use our domain specific word embeddings to find those equivalent targets. To do so we first want to know how many of the targets (lower cased) are in our domain specific embeddings:
``` bash
./tdsa_augmentation/statistics/target_in_embedding_stats.sh
```
| Dataset    | Num Targets (In Embedding)  | TL 1        | TL 2       | TL 3     |
|------------|-----------------------------|-------------|------------|----------|
| Laptop     | 739 (373)                   | 280 (275)   | 296 (98)   | 163 (0)  |
| Restaurant | 914 (434)                   | 379 (368)   | 326 (66)   | 209 (0)  |
| Election   | 1496 (915)                  | 936 (746)   | 429 (166)  | 131 (3)  |

Where TL stands for Target Length e.g. TL 2 is a multi word target that is made up of 2 tokens e.g. "camera lens".

As we can see the embedding has a high coverage of targets that are not Multi Word Expressions (MWE), but do capture some MWEs and overall cover a minimum of 47% of the target words in all of the datasets.

We thus want to use these embeddings to find targets that are semantically similar, therefore for each target find the top *N* most similar target words. In our case we use *N=15* (15 is just an arbitrary number we have chosen and will be better tuned in the later process when using the language models):
``` bash
./tdsa_augmentation/data_augmentation/train_targets_embedding_expander.sh
```
All of the expanded target words can be found at `./resources/data_augmentation/target_words` in the following files; 1. `laptop_train_expanded.json`, 2. `restaurant_train_expanded.json`, 3. `election_train_expanded.json`

Now that we have some strong semantic equivalent candidate words, we can shrink these candidates down further based on the context the original target originates from. To do this for each target and for each context the target appears in through out the training dataset the target will be replaced with one of the semantic equivalent target words. Each time a target is replaced in the training sentence if the language models perplexity score for that sentence is less than (less in perplexity is better) the perplexity score of the original sentence then the target will be added to that training instance. **NOTE** As we are using the language models per training instance, each target can have a different semantic equivalent target lists per training instance as the language model will have a different score for each target based on how well it fits into the sentence.

*The reason why we filter the targets using the Word Vectors and then use the language models to fine tune the semantic equivalent targets is due to the time it would take to run the language models for each target against each target for each training instance. This is very similar to a combination of the following two papers [1](https://www.aclweb.org/anthology/D15-1306/), [2](https://www.aclweb.org/anthology/P19-1328/) the former performs data augmentation based on top N similar words from a word embedding and the latter shows that using a BERT model's similarity between the original and the word substitution sentences are useful for evaluating lexical substitution.* 

Thus we are now going to create the fully expanded training dataset for each of the domains/datasets. This dataset will be exactly the same as the original training datasets that are currently json files, however there will be extra fields denoted by the index of the target where it will contain a list of semantically equivalent targets for that target at that index. An example of one sentence that contains multiple targets and it's equivalent targets is shown below: 
``` json
{"text": "It appears that many lefties are unable to tell the difference between tax evasion and tax avoidance #bbcqt", "text_id": "68544488139964416", "targets": ["tax avoidance", "tax evasion"], "spans": [[87, 100], [71, 82]], "target_sentiments": ["neutral", "neutral"], "target 0": ["tax avoidance", "tax evasion"], "target 1": ["tax evasion", "libor", "tax avoidance"]}
```
To create these expanded training datasets run the following bash script which will produce expanded datasets for the laptop, restaurant and election domain at the following paths respectively `./data/augmented_train/laptop.json`, `./data/augmented_train/restaurant.json`, `./data/augmented_train/election.json`
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

Next we are going to c
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