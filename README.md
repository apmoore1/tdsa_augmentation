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
./data/create_sentiment_splits.sh
```
The validation and test sets are roughly equal in size.

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
./data_augmentation/run_train_predict.sh
```

As the Target Extraction systems have been trained and have outputted there predictions on the `relevant large un-labeled text corpora` we can now extract out the 90% confident targets for each dataset and store them in `./resources/target_words` as `restaurant_predicted.txt`, `laptop_predicted.txt`, and `election_predicted.txt`:
```
./data_augmentation/extract_predicted_targets.sh.sh
```