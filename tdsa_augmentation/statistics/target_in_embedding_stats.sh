#!/bin/bash
if [ $2 == "DS" ]; then
  if [ $3 == "not-predicted" ]; then
    echo "Statistics on the number of lower cased target words in the training dataset and how many of them are in the DS embedding"
    echo "Laptop dataset"
    python ./tdsa_augmentation/statistics/target_in_embedding_stats.py ./resources/target_words/laptop_train.txt ./resources/word_embeddings/amazon_300_phrases_3 --fraction $1
    echo "Restaurant dataset"
    python ./tdsa_augmentation/statistics/target_in_embedding_stats.py ./resources/target_words/restaurant_train.txt ./resources/word_embeddings/yelp_300_phrases_3 --fraction $1
    echo "Eelection dataset"
    python ./tdsa_augmentation/statistics/target_in_embedding_stats.py ./resources/target_words/election_train.txt ./resources/word_embeddings/mp_300_phrases_3 --fraction $1
  elif [ $3 == "predicted" ]; then
    echo "Statistics on the number of lower cased target words in the training dataset and from those targets predicted using an external corpus and how many of them are in the DS embedding"
    echo "Laptop dataset"
    python ./tdsa_augmentation/statistics/target_in_embedding_stats.py ./resources/target_words/laptop_predicted.txt ./resources/word_embeddings/amazon_300_phrases_3 --fraction $1
    echo "Restaurant dataset"
    python ./tdsa_augmentation/statistics/target_in_embedding_stats.py ./resources/target_words/restaurant_predicted.txt ./resources/word_embeddings/yelp_300_phrases_3 --fraction $1
    echo "Eelection dataset"
    python ./tdsa_augmentation/statistics/target_in_embedding_stats.py ./resources/target_words/election_predicted.txt ./resources/word_embeddings/mp_300_phrases_3 --fraction $1
  fi
elif [ $2 == "GLOVE" ]; then
  if [ $3 == "not-predicted" ]; then
    echo "Statistics on the number of lower cased target words in the training dataset and how many of them are in the GLOVE embedding"
    echo "Laptop dataset"
    python ./tdsa_augmentation/statistics/target_in_embedding_stats.py ./resources/target_words/laptop_train.txt ./resources/word_embeddings/shrunk_target_vectors/laptop_from_glove_840 --fraction $1
    echo "Restaurant dataset"
    python ./tdsa_augmentation/statistics/target_in_embedding_stats.py ./resources/target_words/restaurant_train.txt ./resources/word_embeddings/shrunk_target_vectors/rest_from_glove_840 --fraction $1
    echo "Eelection dataset for Glove 300"
    python ./tdsa_augmentation/statistics/target_in_embedding_stats.py ./resources/target_words/election_train.txt ./resources/word_embeddings/shrunk_target_vectors/election_from_glove_840 --fraction $1
    echo "Eelection dataset for Glove Twiter"
    python ./tdsa_augmentation/statistics/target_in_embedding_stats.py ./resources/target_words/election_train.txt ./resources/word_embeddings/shrunk_target_vectors/election_from_glove_twitter --fraction $1
  else
    echo "For the GLOVE embedding the third argument has to be 'not-predicted'"
  fi
else
  echo "Do not recognise the second argument. Has to be either 'DS' or 'GLOVE'"
fi
echo "Done"
