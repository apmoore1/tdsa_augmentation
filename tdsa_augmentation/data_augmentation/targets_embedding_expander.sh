#!/bin/bash

if [ $3 == "not-predicted" ]; then
  echo "Finding similar targets using the targets from the training data"
  echo "Laptop"
  python tdsa_augmentation/data_augmentation/embedder_expander.py ./resources/target_words/laptop_train.txt ./resources/word_embeddings/amazon_300_phrases_3 $1 ./resources/data_augmentation/target_words/laptop_train_expanded_1.json $2
  echo "Restaurant"
  python tdsa_augmentation/data_augmentation/embedder_expander.py ./resources/target_words/restaurant_train.txt ./resources/word_embeddings/yelp_300_phrases_3 $1 ./resources/data_augmentation/target_words/restaurant_train_expanded_1.json $2
  echo "Election"
  python tdsa_augmentation/data_augmentation/embedder_expander.py ./resources/target_words/election_train.txt ./resources/word_embeddings/mp_300_phrases_3 $1 ./resources/data_augmentation/target_words/election_train_expanded_1.json $2
elif [ $3 == "predicted" ]; then
  echo "Finding similar targets using the targets from the training and predicted data"
  echo "Laptop"
  python tdsa_augmentation/data_augmentation/embedder_expander.py ./resources/target_words/laptop_predicted.txt ./resources/word_embeddings/amazon_300_phrases_3 $1 ./resources/data_augmentation/target_words/laptop_predicted_train_expanded_1.json $2 --additional_train_targets ./resources/target_words/laptop_train.txt
  echo "Restaurant"
  python tdsa_augmentation/data_augmentation/embedder_expander.py ./resources/target_words/restaurant_predicted.txt ./resources/word_embeddings/yelp_300_phrases_3 $1 ./resources/data_augmentation/target_words/restaurant_predicted_train_expanded_1.json $2 --additional_train_targets ./resources/target_words/restaurant_train.txt
  echo "Election"
  python tdsa_augmentation/data_augmentation/embedder_expander.py ./resources/target_words/election_predicted.txt ./resources/word_embeddings/mp_300_phrases_3 $1 ./resources/data_augmentation/target_words/election_predicted_train_expanded_1.json $2 --additional_train_targets ./resources/target_words/election_train.txt
else
  echo "Do not recognise the third argument. Has to be either 'not-predicted' or 'predicted'"
fi
echo "Done"