#!/bin/bash
echo "Finding similar targets using the targets from the training data"
echo "Laptop"
python tdsa_augmentation/data_augmentation/embedder_expander.py ./resources/target_words/laptop_train.txt ./resources/word_embeddings/amazon_300_phrases_3 15 ./resources/data_augmentation/target_words/laptop_train_expanded_1.json 1.0
echo "Restaurant"
python tdsa_augmentation/data_augmentation/embedder_expander.py ./resources/target_words/restaurant_train.txt ./resources/word_embeddings/yelp_300_phrases_3 15 ./resources/data_augmentation/target_words/restaurant_train_expanded_1.json 1.0
echo "Election"
python tdsa_augmentation/data_augmentation/embedder_expander.py ./resources/target_words/election_train.txt ./resources/word_embeddings/mp_300_phrases_3 15 ./resources/data_augmentation/target_words/election_train_expanded_1.json 1.0