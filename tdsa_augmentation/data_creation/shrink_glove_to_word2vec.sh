#!/bin/bash
echo "Shrinking large glove to restaurant training targets only and then converting to a Word2Vec like model"
python tdsa_augmentation/data_creation/shrink_glove_to_targets.py ./data/original_restaurant_sentiment/train.json ./resources/word_embeddings/glove.840B.300d.txt ./resources/word_embeddings/shrunk_target_vectors/rest_from_glove_840
echo "Shrinking large glove to laptop training targets only and then converting to a Word2Vec like model"
python tdsa_augmentation/data_creation/shrink_glove_to_targets.py ./data/original_laptop_sentiment/train.json ./resources/word_embeddings/glove.840B.300d.txt ./resources/word_embeddings/shrunk_target_vectors/laptop_from_glove_840
echo "Shrinking large glove to Twitter Election targets only and then converting to a Word2Vec like model"
python tdsa_augmentation/data_creation/shrink_glove_to_targets.py ./data/original_election_sentiment/train.json ./resources/word_embeddings/glove.840B.300d.txt ./resources/word_embeddings/shrunk_target_vectors/election_from_glove_840
echo "Shrinking Twitter glove to Twitter Election targets only and then converting to a Word2Vec like model"
python tdsa_augmentation/data_creation/shrink_glove_to_targets.py ./data/original_election_sentiment/train.json ./resources/word_embeddings/glove.twitter.27B.200d.txt ./resources/word_embeddings/shrunk_target_vectors/election_from_glove_twitter