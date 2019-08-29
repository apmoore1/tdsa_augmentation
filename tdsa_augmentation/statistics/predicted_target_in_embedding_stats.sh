#!/bin/bash
echo "Statistics on the number of lower cased target words that have been predicted and are not in the training dataset targets and how many of them are in the embedding"
echo "Laptop dataset"
python ./tdsa_augmentation/statistics/target_in_embedding_stats.py ./resources/target_words/laptop_predicted.txt ./resources/word_embeddings/amazon_300_phrases_3
echo "Restaurant dataset"
python ./tdsa_augmentation/statistics/target_in_embedding_stats.py ./resources/target_words/restaurant_predicted.txt ./resources/word_embeddings/yelp_300_phrases_3
echo "Eelection dataset"
python ./tdsa_augmentation/statistics/target_in_embedding_stats.py ./resources/target_words/election_predicted.txt ./resources/word_embeddings/mp_300_phrases_3