#!/bin/bash
echo "Finding similar targets using the targets from the predicted and training data"
echo "Laptop"
python tdsa_augmentation/data_augmentation/embedder_expander.py ./resources/target_words/all_laptop.txt ./resources/word_embeddings/amazon_300_phrases_3 15 ./resources/data_augmentation/target_words/laptop_predicted_train_expanded.json
echo "Restaurant"
python tdsa_augmentation/data_augmentation/embedder_expander.py ./resources/target_words/all_restaurant.txt ./resources/word_embeddings/yelp_300_phrases_3 15 ./resources/data_augmentation/target_words/restaurant_predicted_train_expanded.json
echo "Election"
python tdsa_augmentation/data_augmentation/embedder_expander.py ./resources/target_words/all_election.txt ./resources/word_embeddings/mp_300_phrases_3 15 ./resources/data_augmentation/target_words/election_predicted_train_expanded.json