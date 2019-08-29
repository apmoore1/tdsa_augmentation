#!/bin/bash
echo "Creates the fully augmented training dataset using only the targets from the training dataset"
echo "Laptop"
python tdsa_augmentation/data_augmentation/lm_expander_data_creator.py ./resources/data_augmentation/target_words/laptop_predicted_train_expanded.json ./resources/language_models/amazon_model.tar.gz ./data/original_laptop_sentiment/train.json ./data/augmented_train/laptop_predicted.json --cuda
echo "Restaurant"
python tdsa_augmentation/data_augmentation/lm_expander_data_creator.py ./resources/data_augmentation/target_words/restaurant_predicted_train_expanded.json ./resources/language_models/yelp_model.tar.gz ./data/original_restaurant_sentiment/train.json ./data/augmented_train/restaurant_predicted.json --cuda
echo "Election"
python tdsa_augmentation/data_augmentation/lm_expander_data_creator.py ./resources/data_augmentation/target_words/election_predicted_train_expanded.json ./resources/language_models/election_model.tar.gz ./data/original_election_sentiment/train.json ./data/augmented_train/election_predicted.json --cuda
echo "Done"