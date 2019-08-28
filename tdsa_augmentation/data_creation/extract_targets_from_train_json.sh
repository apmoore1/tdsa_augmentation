#!/bin/bash
echo "Extracting targets from Laptop training data"
python ./tdsa_augmentation/data_creation/extract_targets_from_json.py ./data/original_laptop_sentiment/train.json ./resources/target_words/laptop_train.txt
echo "Extracting targets from Restaurant training data"
python ./tdsa_augmentation/data_creation/extract_targets_from_json.py ./data/original_restaurant_sentiment/train.json ./resources/target_words/restaurant_train.txt
echo "Extracting targets from Election training data"
python ./tdsa_augmentation/data_creation/extract_targets_from_json.py ./data/original_election_sentiment/train.json ./resources/target_words/election_train.txt