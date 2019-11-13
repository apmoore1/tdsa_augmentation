#!/bin/bash
echo "Extracting targets from the Laptop domain"
python ./tdsa_augmentation/data_augmentation/extract_predicted_targets.py ./resources/predicted_targets/laptop.json ./data/original_laptop_sentiment/train.json 0.9 ./resources/target_words/laptop_predicted.txt
echo "Extracting targets from the Restaurant domain"
python ./tdsa_augmentation/data_augmentation/extract_predicted_targets.py ./resources/predicted_targets/restaurant.json ./data/original_restaurant_sentiment/train.json 0.9 ./resources/target_words/restaurant_predicted.txt
echo "Extracting targets from the Election domain"
python ./tdsa_augmentation/data_augmentation/extract_predicted_targets.py ./resources/predicted_targets/election.json ./data/original_election_sentiment/train.json 0.9 ./resources/target_words/election_predicted.txt
echo "Done"