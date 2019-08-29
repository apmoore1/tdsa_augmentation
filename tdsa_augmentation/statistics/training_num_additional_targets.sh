#!/bin/bash
echo "Displays the statistics of the number of additional targets in the expanded augmented training datasets"
echo "Laptop"
python tdsa_augmentation/statistics/number_additional_targets.py ./data/augmented_train/laptop.json ./resources/data_augmentation/target_words/laptop_train_expanded.json
echo "Restaurant"
python tdsa_augmentation/statistics/number_additional_targets.py ./data/augmented_train/restaurant.json ./resources/data_augmentation/target_words/restaurant_train_expanded.json
echo "Election"
python tdsa_augmentation/statistics/number_additional_targets.py ./data/augmented_train/election.json ./resources/data_augmentation/target_words/election_train_expanded.json