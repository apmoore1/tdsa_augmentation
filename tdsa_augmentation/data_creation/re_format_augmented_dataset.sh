#!/bin/bash
echo "Reformatting the laptop datasets including the training only, predicted only, and combined augmented datasets"
python tdsa_augmentation/data_creation/re_format_augmented_dataset.py ./data/augmented_train/laptop_1.json ./data/augmented_train/reformated_laptop_1.json
python tdsa_augmentation/data_creation/re_format_augmented_dataset.py ./data/augmented_train/laptop_predicted_1.json ./data/augmented_train/reformated_laptop_predicted_1.json
python tdsa_augmentation/data_creation/re_format_augmented_dataset.py ./data/augmented_train/laptop_combined.json ./data/augmented_train/reformated_laptop_combined.json
echo "Reformatting the restaurant datasets including the training only, predicted only, and combined augmented datasets"
python tdsa_augmentation/data_creation/re_format_augmented_dataset.py ./data/augmented_train/restaurant_1.json ./data/augmented_train/reformated_restaurant_1.json
python tdsa_augmentation/data_creation/re_format_augmented_dataset.py ./data/augmented_train/restaurant_predicted_1.json ./data/augmented_train/reformated_restaurant_predicted_1.json
python tdsa_augmentation/data_creation/re_format_augmented_dataset.py ./data/augmented_train/restaurant_combined.json ./data/augmented_train/reformated_restaurant_combined.json
echo "Reformatting the election datasets including the training only, predicted only, and combined augmented datasets"
python tdsa_augmentation/data_creation/re_format_augmented_dataset.py ./data/augmented_train/election_1.json ./data/augmented_train/reformated_election_1.json
python tdsa_augmentation/data_creation/re_format_augmented_dataset.py ./data/augmented_train/election_predicted_1.json ./data/augmented_train/reformated_election_predicted_1.json
python tdsa_augmentation/data_creation/re_format_augmented_dataset.py ./data/augmented_train/election_combined.json ./data/augmented_train/reformated_election_combined.json