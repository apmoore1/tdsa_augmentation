#!/bin/bash
echo "Combining laptops the augmented dataset from the training and predicted targets"
python tdsa_augmentation/data_augmentation/combine_train_predicted_datasets.py ./data/augmented_train/laptop_1.json ./data/augmented_train/laptop_predicted_1.json ./data/augmented_train/laptop_combined.json
echo "Combining restaurants the augmented dataset from the training and predicted targets"
python tdsa_augmentation/data_augmentation/combine_train_predicted_datasets.py ./data/augmented_train/restaurant_1.json ./data/augmented_train/restaurant_predicted_1.json ./data/augmented_train/restaurant_combined.json
echo "Combining elections the augmented dataset from the training and predicted targets"
python tdsa_augmentation/data_augmentation/combine_train_predicted_datasets.py ./data/augmented_train/election_1.json ./data/augmented_train/election_predicted_1.json ./data/augmented_train/election_combined.json