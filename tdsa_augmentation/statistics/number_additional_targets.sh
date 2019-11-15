#!/bin/bash
if [ $1 == "not-predicted" ]; then
  echo "Displays the statistics of the number of additional targets in the expanded augmented training datasets"
  echo "Laptop"
  python tdsa_augmentation/statistics/number_additional_targets.py ./data/augmented_train/laptop_1.json ./resources/data_augmentation/target_words/laptop_train_expanded_1.json
  echo "Restaurant"
  python tdsa_augmentation/statistics/number_additional_targets.py ./data/augmented_train/restaurant_1.json ./resources/data_augmentation/target_words/restaurant_train_expanded_1.json
  echo "Election"
  python tdsa_augmentation/statistics/number_additional_targets.py ./data/augmented_train/election_1.json ./resources/data_augmentation/target_words/election_train_expanded_1.json
elif [ $1 == "predicted" ]; then
  echo "Displays the statistics of the number of additional targets in the expanded augmented training datasets using the additional predicted targets"
  echo "Laptop"
  python tdsa_augmentation/statistics/number_additional_targets.py ./data/augmented_train/laptop_predicted_1.json ./resources/data_augmentation/target_words/laptop_predicted_train_expanded_1.json
  echo "Restaurant"
  python tdsa_augmentation/statistics/number_additional_targets.py ./data/augmented_train/restaurant_predicted_1.json ./resources/data_augmentation/target_words/restaurant_predicted_train_expanded_1.json
  echo "Election"
  python tdsa_augmentation/statistics/number_additional_targets.py ./data/augmented_train/election_predicted_1.json ./resources/data_augmentation/target_words/election_predicted_train_expanded_1.json
elif [ $1 == "combined" ]; then
  echo "Displays the statistics of the number of additional targets in the expanded augmented training datasets using the combined predicted and training targets"
  echo "Laptop"
  python tdsa_augmentation/statistics/number_additional_targets.py ./data/augmented_train/laptop_combined.json ./resources/data_augmentation/target_words/laptop_predicted_train_expanded_1.json
  echo "Restaurant"
  python tdsa_augmentation/statistics/number_additional_targets.py ./data/augmented_train/restaurant_combined.json ./resources/data_augmentation/target_words/restaurant_predicted_train_expanded_1.json
  echo "Election"
  python tdsa_augmentation/statistics/number_additional_targets.py ./data/augmented_train/election_combined.json ./resources/data_augmentation/target_words/election_predicted_train_expanded_1.json
else
  echo "Do not recognise the third argument. Has to be either 'not-predicted', 'predicted', or 'combined'"
fi
echo "Done"