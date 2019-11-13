#!/bin/bash
if [ $2 == "not-predicted" ]; then
  echo "Creates the fully augmented training dataset using only the targets from the training dataset"
  echo "Laptop"
  python tdsa_augmentation/data_augmentation/lm_expander_data_creator.py ./resources/data_augmentation/target_words/laptop_train_expanded_1.json ./resources/language_models/laptop_model.tar.gz ./data/original_laptop_sentiment/train.json ./data/augmented_train/laptop_1.json --cuda --batch_size $1
  echo "Restaurant"
  python tdsa_augmentation/data_augmentation/lm_expander_data_creator.py ./resources/data_augmentation/target_words/restaurant_train_expanded_1.json ./resources/language_models/restaurant_model.tar.gz ./data/original_restaurant_sentiment/train.json ./data/augmented_train/restaurant_1.json --cuda --batch_size $1
  echo "Election"
  python tdsa_augmentation/data_augmentation/lm_expander_data_creator.py ./resources/data_augmentation/target_words/election_train_expanded_1.json ./resources/language_models/election_model.tar.gz ./data/original_election_sentiment/train.json ./data/augmented_train/election_1.json --cuda --batch_size $1
elif [ $2 == "predicted" ]; then
  echo "Creates the fully augmented training dataset using the targets from the predicted dataset"
  echo "Laptop"
  python tdsa_augmentation/data_augmentation/lm_expander_data_creator.py ./resources/data_augmentation/target_words/laptop_predicted_train_expanded_1.json ./resources/language_models/laptop_model.tar.gz ./data/original_laptop_sentiment/train.json ./data/augmented_train/laptop_predicted_1.json --cuda --batch_size $1
  echo "Restaurant"
  python tdsa_augmentation/data_augmentation/lm_expander_data_creator.py ./resources/data_augmentation/target_words/restaurant_predicted_train_expanded_1.json ./resources/language_models/restaurant_model.tar.gz ./data/original_restaurant_sentiment/train.json ./data/augmented_train/restaurant_predicted_1.json --cuda --batch_size $1
  echo "Election"
  python tdsa_augmentation/data_augmentation/lm_expander_data_creator.py ./resources/data_augmentation/target_words/election_predicted_train_expanded_1.json ./resources/language_models/election_model.tar.gz ./data/original_election_sentiment/train.json ./data/augmented_train/election_predicted_1.json --cuda --batch_size $1
else
  echo "Do not recognise the second argument. Has to be either 'not-predicted' or 'predicted'"
fi
echo "Done"