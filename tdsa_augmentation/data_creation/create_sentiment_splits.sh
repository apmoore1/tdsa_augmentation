#!/bin/bash
echo "Creating Laptop sentiment splits"
python ./tdsa_augmentation/data_creation/create_dataset_splits.py sentiment ./data/SemEval\'14-ABSA-TrainData_v2\ \&\ AnnotationGuidelines/Laptop_Train_v2.xml ./data/ABSA_Gold_TestData/Laptops_Test_Gold.xml semeval_2014 ./data/original_laptop_sentiment/train.json ./data/original_laptop_sentiment/val.json ./data/original_laptop_sentiment/test.json
echo "Creating Restaurant sentiment splits"
python ./tdsa_augmentation/data_creation/create_dataset_splits.py sentiment ./data/SemEval\'14-ABSA-TrainData_v2\ \&\ AnnotationGuidelines/Restaurants_Train_v2.xml ./data/ABSA_Gold_TestData/Restaurants_Test_Gold.xml semeval_2014 ./data/original_restaurant_sentiment/train.json ./data/original_restaurant_sentiment/val.json ./data/original_restaurant_sentiment/test.json
echo "Creating Election sentiment splits"
python ./tdsa_augmentation/data_creation/create_dataset_splits.py sentiment nothing nothing election_twitter ./data/original_election_sentiment/train.json ./data/original_election_sentiment/val.json ./data/original_election_sentiment/test.json
echo "Done"