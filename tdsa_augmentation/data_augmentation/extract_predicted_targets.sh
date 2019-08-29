#!/bin/bash
echo "Extracting targets from the Laptop domain"
python ./tdsa_augmentation/data_augmentation/extract_predicted_targets.py ./resources/predicted_targets/laptop.json ./data/SemEval\'14-ABSA-TrainData_v2\ \&\ AnnotationGuidelines/Laptop_Train_v2.xml ./data/ABSA_Gold_TestData/Laptops_Test_Gold.xml 0.9 ./resources/target_words/laptop_predicted.txt
echo "Extracting targets from the Restaurant domain"
python ./tdsa_augmentation/data_augmentation/extract_predicted_targets.py ./resources/predicted_targets/restaurant.json ./data/SemEval\'14-ABSA-TrainData_v2\ \&\ AnnotationGuidelines/Restaurants_Train_v2.xml ./data/ABSA_Gold_TestData/Restaurants_Test_Gold.xml 0.9 ./resources/target_words/restaurant_predicted.txt
echo "Extracting targets from the Election domain"
python ./tdsa_augmentation/data_augmentation/extract_predicted_targets.py ./resources/predicted_targets/election.json nothing nothing 0.9 ./resources/target_words/election_predicted.txt --election
echo "Done"