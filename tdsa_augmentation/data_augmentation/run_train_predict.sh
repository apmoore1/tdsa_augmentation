#!/bin/bash
echo "Training and predicting for the Laptop domain"
python ./tdsa_augmentation/data_augmentation/target_extraction_train_predict.py semeval_2014 ./tdsa_augmentation/data_augmentation/target_extraction_configs/laptop.jsonnet ./resources/data_augmentation/target_extraction_models/laptop ./resources/language_model_datasets/amazon_filtered_split_train.txt ./resources/predicted_targets/laptop.json --train_fp ./data/SemEval\'14-ABSA-TrainData_v2\ \&\ AnnotationGuidelines/Laptop_Train_v2.xml --test_fp ./data/ABSA_Gold_TestData/Laptops_Test_Gold.xml --number_to_predict_on 1000000 --batch_size 256 --cuda
echo "Training and predicting for the Restaurant domain"
python ./tdsa_augmentation/data_augmentation/target_extraction_train_predict.py semeval_2014 ./tdsa_augmentation/data_augmentation/target_extraction_configs/restaurant.jsonnet ./resources/data_augmentation/target_extraction_models/restaurant ./resources/language_model_datasets/yelp_filtered_split_train.txt ./resources/predicted_targets/restaurant.json --train_fp ./data/SemEval\'14-ABSA-TrainData_v2\ \&\ AnnotationGuidelines/Restaurants_Train_v2.xml --test_fp ./data/ABSA_Gold_TestData/Restaurants_Test_Gold.xml --number_to_predict_on 1000000 --batch_size 256 --cuda
echo "Training and predicting for the Election domain"
python ./tdsa_augmentation/data_augmentation/target_extraction_train_predict.py election_twitter ./tdsa_augmentation/data_augmentation/target_extraction_configs/election.jsonnet ./resources/data_augmentation/target_extraction_models/election ./resources/language_model_datasets/election_filtered_split_train.txt ./resources/predicted_targets/election.json --number_to_predict_on 1000000 --batch_size 256 --cuda
echo "Done"