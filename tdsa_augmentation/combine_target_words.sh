echo "Combining Target words for the"
echo "Laptop dataset"
cat ./resources/target_words/laptop_predicted.txt ./resources/target_words/laptop_train.txt > ./resources/target_words/all_laptop.txt
wc -l ./resources/target_words/all_laptop.txt
echo "Restaurant dataset"
cat ./resources/target_words/restaurant_predicted.txt ./resources/target_words/restaurant_train.txt > ./resources/target_words/all_restaurant.txt
wc -l ./resources/target_words/all_restaurant.txt
echo "Election dataset"
cat ./resources/target_words/election_predicted.txt ./resources/target_words/election_train.txt > ./resources/target_words/all_election.txt
wc -l ./resources/target_words/all_election.txt