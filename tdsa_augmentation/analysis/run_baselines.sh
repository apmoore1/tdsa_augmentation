pythonpath=$1
save_dir=$2
number_runs=$3
config_dir=$4

## declare an array variable
declare -a model_arr=("ian" "tdlstm" "interae")
declare -a domain_arr=("laptop" "election" "restaurant")
declare -a word_rep_arr=("--word_embedding" 
                         "--word_embedding_ds" 
                         "--elmo --word_embedding" 
                         "--elmo_ds --word_embedding" 
                         "--elmo_ds --word_embedding_ds")

for domain in "${domain_arr[@]}"
do
  domain_data_dir="./data/original_"$domain"_sentiment"
  for model in "${model_arr[@]}"
  do
    for word_rep in "${word_rep_arr[@]}"
    do
      config_fp="$config_dir/$model.jsonnet"
      echo "Running model $model on domain $domain with word rep $word_rep $number_runs times"
      $pythonpath ./tdsa_augmentation/analysis/flexi_run_models.py $domain_data_dir $config_fp $number_runs $domain $model $save_dir $word_rep
    done
  done
done

