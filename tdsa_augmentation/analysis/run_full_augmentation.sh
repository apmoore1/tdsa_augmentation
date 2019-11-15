pythonpath=$1
save_dir=$2
number_runs=$3
config_dir=$4

## declare an array variable
declare -a augmentation_arr=("training" "predicted" "combined")
declare -a model_arr=("ian" "interae" "tdlstm")
declare -a domain_arr=("laptop" "election" "restaurant")
declare -a word_rep_arr=("--word_embedding")

for aug_set in "${augmentation_arr[@]}"
do
  temp_save_dir="$save_dir$aug_set"
  for domain in "${domain_arr[@]}"
  do
    domain_data_dir="./data/augmented_datasets/$domain/$aug_set"
    for model in "${model_arr[@]}"
    do
      for word_rep in "${word_rep_arr[@]}"
      do
        config_fp="$config_dir/$model.jsonnet"
        echo "Running model $model on domain $domain using augmentation $aug_set with word rep $word_rep $number_runs times"
        $pythonpath ./tdsa_augmentation/analysis/flexi_run_models.py $domain_data_dir $config_fp $number_runs $domain $model $temp_save_dir $word_rep
      done
    done
  done
done

