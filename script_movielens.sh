python run_CRL.py --env ml-1m  # run ML-1M dataset (default)


# 最全的方式：
#for augment_type in "seq" "mat": do
#  for augment_rate in 0 1 3 5 8 10; do
#    for strategy in "rating" "diversity" "random"; do
#      python run_CRL.py --env ml-1m --augment_type $augment_type --augment_rate $augment_rate --augment_strategies $strategy --message "$augment_type_$augment_rate_$strategy" &
#    done
#  done
#done


# 靠经验：
# --augment_type seq
python run_CRL.py --env ml-1m  --cuda 0 --augment_rate 0 --message  "seq_0" &
python run_CRL.py --env ml-1m  --cuda 1 --augment_rate 0.1 --message "seq_0.1" &
python run_CRL.py --env ml-1m  --cuda 2 --augment_rate 0.5 --message "seq_0.5" &
python run_CRL.py --env ml-1m  --cuda 3 --augment_rate 8 --message  "seq_8" &
python run_CRL.py --env ml-1m  --cuda 4 --augment_rate 5 --message  "seq_5" &
python run_CRL.py --env ml-1m  --cuda 5 --augment_rate 3 --message  "seq_3" &
python run_CRL.py --env ml-1m  --cuda 6 --augment_rate 1 --message  "seq_1" &

# --augment_type mat
python run_CRL.py --env ml-1m  --cuda 7 --augment_rate 0.1 --augment_type "mat" --message "mat_0.1" &
python run_CRL.py --env ml-1m  --cuda 1 --augment_rate 3  --augment_type "mat" --message "mat_3" &

# --augment_strategies
python run_CRL.py --env ml-1m  --cuda 0 --augment_rate 3 --augment_type "seq"  --augment_strategies "rating"    --message "rating_3" &
python run_CRL.py --env ml-1m  --cuda 2 --augment_rate 1  --augment_type "seq"  --augment_strategies "rating"    --message "rating_1" &
python run_CRL.py --env ml-1m  --cuda 3 --augment_rate 3 --augment_type "seq"  --augment_strategies "diversity" --message "diversity_3" &
python run_CRL.py --env ml-1m  --cuda 4 --augment_rate 1  --augment_type "seq"  --augment_strategies "diversity" --message "diversity_1" &
python run_CRL.py --env ml-1m  --cuda 5 --augment_rate 3 --augment_type "seq"  --augment_strategies "random"    --message "mat_3" &
python run_CRL.py --env ml-1m  --cuda 6 --augment_rate 1  --augment_type "seq"  --augment_strategies "random"    --message "mat_1" &



# n_layer=3


python run_CRL.py --env ml-1m  --cuda 3 --augment_rate 0 --message  "layer3_seq_0" &
python run_CRL.py --env ml-1m  --cuda 1 --augment_rate 0.1 --message "layer3_seq_0.1" &
python run_CRL.py --env ml-1m  --cuda 2 --augment_rate 0.5 --message "layer3_seq_0.5" &
#python run_CRL.py --env ml-1m  --cuda 3 --augment_rate 8 --message  "layer3_seq_8" &
#python run_CRL.py --env ml-1m  --cuda 4 --augment_rate 5 --message  "layer3_seq_5" &
python run_CRL.py --env ml-1m  --cuda 5 --augment_rate 3 --message  "layer3_seq_3" &
python run_CRL.py --env ml-1m  --cuda 4 --augment_rate 1 --message  "layer3_seq_1" &

# --augment_type mat
python run_CRL.py --env ml-1m  --cuda 7 --augment_rate 0.1 --augment_type "mat" --message "layer3_mat_0.1" &
python run_CRL.py --env ml-1m  --cuda 1 --augment_rate 3  --augment_type "mat" --message "layer3_mat_3" &

# --augment_strategies
python run_CRL.py --env ml-1m  --cuda 0 --augment_rate 3 --augment_type "seq"  --augment_strategies "rating"    --message "layer3_rating_3" &
python run_CRL.py --env ml-1m  --cuda 2 --augment_rate 1  --augment_type "seq"  --augment_strategies "rating"    --message "layer3_rating_1" &
python run_CRL.py --env ml-1m  --cuda 3 --augment_rate 3 --augment_type "seq"  --augment_strategies "diversity" --message "layer3_diversity_3" &
python run_CRL.py --env ml-1m  --cuda 4 --augment_rate 1  --augment_type "seq"  --augment_strategies "diversity" --message "layer3_diversity_1" &
python run_CRL.py --env ml-1m  --cuda 5 --augment_rate 3 --augment_type "seq"  --augment_strategies "random"    --message "layer3_mat_3" &
python run_CRL.py --env ml-1m  --cuda 6 --augment_rate 1  --augment_type "seq"  --augment_strategies "random"    --message "layer3_mat_1" &


