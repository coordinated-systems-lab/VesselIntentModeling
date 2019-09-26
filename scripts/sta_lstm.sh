cd ../

python -u main.py \
--train \
--val \
--test \
--hidden_size 16 \
--gpuid 0 \
--batch_size 128 \
--eval_batch_size 128 \
--model_type sta_lstm1 \
--scheduler \
--optimizer Adam \
--learning_rate 0.001 \
--param_domain 3.0 \
--maxVessels 15 \
--sequence_length 10 \
--data_directory data/ \
--split_data \
--save_data \
--dataset_directory dataset/ \
--prediction_length 5 \
--delta_rb 30 \
--delta_cog 45 \
