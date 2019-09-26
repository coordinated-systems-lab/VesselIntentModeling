cd ../

python -u main.py \
--train \
--test \
--val \
--data_dir data/ \
--hidden_size 128 \
--gpuid 2 \
--batch_size 128 \
--model_type temporal_vlstm \
--learning_rate 0.001 \
--optimizer Adam \
--eval_batch_size 128 \
--param_domain 1 \
--criterion ADE \
--sequence_length 10 \
--scheduler \
--dataset_directory sample_dataset/ \
--prediction_length 5 \
--domain_init constant \
--net_dir models/ \
--delta_rb 30 \
--maxVessels 15 \
--delta_cog 30

