cd ../

python -u main.py \
--train \
--test \
--val \
--hidden_size 128 \
--gpuid 0 \
--batch_size 128 \
--model_type spatial_vlstm \
--learning_rate 0.001 \
--optimizer Adam \
--eval_batch_size 128 \
--param_domain 3.0 \
--dataset_directory sample_dataset/ \
--criterion ADE \
--sequence_length 10 \
--scheduler \
--prediction_length 5 \
--net_dir models/ \
--delta_rb 30 \
--maxVessels 15 \
--delta_cog 30

