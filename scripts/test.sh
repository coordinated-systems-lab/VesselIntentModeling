cd ../

python -u main.py \
--test \
--sequence_length 10 \
--prediction_length 5 \
--delta_rb 30 \
--delta_cog 30 \
--maxVessels 15 \
--param_domain 3.0 \
--eval_batch_size 1 \
--model_type spatial_vlstm \
--hidden_size 128 \
--gpuid 0 \
--dataset_directory sample_dataset/ \
--net_dir models/ \
--criterion ADE
