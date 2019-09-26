cd ../ 
 
python -u main.py \
--train \
--test \
--val \
--data_dir data/ \
--hidden_size 128 \
--gpuid 0 \
--batch_size 128 \
--model_type vlstm \
--learning_rate 0.001 \
--optimizer Adam \
--eval_batch_size 128 \
--param_domain 1.5 \
--scheduler \
--data_directory data2/ \
--dataset_directory sample_dataset/ \
--criterion ADE \
--sequence_length 10 \
--prediction_length 10 \
--domain_init constant \
--net_dir models/ \
--delta_rb 30 \
--maxVessels 15 \
--delta_cog 30


