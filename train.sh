python -u main.py \
--train \
--gpuid 0 \
--hidden_size 6 \
--feature_size 2 \
--batch_size 64 \
--sequence_length 5 \
--prediction_length 5 \
--learning_rate 0.001 \
--param_domain 2 \
--criterion_type dist_error \
--threshold 0.03 \
--optimizer Adam \
--delta_bearing 60 \
--delta_heading 60 \
--model ${1:-spatial_temporal_model} 