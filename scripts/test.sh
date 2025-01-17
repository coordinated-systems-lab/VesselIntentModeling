python -u main.py --test --model ${1:-spatial_temporal_model} --gpuid 0 \
--hidden_size 6 \
--feature_size 2 \
--zone 11 \
--criterion_type dist_error \
--sequence_length 5 \
--prediction_length 5 \
--delta_bearing 60 \
--delta_heading 60
