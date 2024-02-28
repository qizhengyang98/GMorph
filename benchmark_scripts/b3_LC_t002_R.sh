cd ../test_metamorph/face
python ./test_face_b3.py \
--policy_select=LCBased \
--log_name=b3_LC_t002_R \
--load_weight \
--acc_drop_thres=0.02 \
--alpha=0.99 \
--fine_tune_epochs=35 \
--early_stop_check_epochs=5 \
--max_iteration=200 \
--batch_size=32 \
--num_workers=4 \
--sub_graph_finetune \
--finetune_early_stop \
--enable_filtering_rules