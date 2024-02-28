cd ../test_metamorph/transformer_model
python ./test_vit_b6.py \
--policy_select=LCBased \
--log_name=b6_LC_t002_R \
--load_weight \
--acc_drop_thres=0.02 \
--alpha=0.99 \
--fine_tune_epochs=16 \
--early_stop_check_epochs=2 \
--max_iteration=200 \
--batch_size=32 \
--num_workers=4 \
--finetune_early_stop \
--enable_filtering_rules