cd ../metamorph/test
python ./compiler_test.py \
--policy_select=LCBased \
--log_name=b1_LC_t001 \
--load_weight \
--acc_drop_thres=0.01 \
--alpha=0.99 \
--fine_tune_epochs=40 \
--early_stop_check_epochs=2 \
--max_iteration=200 \
--batch_size=256 \
--num_workers=4 \
--finetune_early_stop