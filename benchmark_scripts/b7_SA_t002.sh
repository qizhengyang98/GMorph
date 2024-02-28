cd ../test_metamorph/transformer_model
python ./test_glue_b7.py \
--policy_select=SimulatedAnnealing \
--log_name=b7_SA_t002 \
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