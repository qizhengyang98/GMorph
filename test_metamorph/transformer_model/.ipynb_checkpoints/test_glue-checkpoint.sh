python test_glue.py \
--policy_select=SimulatedAnnealing \
--log_name=glue_t001_1 \
--load_weight \
--acc_drop_thres=0.01 \
--alpha=0.99 \
--fine_tune_epochs=10 \
--early_stop_check_epochs=2 \
--max_iteration=200 \
--finetune_early_stop \
#--enable_filtering_rules