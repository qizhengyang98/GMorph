cd ../test_metamorph/scene
python ./test_scene_b4.py \
--policy_select=LCBased \
--log_name=b4_LC_t002 \
--load_weight \
--acc_drop_thres=0.02 \
--alpha=0.99 \
--fine_tune_epochs=40 \
--early_stop_check_epochs=5 \
--max_iteration=200 \
--batch_size=32 \
--num_workers=4 \
--finetune_early_stop