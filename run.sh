


python /home/mila/w/wook/scratch/deep_rl_practice/house_price_mlp/train.py \
--batch_size 64 \
--epochs 500 \
--learning_rate 0.01 \
--hidden_layers 512,256,128,64,32 \
--activation_function relu \
--dropout_rate 0.1 \
--optimizer adam \
--loss_function mae \
--validation_split 0.2 \
--early_stopping True \
--model_save_path /home/mila/w/wook/scratch/deep_rl_practice/house_price_mlp/models/mlp_100_epochs \
--tensorboard_log_dir /home/mila/w/wook/scratch/deep_rl_practice/house_price_mlp/logs \
--data_path /home/mila/w/wook/scratch/deep_rl_practice/house_price_mlp/house_prices_data \
--log_transform_targets False \