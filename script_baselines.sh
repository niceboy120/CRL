

python run_SL_main.py --env ml-1m         --epoch 5 --device 0 --augment_rate 0 --model_name esmm &
python run_SL_main.py --env KuaiRand-Pure --epoch 5 --device 1 --augment_rate 0 --model_name esmm &
python run_SL_main.py --env Zhihu-1M      --epoch 5 --device 2 --augment_rate 0 --model_name esmm &

python run_SL_main.py --env ml-1m         --epoch 5 --device 3 --augment_rate 0 --model_name mmoe &
python run_SL_main.py --env KuaiRand-Pure --epoch 5 --device 4 --augment_rate 0 --model_name mmoe &
python run_SL_main.py --env Zhihu-1M      --epoch 5 --device 5 --augment_rate 0 --model_name mmoe &

python run_SL_main.py --env ml-1m         --epoch 5 --device 6 --augment_rate 0 --model_name ple &
python run_SL_main.py --env KuaiRand-Pure --epoch 5 --device 7 --augment_rate 0 --model_name ple &
python run_SL_main.py --env Zhihu-1M      --epoch 5 --device 0 --augment_rate 0 --model_name ple &

python run_SL_main.py --env ml-1m         --epoch 5 --device 1 --augment_rate 0 --model_name sharedbottom &
python run_SL_main.py --env KuaiRand-Pure --epoch 5 --device 2 --augment_rate 0 --model_name sharedbottom &
python run_SL_main.py --env Zhihu-1M      --epoch 5 --device 3 --augment_rate 0 --model_name sharedbottom &

python run_SL_main.py --env ml-1m         --epoch 5 --device 4 --augment_rate 0 --model_name singletask &
python run_SL_main.py --env KuaiRand-Pure --epoch 5 --device 5 --augment_rate 0 --model_name singletask &
python run_SL_main.py --env Zhihu-1M      --epoch 5 --device 6 --augment_rate 0 --model_name singletask &


# RMTL
python run_RMTL.py --env ml-1m         --epoch 30 --device 0 --augment_rate 0 --actor_model_name esmm &
python run_RMTL.py --env KuaiRand-Pure --epoch 30 --device 1 --augment_rate 0 --actor_model_name esmm &
python run_RMTL.py --env Zhihu-1M      --epoch 30 --device 2 --augment_rate 0 --actor_model_name esmm &

python run_RMTL.py --env ml-1m         --epoch 30 --device 3 --augment_rate 0 --actor_model_name mmoe &
python run_RMTL.py --env KuaiRand-Pure --epoch 30 --device 4 --augment_rate 0 --actor_model_name mmoe &
python run_RMTL.py --env Zhihu-1M      --epoch 30 --device 5 --augment_rate 0 --actor_model_name mmoe &

python run_RMTL.py --env ml-1m         --epoch 30 --device 6 --augment_rate 0 --actor_model_name ple &
python run_RMTL.py --env KuaiRand-Pure --epoch 30 --device 7 --augment_rate 0 --actor_model_name ple &
python run_RMTL.py --env Zhihu-1M      --epoch 30 --device 0 --augment_rate 0 --actor_model_name ple &

python run_RMTL.py --env ml-1m         --epoch 30 --device 1 --augment_rate 0 --actor_model_name sharedbottom &
python run_RMTL.py --env KuaiRand-Pure --epoch 30 --device 2 --augment_rate 0 --actor_model_name sharedbottom &
python run_RMTL.py --env Zhihu-1M      --epoch 30 --device 3 --augment_rate 0 --actor_model_name sharedbottom &

python run_RMTL.py --env ml-1m         --epoch 30 --device 4 --augment_rate 0 --actor_model_name singletask &
python run_RMTL.py --env KuaiRand-Pure --epoch 30 --device 5 --augment_rate 0 --actor_model_name singletask &
python run_RMTL.py --env Zhihu-1M      --epoch 30 --device 6 --augment_rate 0 --actor_model_name singletask &


# DT4Rec




# CDT4Rec
python run_DT4Rec.py --env ml-1m         --epoch 30 --cuda 0 --augment_rate 0 &
python run_DT4Rec.py --env KuaiRand-Pure --epoch 30 --cuda 1 --augment_rate 0 &
python run_DT4Rec.py --env Zhihu-1M      --epoch 30 --cuda 2 --augment_rate 0 &

python run_CDT4Rec.py --env ml-1m         --epoch 30 --cuda 7 --augment_rate 0 &
python run_CDT4Rec.py --env KuaiRand-Pure --epoch 30 --cuda 6 --augment_rate 0 &
python run_CDT4Rec.py --env Zhihu-1M      --epoch 30 --cuda 5 --augment_rate 0 &