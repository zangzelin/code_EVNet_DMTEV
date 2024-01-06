CUDA_VISIBLE_DEVICES=0 python ./patemb_main.py --project_name best_result  --K=5 --Uniform_t=2 --data_name=HCL60K3037D --detaalpha=1.005 --l2alpha=10 --nu=0.05 --num_fea_aim=300 --num_pat=8 --showmainfig=1 --save_checkpoint 1 &
CUDA_VISIBLE_DEVICES=1 python ./patemb_main.py --project_name best_result  --K=30 --Uniform_t=2 --data_name=Gast10k1457 --detaalpha=1.005 --l2alpha=10 --nu=0.01 --num_fea_aim=200 --num_pat=8 --showmainfig=1 --save_checkpoint 1 &
CUDA_VISIBLE_DEVICES=2 python ./patemb_main.py --project_name best_result  --K=5 --Uniform_t=2 --data_name=Mnist --detaalpha=1.005 --l2alpha=10 --nu=0.02 --num_fea_aim=200 --num_pat=8 --showmainfig=1 --save_checkpoint 1 &
CUDA_VISIBLE_DEVICES=3 python ./patemb_main.py --project_name best_result  --K=15 --Uniform_t=2 --data_name=Colon --detaalpha=1.005 --l2alpha=10 --nu=0.02 --num_fea_aim=200 --num_pat=8 --showmainfig=1 --save_checkpoint 1 &
CUDA_VISIBLE_DEVICES=7 python ./patemb_main.py --project_name best_result  --K=15 --Uniform_t=2 --data_name=MCAD9119 --detaalpha=1.005 --l2alpha=10 --nu=0.01 --num_fea_aim=300 --num_pat=8 --showmainfig=1 --save_checkpoint 1 &
CUDA_VISIBLE_DEVICES=5 python ./patemb_main.py --project_name best_result  --K=5 --Uniform_t=2 --data_name=PBMCD2638 --detaalpha=1.005 --l2alpha=10 --nu=0.01 --num_fea_aim=200 --num_pat=8 --showmainfig=1 --save_checkpoint 1 &
CUDA_VISIBLE_DEVICES=6 python ./patemb_main.py --project_name best_result  --K=15 --Uniform_t=2 --data_name=KMnist --detaalpha=1.001 --l2alpha=10 --nu=0.02 --num_fea_aim=200 --num_pat=8 --showmainfig=1 --save_checkpoint 1 &
CUDA_VISIBLE_DEVICES=6 python ./patemb_main.py --project_name best_result  --K=5 --Uniform_t=2 --data_name=Mnist --detaalpha=1.005 --l2alpha=10 --nu=0.02 --num_fea_aim=100 --num_pat=8 --showmainfig=1



