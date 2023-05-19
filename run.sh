# ====================================
# For token merge
# ====================================

pth_diff=esoc

# vits
# vit_base_patch16_224.orig=None

# datasets
imagenet=/home/esoc/repo/ILSVRC/Data/CLS-LOC
imagenet_sample=/home/esoc/repo/ILSVRC/Data/CLS-LOC/val/n02190166/ILSVRC2012_val_00014356.JPEG

### models
## code in local
# vit
vit_small_patch16_224=/home/${pth_diff}/kwangsoo/paper_codes/pretrained/baseline/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz
vit_base_patch16_224_im21k=/home/esoc/kwangsoo/paper_codes/pretrained/baseline/B_16-i1k-300ep-lr_0.001-aug_light0-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz

# deits
deit_tiny_patch16_224=/home/${pth_diff}/kwangsoo/paper_codes/pretrained/baseline/deit_tiny_patch16_224-a1311bcf.pth
deit_small_patch16_224=/home/${pth_diff}/kwangsoo/paper_codes/pretrained/baseline/deit_small_patch16_224-cd65a155.pth
deit_tiny_distilled_patch16_224=/home/${pth_diff}/kwangsoo/paper_codes/pretrained/baseline/deit_tiny_distilled_patch16_224-b40b3cf7.pth
deit_small_distilled_patch16_224=/home/${pth_diff}/kwangsoo/paper_codes/pretrained/baseline/deit_small_distilled_patch16_224-649709d9.pth
deit_base_patch16_224=None

# swins
swin_tiny_patch4_window7_224=None
swin_small_patch4_window7_224=None
swin_base_patch4_window7_224=None
# lvvit
lvvit_tiny_224=None
# official tome pt models
deit_t_r13=/home/esoc/kwangsoo/paper_codes/pretrained/ToMe/deit_T_r13.pth
deit_s_r13=/home/esoc/kwangsoo/paper_codes/pretrained/ToMe/deit_S_r13.pth

# my ft
deit_s_r13_tome=/home/esoc/kwangsoo/paper_codes/for_comparison/evit/train_log/deit_s_tome_r13_30ep_ft/best_acc_ep20.pth.tar
deit_s_r13_mine=/home/esoc/kwangsoo/paper_codes/for_comparison/evit/train_log/deit_s_mine_r13_30ep_ft_again/best_acc_ep25.pth.tar

## code in timm
# regnety
vit_small_patch16_224=None
vit_base_patch16_224=None
vit_large_patch16_224=None
vit_large_patch16_384=None
regnety_160=None

crossvit_tiny_240=None
crossvit_small_240=None

# baseline model path
baseline_pth=/home/${pth_diff}/kwangsoo/paper_codes/pretrained/baseline

# parameters
pretrained_download_pth=${baseline_pth}
batch_size=128

log_dir=./logs/tb_logs
save_path=./save/test

# pretrained_local_pth=${deit_small_patch16_224}
model_name=deit_small_patch16_224
tome_r=13
keep_rate=0.7
drop_loc=(3,6,9)
trade_off=0.5

# ================================================== script ==================================================
# # ============================================================================================================
# # 1gpu tome profiling
# batch_size=1
# CUDA_VISIBLE_DEVICES=1 python profiling.py --model_name ${model_name} \
# --batch_size ${batch_size} --input_size 224 \
# --data_path ${imagenet} --pt_dl ${pretrained_download_pth} --tome_r ${tome_r} \
# --use_amp True --dist_eval True \
# | tee profile_${model_name}_tome_r${tome_r}.txt

# batch_size=64
# for tome_r in 0 8 16
# do
    # # 1gpu tome benchmark
    # CUDA_VISIBLE_DEVICES=1 python benchmark.py --model_name ${model_name} \
    # --batch_size ${batch_size} --input_size 384 \
    # --data_path ${imagenet} --pt_dl ${pretrained_download_pth} --tome_r ${tome_r} \
    # --eval --use_amp True --dist_eval True \
#     | tee ${model_name}_bm_r${tome_r}.txt
#     mv ${model_name}_bm_r${tome_r}.txt ./logs
# done

# for tome_r in 8 16
# do
#     # 1gpu tome benchmark
#     CUDA_VISIBLE_DEVICES=0 python benchmark.py --model_name ${model_name} \
#     --batch_size ${batch_size} --input_size 384 \
#     --data_path ${imagenet} --pt_dl ${pretrained_download_pth} --tome_r ${tome_r} \
#     --eval --use_amp True --dist_eval True --prune \
#     | tee ${model_name}_bm_prr${tome_r}.txt
#     mv ${model_name}_bm_prr${tome_r}.txt ./logs
# done
# mv ${model_name}_bm_r0_16.txt ./logs



# # ============================================================================================================

# # ============================================================================================================
# 1gpu custom test
# for threshold in 0 1 2 3 4 5 6 7 8 9 10 11
# for tome_r in 13
# for tome_r in 8 9 10 11 12 13 14 15 16
# do
#     CUDA_VISIBLE_DEVICES=0 python main.py --model_name ${model_name} \
#     --batch_size ${batch_size} --input_size 224 \
#     --data_path ${imagenet} --pt_dl ${pretrained_download_pth} --tome_r ${tome_r} \
#     --eval --use_amp False --dist_eval False \
#     | tee ./txt/ablation/${model_name}_attn_mean_score_r${tome_r}_test.txt

#     sleep 1m
# done
# | tee ${model_name}_${keep_rate}_custom_test.txt
#     # --data_path ${imagenet} --pt_local ${deit_s_r13} --tome_r ${tome_r} \
    # --data_path ${imagenet} --pt_dl ${pretrained_download_pth} --tome_r ${tome_r} \

# 1gpu custom benchmark
# for threshold in 0 1 2 3 4 5 6 7 8 9 10 11
# for tome_r in 1 2 3 4 5 6 7
# for tome_r in 8 9 10 11 12 13 14 15 16
for tome_r in 13
do
    batch_size=128

    CUDA_VISIBLE_DEVICES=1 python benchmark.py --model_name ${model_name} \
    --batch_size ${batch_size} --input_size 224 \
    --data_path ${imagenet} --pt_dl ${pretrained_download_pth} --tome_r ${tome_r} \
    --eval --use_amp False --dist_eval False
    # | tee ./txt/${model_name}_mine_attn_r${tome_r}_benchmark.txt
    # | tee ${model_name}_${keep_rate}_custom_benchmark.txt

    sleep 30s

done

# --mymodel
    # --data_path ${imagenet} --pt_local ${myft} --tome_r ${tome_r} \
# mv ${model_name}_${keep_rate}_custom_benchmark.txt ./logs

# # ============================================================================================================
# 1gpu custom test, pt model in local
# CUDA_VISIBLE_DEVICES=0 python main.py --model_name ${model_name} \
# --batch_size ${batch_size} --input_size 224 \
# --data_path ${imagenet} --pt_local ${deit_s_r13} --tome_r ${tome_r} \
# --eval --use_amp False --dist_eval False --mymodel
# | tee ${model_name}_tome_official.txt
# | tee ${model_name}_r${tome_r}_official.txt

# # 1gpu custom test, get data
# batch_size=50
# # for tome_r in 13
# for tome_r in 8 9 10 11 12 13 14 15 16
# do
#     CUDA_VISIBLE_DEVICES=0 python main.py --model_name ${model_name} \
#     --batch_size ${batch_size} --input_size 224 \
#     --data_path ${imagenet} --pt_dl ${pretrained_download_pth} --tome_r ${tome_r} \
#     --eval --use_amp False --dist_eval False --mymodel --with_data \
#     --save_path ./r${tome_r} \
#     | tee ${model_name}_tome_r${tome_r}_std_avg.txt

#     sleep 30s
# done

# # 2gpu custom test
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env \
# main.py --model_name ${model_name} \
# --batch_size ${batch_size} --input_size 224 \
# --data_path ${imagenet} --pt_dl ${pretrained_download_pth} --keep_rate ${keep_rate} --drop_loc ${drop_loc} \
# --eval --use_amp False --dist_eval False --mymodel
# | tee ${model_name}_test_${keep_rate}_custom.txt
# # mv ${model_name}_test_r${tome_r}_custom ./logs


# # 1gpu dense test
# CUDA_VISIBLE_DEVICES=1 python main.py --model_name ${model_name} \
# --batch_size ${batch_size} --input_size 224 \
# --data_path ${imagenet} --pt_dl ${pretrained_download_pth} \
# --eval --use_amp False --dist_eval False
# # | tee ${model_name}_test_r${tome_r}_original.txt


# # 2gpu original test
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env \
# main.py --model_name ${model_name} \
# --batch_size ${batch_size} --input_size 224 \
# --data_path ${imagenet} --pt_dl ${pretrained_download_pth} --tome_r ${tome_r} \
# --eval --use_amp False --dist_eval True --mymodel \
# | tee ${model_name}_test_r${tome_r}_original.txt
# # mv ${model_name}_test_r${tome_r}_original ./logs
# # ============================================================================================================
