MODEL_FLAGS="--wm_length 23 --attention_resolutions 32,16,8 --class_cond False --image_size 256 --num_channels 256 --learn_sigma True --num_head_channels 64 --num_res_blocks 2 --resblock_updown True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 4"
NUM_GPUS=1
mpiexec -n $NUM_GPUS python scripts/image_train.py --data_dir ../../../data/imagenet/val --resume_checkpoint models/256x256_diffusion_uncond.pt $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS


# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_s;ize 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
# python classifier_sample.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS