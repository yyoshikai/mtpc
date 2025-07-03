
# original command (single node)
# python -m torch.distributed.launch --nproc_per_node=8 main_vicregl.py --fp16 --exp-dir /path/to/experiment/ --arch resnet50 --epochs 100 --batch-size 512 --optimizer lars --base-lr 0.3 --weight-decay 1e-06 --size-crops 224 --num-crops 2 --min_scale_crops 0.08 --max_scale_crops 1.0 --alpha 0.75

# python run_with_submitit.py --nodes 4 --ngpus 8 --fp16 --exp-dir /path/to/experiment/ --arch resnet50 --epochs 300 --batch-size 2048 --optimizer lars --base-lr 0.2 --weight-decay 1e-06 --size-crops 224 --num-crops 2 --min_scale_crops 0.08 --max_scale_crops 1.0 --alpha 0.75

# rm -r results/1
# torchrun --nproc_per_node=1 main_vicregl_tggate1.py --fp16 --exp-dir results/1 --arch resnet50 --epochs 100 --batch-size 32 --optimizer lars --base-lr 0.3 --weight-decay 1e-06 --size-crops 224 --num-crops 2 --min_scale_crops 0.08 --max_scale_crops 1.0 --alpha 0.75 --dataset /workspace/patho/preprocess/results/tggate_liver_late --eval-freq 1

rm -r results/test
torchrun --nproc_per_node=1 main_vicregl_tggate.py --fp16 --exp-dir results/test --arch resnet50 --epochs 100 --batch-size 32 --optimizer lars --base-lr 0.3 --weight-decay 1e-06 --size-crops 224 --num-crops 2 --min_scale_crops 0.08 --max_scale_crops 1.0 --alpha 0.75 --eval-freq 1 --mtpc-main 1.0 --init-weight /workspace/mtpc/pretrain/results/250630_main/in_bt/resnet50.pth --log-tensors-interval 1 --seed 0