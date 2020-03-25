python multigpu_train.py --batch_size_per_gpu=16 --checkpoint_path=./tmp/east_icdar2013_pvanet_original/  --training_data_path=/home/minjun/Jupyter/data/ocr/2013/Text\_Localization/train --decoder=original --backbone=Pvanet --gpu_list=0 --restore=True --dataset=icdar13

python multigpu_train.py --batch_size_per_gpu=16 --checkpoint_path=./tmp/east_icdar2013_mobilenet_original/  --training_data_path=/home/minjun/Jupyter/data/ocr/2013/Text\_Localization/train --decoder=original --backbone=Mobilenet --gpu_list=0 --restore=True --dataset=icdar13

python multigpu_train.py --batch_size_per_gpu=16 --checkpoint_path=./tmp/east_icdar2013_mobilenet_expand8/  --training_data_path=/home/minjun/Jupyter/data/ocr/2013/Text\_Localization/train --decoder=Expand8 --backbone=Mobilenet --gpu_list=0 --restore=True --dataset=icdar13
