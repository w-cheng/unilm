#!/bin/bash
mkdir -p sroie_sample_data sroie_sample_base_output
rm sroie_sample_data/*
python ./scripts/funsd_preprocess.py --data_dir ./SROIE_sample/SROIE_dataset/train/annotations/ --data_split train --output_dir sroie_sample_data --max_len 510
python ./scripts/funsd_preprocess.py --data_dir ./SROIE_sample/SROIE_dataset/test/annotations/ --data_split test --output_dir sroie_sample_data --max_len 510

cat sroie_sample_data/train.txt | cut -d$'\t' -f 2 | grep -v "^$"| sort | uniq > sroie_sample_data/labels.txt

python run_seq_labeling.py --data_dir sroie_sample_data --model_type layoutlm --model_name_or_path layoutlm-base-uncased --do_lower_case --max_seq_length 512 --do_train --num_train_epochs 100 --logging_steps 10 --save_steps -1 --output_dir sroie_sample_base_output --labels sroie_sample_data/labels.txt --overwrite_output_dir --overwrite_cache --per_gpu_train_batch_size 1 --fp16 --do_predict
