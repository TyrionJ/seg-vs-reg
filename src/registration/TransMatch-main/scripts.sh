# train
python Train.py --data_set Dataset001_OASIS --gpu 0
python Train.py --data_set Dataset002_InHouse --gpu 3
python Train.py --data_set Dataset004_OASIS-Sim --gpu 1
python Train.py --data_set Dataset005_InHouse-Sim --gpu 2

# infer
python inference.py --gpu 0 --data_set Dataset001_OASIS --save_dir /remote-home/hejj/Data/runtime/SoR_folder/TransMatch/preds/Dataset001_OASIS
python inference.py --gpu 0 --data_set Dataset002_InHouse --save_dir /remote-home/hejj/Data/runtime/SoR_folder/TransMatch/preds/Dataset002_InHouse
python inference.py --gpu 3 --data_set Dataset004_OASIS-Sim --save_dir /remote-home/hejj/Data/runtime/SoR_folder/TransMatch/preds/Dataset004_OASIS-Sim
python inference.py --gpu 3 --data_set Dataset005_InHouse-Sim --save_dir /remote-home/hejj/Data/runtime/SoR_folder/TransMatch/preds/Dataset005_InHouse-Sim