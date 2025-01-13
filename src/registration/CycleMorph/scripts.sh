
# train
python train.py -d 0 -ds Dataset001_OASIS
python train.py -d 3 -ds Dataset002_InHouse
python train.py -d 1 -ds Dataset004_OASIS-Sim
python train.py -d 3 -ds Dataset005_InHouse-Sim

# Infer
python infer.py -ds Dataset001_OASIS -d 0
python infer.py -ds Dataset002_InHouse -d 1
python infer.py -ds Dataset004_OASIS-Sim -d 2
python infer.py -ds Dataset005_InHouse-Sim -d 3
