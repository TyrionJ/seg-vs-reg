
# train
python train_TransMorph.py -d 0 -ds Dataset001_OASIS
python train_TransMorph.py -d 2 -ds Dataset002_InHouse
python train_TransMorph.py -d 1 -ds Dataset004_OASIS-Sim
python train_TransMorph.py -d 1 -ds Dataset005_InHouse-Sim

# Infer
python infer_TransMorph.py -ds Dataset001_OASIS -d 1
python infer_TransMorph.py -ds Dataset002_InHouse -d 2
python infer_TransMorph.py -ds Dataset004_OASIS-Sim -d 1
python infer_TransMorph.py -ds Dataset005_InHouse-Sim -d 2
