# source envs.sh to activate right environment
# and export some python variables for correct functioning.

# nomenclature is,for now, hardcoded... automatization is encouraged, not so difficult,
# I think I can automize it in less than a day, but for now I have plenty work to do.
# Nomenclature for naming the model and plots:
#   datasetname: the name of the dataset used to train the network
#   archname: the name of the architecture of the neural conv net
#   num_channs: number of channels used in the images: 3 for color, 1 for grayscale
#   num_epochs: number of epochs during treaining phase: for 1 epoch use 001 for 30 use 030.
#   operation: if there was no change in the hyperparams listed here, add 1 to the last
#            training or operation done. (e.g. ____aug_001.h5, ____aug_002.h5, ____aug_003.h5, etc)
#
# {datasetname}_{archname}_ch{num_channs:1}_it{num_epochs:3}_aug_{operation}.h5
for vars in 0.001,02 0.0001,03 0.0005,05 0.00007,06
do
    IFS=',' read etha number <<< "${vars}";
    python3 trainer/main.py \
        --architecture lenet \
        --dataset datasets/NCWV2 \
        --model-name NCWV2_minivgg_ch3_it150_aug_${number}.h5 \
        --epochs 150 \
        --etha ${etha} \
        --optimizer adam
done