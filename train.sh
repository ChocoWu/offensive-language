#!/bin/bash

batch_sizes=(16 32 64 80 100 128)
lrs=(0.05 0.01 0.001 0.003 0.005 0.0001)
dropout_probs=(0.3 0.4 0.5 0.6 0.7)
max_sents=(5 6 7)
max_words=(30 35 40 45)
atten_sizes=(100 200 300)

echo "staring train the model"

for batsh_size in ${batsh_size[@]}
do
    for lr in ${lrs[@]}
    do
        for dropout_prob in ${dropout_probs[@]}
        do
            for max_sent in ${max_sents[@]}
            do
                for max_word in ${max_words[@]}
                do
                    for atten_size in $ ${atten_sizes[@]}
                    do
                        echo `python -u train.py --batch_size=$batch_size --lr=$lr --dropout_prob=$dropout_prob --max_sent=$max_sent --max_word=$max_word --w_atten_size=$atten_size --s_atten_size=$atten_size`
                        echo "end training the model"
                    done
                done
            done
        done
    done
done
