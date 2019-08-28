#!/bin/bash
# ./run.sh



# merge_layer="attention"
# # "swem-aver" "swem-max" "bow" "bilstm" "bilstm-max" "bilstm-mean"　"cnn" "hierarchical-convnet" "transformer"
# shared_context_network=("gru")
# for v in ${shared_context_network[@]}
# do
#   echo $v
#   python3.6 ned_trainer.py $v $merge_layer
# done


# merge_layer="attention"
# shared_context_network=("transformer")
# # shared_context_network=("cnn" "bilstm" "bilstm-max" "bilstm-mean" "hierarchical-convnet")
# for v in ${shared_context_network[@]}
# do
#   echo $v
#   python3.6 ned_trainer.py $v $merge_layer
# done


# merge_layer="bi-attention"
# shared_context_network=("transformer")
# for v in ${shared_context_network[@]}
# do
#   echo $v
#   python3.6 ned_trainer.py $v $merge_layer
# done



# # "transformerT" "transformerQ"
# shared_context_network=("transformerQ")
# merge_layer="?"
# for v in ${shared_context_network[@]}
# do
#   echo $v
#   python3 ned_trainer.py $v $merge_layer
# done

# # python3 ned_trainer.py "bilstm-max" "attention"



#======================================================


# shared_context_network=("cnn")
# merge_layer='attention'
# mode = 'gating'
# for v in ${shared_context_network[@]}
# do
#   echo $v
#   python3.6 ned_trainer_bmc.py $v $merge_layer $mode
# done

python3.6 ned_trainer_bmc.2.py
python3.6 ned_trainer_bmc.py

