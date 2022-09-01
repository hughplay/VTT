# Base CST shared
python train.py experiment=baseline_cst_shared

# Base TTNet using GLocal context features
python train.py experiment=baseline_ttnet_glocal

# *Base TTNet using LSTM text decoder
python train.py experiment=baseline_ttnet_lstm \
    criterion.loss.logit_shift=0 criterion.loss.label_shift=-1

# Tie Embedding
python train.py experiment=baseline_ttnet \
    model.tie_embedding=true name=baseline_ttnet_tie \
    callbacks.checkpoint.monitor="val/BLEU_4"

# *Fusion context into label ids by adding context to word embedding
python train.py experiment=baseline_ttnet_context \
    model.decoder_context_fusion=add name=baseline_ttnet_context_add

# *Fusion context into label ids by concatenation and linear projection
python train.py experiment=baseline_ttnet_context \
    model.decoder_context_fusion=concat name=baseline_ttnet_context_concat
