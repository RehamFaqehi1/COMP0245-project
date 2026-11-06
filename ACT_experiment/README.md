# ACT experiment

[Project overview](../README.md) · [D1 MLP experiments and results](../final/robotnik_arm_task/README.md)

This folder stores an **Action Chunking Transformer (ACT)** experiment export.
Its configuration and training-step record document a separate architecture
exploration; they do not establish offline accuracy or closed-loop success.
ACT is **not included** in the 975-episode Original Expert/12-MLP benchmark.

## Saved policy configuration

Source: [pretrained_model/config.json](last/last/pretrained_model/config.json).

| Setting | Saved value |
|---|---|
| Observations | One state observation, `observation.environment_state`, width 33 |
| Actions | Width 11; action chunk size 100 and `n_action_steps=100` |
| Transformer | Model dimension 512, 8 heads, feed-forward width 3,200 |
| Layers | 4 encoder layers, 1 decoder layer |
| Latent model | VAE enabled; latent width 32, 4 VAE-encoder layers, KL weight 10 |
| Regularization | Dropout 0.1 |
| Device | CUDA in the saved configuration |

Although generic vision-backbone settings appear in the configuration, the
listed observation input is state-only; no image feature is declared. Do not
interpret this export as evidence of a trained visual perception pipeline.

## Training record and provenance limits

[train_config.json](last/last/pretrained_model/train_config.json) specifies AdamW,
learning rate 0.00001, weight decay 0.0001, batch size 8, seed 1,000 and 50,000
training steps. The [training-step record](last/last/training_state/training_step.json)
records step **50,000**. The dataset identifier is `local/comp0245_act_train_100hz`
and its original root is `/workspace/act_dataset/train`; the name alone does not
verify the export's sampling/conversion procedure or D1/D2 split membership.
Those historical paths are preserved in the saved configuration.

The configuration has `env=null`, `env_eval_freq=0`, `eval_steps=0` and
`dataset.eval_split=0.0`. A generic `eval.n_episodes=50` setting is **not** a saved
50-episode evaluation result. No validation metric report, learning curve or
matched closed-loop success report is supplied in this folder; no such scores
are claimed. The configuration is evidence of settings, not an executable
end-to-end dataset conversion or simulator integration workflow.

## Files and availability

The existing nested `last/last/` layout is retained:

- [pretrained_model/](last/last/pretrained_model/): policy configuration, training
  configuration, pre/postprocessor definitions, normalization tensors and locally
  saved `model.safetensors` weights.
- [training_state/](last/last/training_state/): optimizer parameter groups, optimizer
  tensors, RNG state and the training-step record.

Large model and optimizer tensor files have targeted ignore rules and may exist
only in a local checkout. Do not infer that all weights or the original training
dataset are downloadable from the presence of configuration files. The smaller
normalization/configuration files are separate artifacts and must remain paired
with the policy if it is used later. No simulator, training or evaluation command
is invented here because this export does not include a verified integration CLI.
