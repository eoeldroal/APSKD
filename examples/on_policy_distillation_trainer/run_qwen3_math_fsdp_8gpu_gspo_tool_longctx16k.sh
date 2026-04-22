#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

project_name="${PROJECT_NAME:-verl_qwen3_math_tool_gspo_fsdp}"

default_seed_model_path="$ROOT_DIR/checkpoints/merged_hf/qwen3_1p7b_skd_step50"
base_model_path="${BASE_MODEL_PATH:-$default_seed_model_path}"
seed_tag="${SEED_TAG:-$(basename "$base_model_path")}"

rl_training_steps="${RL_TRAINING_STEPS:-400}"
total_training_steps="$rl_training_steps"

experiment_name="${EXPERIMENT_NAME:-gspo-qwen3-1.7b-s8-nemotron-tool-ctx16k-${seed_tag}}"

nemotron_train_path="$ROOT_DIR/data/nemotron_cascade_rl_math_multiturn_w_tool/train.parquet"
train_files="['$nemotron_train_path']"

math500_test_path="$ROOT_DIR/data/math500/test.parquet"
aime2024_test_path="$ROOT_DIR/data/aime-2024.parquet"
test_files="['$aime2024_test_path', '$math500_test_path']"

reward_fn_path="$ROOT_DIR/examples/on_policy_distillation_trainer/reward_fn_math_verify.py"
tool_config_path="$ROOT_DIR/examples/on_policy_distillation_trainer/config/tool_config/sandbox_fusion_tool_config.yaml"

checkpoint_dir="$ROOT_DIR/checkpoints/$project_name/$experiment_name"
log_dir="$ROOT_DIR/data-log/$project_name/$experiment_name"
val_dump_dir="$log_dir/validation_jsonl"
rollout_dump_dir="$log_dir/rollout_jsonl"
mkdir -p "$checkpoint_dir" "$log_dir" "$val_dump_dir" "$rollout_dump_dir"

max_prompt_length=512
max_response_length=12288
sequence_budget=$(( max_prompt_length + max_response_length ))

train_batch_size=128
ppo_mini_batch_size=32
ppo_micro_batch_size_per_gpu=8
n_resp_per_prompt=8

actor_ppo_max_token_len=18000
infer_ppo_max_token_len=16000

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HYDRA_FULL_ERROR=1
export RAY_LOGGING_LEVEL="${RAY_LOGGING_LEVEL:-DEBUG}"
export VERL_MONKEY_PATCH_TIMING="${VERL_MONKEY_PATCH_TIMING:-1}"
export VERL_MONKEY_PATCH_TIMING_SLOW_MS="${VERL_MONKEY_PATCH_TIMING_SLOW_MS:-50}"
export VERL_MONKEY_PATCH_GPU_UTIL="${VERL_MONKEY_PATCH_GPU_UTIL:-1}"
export VERL_MONKEY_PATCH_GPU_UTIL_SAMPLE_MS="${VERL_MONKEY_PATCH_GPU_UTIL_SAMPLE_MS:-200}"
export VERL_REWARD_DEBUG="${VERL_REWARD_DEBUG:-0}"
export VERL_REWARD_DEBUG_SLOW_MS="${VERL_REWARD_DEBUG_SLOW_MS:-50}"
export VERL_REWARD_DEBUG_PREVIEW_CHARS="${VERL_REWARD_DEBUG_PREVIEW_CHARS:-160}"

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size="$train_batch_size" \
    data.max_prompt_length="$max_prompt_length" \
    data.max_response_length="$max_response_length" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    reward.custom_reward_function.path="$reward_fn_path" \
    reward.custom_reward_function.name=compute_score_math_verify \
    actor_rollout_ref.model.path="$base_model_path" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$ppo_mini_batch_size" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ppo_micro_batch_size_per_gpu" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$actor_ppo_max_token_len" \
    actor_rollout_ref.actor.clip_ratio_low=0.0003 \
    actor_rollout_ref.actor.clip_ratio_high=0.0004 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.n="$n_resp_per_prompt" \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$infer_ppo_max_token_len" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$infer_ppo_max_token_len" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$tool_config_path" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.default_local_dir="$checkpoint_dir" \
    trainer.validation_data_dir="$val_dump_dir" \
    trainer.rollout_data_dir="$rollout_dump_dir" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.resume_mode=auto \
    trainer.val_before_train=False \
    trainer.save_freq=10 \
    trainer.test_freq=50 \
    trainer.total_training_steps="$total_training_steps" \
    trainer.total_epochs=5 \
    "$@" \
    2>&1 | tee "$log_dir/$experiment_name.log"
