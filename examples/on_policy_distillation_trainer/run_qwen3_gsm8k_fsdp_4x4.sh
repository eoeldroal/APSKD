#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

############################ Quick Config ############################

ROLLOUT_NAME="vllm" # sglang or vllm

STUDENT_MODEL_PATH="${REPO_ROOT}/checkpoints/Qwen3-1.7B"
TEACHER_MODEL_PATH="${REPO_ROOT}/checkpoints/Qwen3-8B"
MATH_VERIFY_REWARD_FN_PATH="${REPO_ROOT}/examples/on_policy_distillation_trainer/reward_fn_math_verify.py"

# Use direct forward-KL distillation on teacher top-k tokens.
USE_POLICY_GRADIENT=False
DISTILLATION_LOSS_MODE="forward_kl_topk"
USE_FUSED_KERNELS=False

DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

PROJECT_NAME='verl_on_policy_distillation_qwen3_deepscaler_fsdp'

MAX_PROMPT=512
MAX_RESPONSE_LENGTH=8192
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
ROLLOUT_MAX_BATCHED_TOKENS=24576
TEACHER_MAX_BATCHED_TOKENS=98304
ENGINE_MAX_NUM_SEQS=512

TRAIN_PROMPT_BSZ=32
STUDENT_MAX_TOKEN_LEN_PER_GPU=25000
USE_DYNAMIC_BSZ=True

STUDENT_WORLD_SIZE=4

TEACHER_RESOURCE_POOL=True
TEACHER_WORLD_SIZE=4
TEACHER_TP=1
TEACHER_GPU_MEMORY_UTILIZATION=0.70

ROLLOUT_GPU_MEMORY_UTILIZATION=0.70

SP=1

EXP_NAME="fsdp/student-Qwen3-1.7B/teacher-Qwen3-8B/train-deepscaler/loss-${DISTILLATION_LOSS_MODE}/pg-${USE_POLICY_GRADIENT}/4x4-v9"

DUMP_ROOT="${REPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXP_NAME}/dumps"
ROLLOUT_DUMP_DIR="${DUMP_ROOT}/rollout"
VALIDATION_DUMP_DIR="${DUMP_ROOT}/validation"

ENFORCE_EAGER=False

# MONKEY PATCH: enable detailed timing logs for rollout / teacher / reward hotspots.
MONKEY_PATCH_TIMING="${MONKEY_PATCH_TIMING:-1}"
MONKEY_PATCH_TIMING_SLOW_MS="${MONKEY_PATCH_TIMING_SLOW_MS:-50}"
MONKEY_PATCH_GPU_UTIL="${MONKEY_PATCH_GPU_UTIL:-1}"
MONKEY_PATCH_GPU_UTIL_SAMPLE_MS="${MONKEY_PATCH_GPU_UTIL_SAMPLE_MS:-200}"

############################ Paths ############################

DATA_ROOT="${REPO_ROOT}/data"
deepscaler_train_path="${DATA_ROOT}/deepscaler/train.parquet"
aime24_test_path="${DATA_ROOT}/aime-2024.parquet"
math500_test_path="${DATA_ROOT}/math500/test.parquet"

TRAIN_FILES="['$deepscaler_train_path']"
TEST_FILES="['$aime24_test_path', '$math500_test_path']"

############################ Parameter Groups ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_PROMPT_BSZ
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

REWARD=(
    reward.custom_reward_function.path="${MATH_VERIFY_REWARD_FN_PATH}"
    reward.custom_reward_function.name=compute_score_math_verify
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL_PATH}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER
)

DISTILLATION=(
    distillation.enabled=True
    distillation.num_workers=4
    distillation.teacher_model.enable_resource_pool=$TEACHER_RESOURCE_POOL
    distillation.teacher_model.n_gpus_per_node=$TEACHER_WORLD_SIZE
    distillation.teacher_model.nnodes=1
    distillation.teacher_model.model_path="${TEACHER_MODEL_PATH}"
    distillation.teacher_model.inference.tensor_model_parallel_size=$TEACHER_TP
    distillation.teacher_model.inference.name=$ROLLOUT_NAME
    distillation.teacher_model.inference.gpu_memory_utilization=$TEACHER_GPU_MEMORY_UTILIZATION
    distillation.teacher_model.inference.enforce_eager=$ENFORCE_EAGER
    distillation.teacher_model.inference.max_model_len=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_batched_tokens=$TEACHER_MAX_BATCHED_TOKENS
    distillation.teacher_model.inference.max_num_seqs=$ENGINE_MAX_NUM_SEQS
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.topk=128
    +distillation.distillation_loss.memory_efficient=True
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.loss_max_clamp=$DISTILLATION_LOSS_MAX_CLAMP
    distillation.distillation_loss.log_prob_min_clamp=$DISTILLATION_LOG_PROB_MIN_CLAMP
)

STUDENT=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP
)

ROLLOUT=(
    actor_rollout_ref.rollout.agent.num_workers=4
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_BATCHED_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$ENGINE_MAX_NUM_SEQS
    actor_rollout_ref.rollout.n=1
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=50
    trainer.test_freq=25
    trainer.total_epochs=5
    trainer.val_before_train=True
    trainer.use_legacy_worker_impl=disable
    trainer.rollout_data_dir=$ROLLOUT_DUMP_DIR
    trainer.validation_data_dir=$VALIDATION_DUMP_DIR
)

############################ Launch ############################

cd "${REPO_ROOT}"

# MONKEY PATCH: searchable env flags for timing instrumentation.
export VERL_MONKEY_PATCH_TIMING="${MONKEY_PATCH_TIMING}"
export VERL_MONKEY_PATCH_TIMING_SLOW_MS="${MONKEY_PATCH_TIMING_SLOW_MS}"
export VERL_MONKEY_PATCH_GPU_UTIL="${MONKEY_PATCH_GPU_UTIL}"
export VERL_MONKEY_PATCH_GPU_UTIL_SAMPLE_MS="${MONKEY_PATCH_GPU_UTIL_SAMPLE_MS}"

# main_ppo already declares Hydra defaults via
# @hydra.main(config_path="config", config_name="ppo_trainer"),
# so no extra --config-path/--config-name is required here.
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${REWARD[@]}" \
    "${DISTILLATION[@]}" \
    "${ROLLOUT[@]}" \
    "${STUDENT[@]}" \
    "${TRAINER[@]}" \
    "$@"
