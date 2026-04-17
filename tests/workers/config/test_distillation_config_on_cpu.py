from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
    RolloutConfig,
    SkdConfig,
)


def test_distillation_teacher_budget_stays_unchanged_without_teacher_prompt():
    distill_cfg = DistillationConfig(
        enabled=True,
        teacher_model=DistillationTeacherModelConfig(
            inference=RolloutConfig(
                name="sglang",
                prompt_length=512,
                response_length=8192,
                max_model_len=8705,
                max_num_batched_tokens=8705,
            )
        ),
        distillation_loss=DistillationLossConfig(loss_mode="forward_kl_topk", topk=128),
        skd=SkdConfig(chunk_size=128, verify_top_k=3, max_chunks_per_sample=256, teacher_system_prompt_path=None),
    )

    assert distill_cfg.teacher_model.inference.max_model_len == 8705
    assert distill_cfg.teacher_model.inference.max_num_batched_tokens == 8705
    assert distill_cfg.teacher_model.inference.prompt_length == 8704
    assert distill_cfg.teacher_model.inference.response_length == 1


def test_distillation_teacher_budget_adds_margin_when_teacher_prompt_enabled():
    distill_cfg = DistillationConfig(
        enabled=True,
        teacher_model=DistillationTeacherModelConfig(
            inference=RolloutConfig(
                name="sglang",
                prompt_length=512,
                response_length=8192,
                max_model_len=8705,
                max_num_batched_tokens=8705,
            )
        ),
        distillation_loss=DistillationLossConfig(loss_mode="forward_kl_topk", topk=128),
        skd=SkdConfig(
            chunk_size=128,
            verify_top_k=3,
            max_chunks_per_sample=256,
            teacher_system_prompt_path="/tmp/teacher_prompt.txt",
        ),
    )

    assert distill_cfg.teacher_model.inference.max_model_len == 9217
    assert distill_cfg.teacher_model.inference.max_num_batched_tokens == 9217
    assert distill_cfg.teacher_model.inference.prompt_length == 9216
    assert distill_cfg.teacher_model.inference.response_length == 1
