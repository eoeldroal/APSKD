from examples.data_preprocess.nemotron_cascade_rl_math_multiturn_w_tool import (
    detect_task_type,
    normalize_problem,
)


def test_normalize_problem_removes_boxed_instruction_but_keeps_task_semantics():
    problem = (
        "Your task is to review and critique the solution paragraph by paragraph. "
        "Once you identify an error in a paragraph, return the index of the paragraph where "
        "the earliest error occurs. Otherwise, return the index of -1. "
        "Please put your final answer (i.e., the index) in \x08oxed{}."
    )

    normalized = normalize_problem(problem)

    assert "oxed{}" not in normalized
    assert "final answer" not in normalized.lower()
    assert "return the index of -1" in normalized


def test_normalize_problem_keeps_non_boxing_format_constraints():
    problem = "Find the distance between points A and B. Give your answer as a number without units."

    normalized = normalize_problem(problem)

    assert normalized == problem


def test_normalize_problem_preserves_boxed_content_inside_problem_statement():
    problem = r"If $x+y=3$, then $x^2+y^2=\boxed{\text{________}}$."

    normalized = normalize_problem(problem)

    assert normalized == problem


def test_detect_task_type_flags_solution_critique():
    critique_problem = "[Solution]\n<paragraph_0>\n...\n</paragraph_0>\nReview and critique this solution."
    direct_problem = "Two cars travel between A and B. After the 15th meeting, how many hours?"

    assert detect_task_type(critique_problem) == "solution_critique"
    assert detect_task_type(direct_problem) == "direct_math"
