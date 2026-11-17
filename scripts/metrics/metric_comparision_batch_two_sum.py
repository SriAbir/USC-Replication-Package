import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer


# ============================================================
# 1. CODE -> NATURAL LANGUAGE SUMMARY
# ============================================================

NEUTRAL_FALLBACK = "No task-specific pattern detected in this turn."


def _has(pattern, text):
    return re.search(pattern, text) is not None


def summarize_code(code: str) -> str:
    """
    Task-aware summary for the Two Sum problem.

    Generic array traversal alone is NOT interpreted as evidence of the
    target task. Stronger task semantics are added only when the code or
    explanation connects array elements with a target sum and returning
    two distinct indices.
    """

    code_lower = code.lower()
    summary_parts = []

    # --------------------------------------------------------
    # Detect core task signals
    # --------------------------------------------------------

    has_target = _has(
        r"\btarget\b|\btargetsum\b|\btarget_sum\b",
        code_lower
    )

    has_indices = _has(
        r"\bindices\b|\bindex\b|\bi\b|\bj\b",
        code_lower
    )

    has_array = _has(
        r"\barray\b|\bnums\b|\bnums\s*\[|\barr\b|\barr\s*\[",
        code_lower
    )

    has_sum_relation = _has(
        r"\bnums\s*\[\s*i\s*\]\s*\+\s*nums\s*\[\s*j\s*\]|"
        r"\barr\s*\[\s*i\s*\]\s*\+\s*arr\s*\[\s*j\s*\]|"
        r"\bsum\b|\badd\s+up\b|\bcomplement\b",
        code_lower
    )

    has_distinct_elements = _has(
        r"\bi\s*!=\s*j\b|\bj\s*!=\s*i\b|\bnot use the same element\b|"
        r"\bdifferent indices\b|\bdistinct indices\b",
        code_lower
    )

    has_hashing = _has(
        r"\bhash\b|\bhashmap\b|\bhash table\b|\bunordered_map\b|"
        r"\bmap\b|\blookup\b|\bcomplement\b",
        code_lower
    )

    has_nested_loops = _has(
        r"\bfor\s*\([^)]*\)\s*\{?[\s\S]{0,300}\bfor\s*\(",
        code_lower
    )

    has_return_pair = _has(
        r"\breturn\b.*\bindex\b|\breturn\b.*\bi\b.*\bj\b|"
        r"\bresult\b.*\b2\b|\bindices\b",
        code_lower
    )

    # --------------------------------------------------------
    # Array / target-sum evidence
    # --------------------------------------------------------

    if has_array and has_target:
        summary_parts.append(
            "This turn works with an array and a target sum."
        )

    if has_sum_relation and (has_array or has_target):
        summary_parts.append(
            "It checks for two values whose sum equals the target."
        )

    # --------------------------------------------------------
    # Index-return requirement
    # --------------------------------------------------------

    if has_indices and (has_target or has_sum_relation):
        summary_parts.append(
            "It identifies or returns the indices of the matching pair."
        )

    # --------------------------------------------------------
    # Distinct-element requirement
    # --------------------------------------------------------

    if has_distinct_elements:
        summary_parts.append(
            "It ensures that the same array element is not used twice."
        )

    # --------------------------------------------------------
    # Hash/complement strategy
    # --------------------------------------------------------

    if has_hashing and (has_target or has_sum_relation):
        summary_parts.append(
            "It uses complement lookup or a hash-based strategy to find the pair efficiently."
        )

    # --------------------------------------------------------
    # Brute-force strategy
    # --------------------------------------------------------

    if has_nested_loops and (has_target or has_sum_relation):
        summary_parts.append(
            "It uses pairwise array traversal to search for the required two elements."
        )

    # --------------------------------------------------------
    # Return behavior
    # --------------------------------------------------------

    if has_return_pair and (has_target or has_sum_relation):
        summary_parts.append(
            "It returns the two matching indices."
        )

    # --------------------------------------------------------
    # Exactly-one-solution assumption
    # --------------------------------------------------------

    if _has(
        r"\bexactly one solution\b|\bone solution\b|\bunique solution\b",
        code_lower
    ):
        summary_parts.append(
            "It assumes the input contains exactly one valid pair."
        )

    # --------------------------------------------------------
    # Neutral fallback
    # --------------------------------------------------------

    if not summary_parts:
        return NEUTRAL_FALLBACK

    return " ".join(summary_parts)


# ============================================================
# 2. KEYWORD EXTRACTION (TASK-AWARE BOOSTER)
# ============================================================

def extract_keywords(text: str) -> str:
    """
    Extract task-aware concepts for the Two Sum problem.

    Generic array-processing concepts alone are not enough. Target-specific
    concepts are emphasized when array elements are connected with a target
    sum, complement search, and returning two distinct indices.
    """

    text = text.lower()
    keywords = set()

    # --------------------------------------------------------
    # Detect signals
    # --------------------------------------------------------

    has_target = _has(
        r"\btarget\b|\btargetsum\b|\btarget_sum\b",
        text
    )

    has_array = _has(
        r"\barray\b|\bnums\b|\bnums\s*\[|\barr\b|\barr\s*\[",
        text
    )

    has_sum_relation = _has(
        r"\bnums\s*\[\s*i\s*\]\s*\+\s*nums\s*\[\s*j\s*\]|"
        r"\barr\s*\[\s*i\s*\]\s*\+\s*arr\s*\[\s*j\s*\]|"
        r"\bsum\b|\badd\s+up\b|\bcomplement\b",
        text
    )

    has_indices = _has(
        r"\bindices\b|\bindex\b",
        text
    )

    has_distinct_elements = _has(
        r"\bi\s*!=\s*j\b|\bj\s*!=\s*i\b|\bnot use the same element\b|"
        r"\bdifferent indices\b|\bdistinct indices\b",
        text
    )

    has_hashing = _has(
        r"\bhash\b|\bhashmap\b|\bhash table\b|\bunordered_map\b|"
        r"\bmap\b|\blookup\b|\bcomplement\b",
        text
    )

    has_nested_loops = _has(
        r"\bfor\s*\([^)]*\)\s*\{?[\s\S]{0,300}\bfor\s*\(",
        text
    )

    # --------------------------------------------------------
    # Core target concepts
    # --------------------------------------------------------

    if has_array and has_target:
        keywords |= {
            "array",
            "target-sum",
        }

    if has_sum_relation and (has_array or has_target):
        keywords |= {
            "two-sum",
            "pair-sum",
            "matching-pair",
        }

    if has_indices and (has_target or has_sum_relation):
        keywords |= {
            "return-indices",
            "index-pair",
        }

    if has_distinct_elements:
        keywords |= {
            "distinct-indices",
            "no-element-reuse",
        }

    # --------------------------------------------------------
    # Efficient lookup concepts
    # --------------------------------------------------------

    if has_hashing and (has_target or has_sum_relation):
        keywords |= {
            "complement",
            "hash-lookup",
            "hash-map",
            "linear-time-search",
        }

    # --------------------------------------------------------
    # Brute-force concepts
    # --------------------------------------------------------

    if has_nested_loops and (has_target or has_sum_relation):
        keywords |= {
            "pairwise-search",
            "nested-loop",
        }

    # --------------------------------------------------------
    # Exactly-one-solution assumption
    # --------------------------------------------------------

    if _has(
        r"\bexactly one solution\b|\bone solution\b|\bunique solution\b",
        text
    ):
        keywords |= {
            "unique-solution",
        }

    # --------------------------------------------------------
    # Generic programming concepts
    # --------------------------------------------------------

    keywords |= {
        "c-program",
        "code-review",
        "program-logic",
    }

    # Deterministic ordering for reproducibility.
    ordered_keywords = sorted(keywords)

    # Preserve original keyword boosting.
    boosted = (
        " ".join(ordered_keywords)
        + " "
        + " ".join(ordered_keywords)
    )

    return boosted


# ============================================================
# 3. TARGET TASK PROTOTYPE
# ============================================================

TASK_PROTOTYPE = """
This task involves writing and improving a C program that returns the indices of two numbers in an array whose sum equals a target value.
The program should identify exactly one valid pair, avoid using the same element twice, and return the two matching indices in any order.
"""


# ============================================================
# 4. BUILD TASK-AWARE EMBEDDING TEXT
# ============================================================

def build_embedding_text(turn: str) -> str:

    summary = summarize_code(turn)

    keywords = extract_keywords(turn)

    clean_turn = str(turn)[:1500]

    final_text = (
        f"SUMMARY: {summary}\n"
        f"KEYWORDS: {keywords}\n"
        f"CONTENT: {clean_turn}"
    )

    return final_text


# ============================================================
# 5. RAW SBERT TEXT
# ============================================================

def build_raw_embedding_text(turn: str) -> str:
    """
    Raw SBERT baseline.

    No task-aware summary.
    No task-aware keyword augmentation.
    """

    return str(turn)[:1500]


# ============================================================
# 6. BLEU + ROUGE
# ============================================================
#
# These metrics compare CONSECUTIVE turns.
#
# Turn 1 -> NaN
# Turn 2 -> Turn 1 vs Turn 2
# Turn 3 -> Turn 2 vs Turn 3
# etc.
#
# ============================================================

BLEU_METRIC = BLEU(
    effective_order=True
)

ROUGE_METRIC = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)


def compute_consecutive_bleu(
    previous_turn: str,
    current_turn: str
) -> float:

    previous_turn = str(previous_turn).strip()
    current_turn = str(current_turn).strip()

    if not previous_turn or not current_turn:
        return np.nan

    score_0_100 = BLEU_METRIC.sentence_score(
        current_turn,
        [previous_turn]
    ).score

    return float(
        score_0_100 / 100.0
    )


def compute_consecutive_rouge(
    previous_turn: str,
    current_turn: str
) -> float:

    previous_turn = str(previous_turn).strip()
    current_turn = str(current_turn).strip()

    if not previous_turn or not current_turn:
        return np.nan

    scores = ROUGE_METRIC.score(
        previous_turn,
        current_turn
    )

    return float(
        scores["rougeL"].fmeasure
    )


# ============================================================
# 7. SBERT COMPUTATION
# ============================================================

def compute_sbert_scores(
    turns,
    model_name="all-MiniLM-L6-v2"
):

    print(
        f"\nLoading SBERT model: {model_name}"
    )

    model = SentenceTransformer(
        model_name
    )

    # --------------------------------------------------------
    # Encode target task prototype
    # --------------------------------------------------------

    task_embedding = model.encode(
        TASK_PROTOTYPE,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # ========================================================
    # RAW SBERT
    # ========================================================

    raw_texts = [
        build_raw_embedding_text(turn)
        for turn in turns
    ]

    raw_embeddings = model.encode(
        raw_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    raw_similarities = util.cos_sim(
        raw_embeddings,
        task_embedding
    ).squeeze(1)

    raw_scores = (
        raw_similarities
        .cpu()
        .numpy()
        .astype(float)
        .tolist()
    )

    # ========================================================
    # TASK-AWARE SBERT
    # ========================================================

    taskaware_texts = [
        build_embedding_text(turn)
        for turn in turns
    ]

    taskaware_embeddings = model.encode(
        taskaware_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    taskaware_similarities = util.cos_sim(
        taskaware_embeddings,
        task_embedding
    ).squeeze(1)

    taskaware_scores = (
        taskaware_similarities
        .cpu()
        .numpy()
        .astype(float)
        .tolist()
    )

    return (
        raw_scores,
        taskaware_scores
    )


# ============================================================
# 8. BERTSCORE
# ============================================================

def compute_bertscore_scores(
    turns,
    model_type="roberta-large"
):

    print(
        f"\nComputing BERTScore using model: "
        f"{model_type}"
    )

    candidates = [
        str(turn)
        for turn in turns
    ]

    references = [
        TASK_PROTOTYPE
        for _ in turns
    ]

    P, R, F1 = bert_score(
        candidates,
        references,
        model_type=model_type,
        verbose=True,
        rescale_with_baseline=False
    )

    precision_scores = (
        P.cpu()
        .numpy()
        .astype(float)
        .tolist()
    )

    recall_scores = (
        R.cpu()
        .numpy()
        .astype(float)
        .tolist()
    )

    f1_scores = (
        F1.cpu()
        .numpy()
        .astype(float)
        .tolist()
    )

    return (
        precision_scores,
        recall_scores,
        f1_scores
    )


# ============================================================
# 9. CONSECUTIVE-TURN LEXICAL METRICS
# ============================================================

def compute_consecutive_scores(turns):

    bleu_scores = [
        np.nan
    ]

    rouge_scores = [
        np.nan
    ]

    for i in range(
        1,
        len(turns)
    ):

        previous_turn = turns[i - 1]
        current_turn = turns[i]

        bleu = compute_consecutive_bleu(
            previous_turn,
            current_turn
        )

        rouge = compute_consecutive_rouge(
            previous_turn,
            current_turn
        )

        bleu_scores.append(
            bleu
        )

        rouge_scores.append(
            rouge
        )

    return (
        bleu_scores,
        rouge_scores
    )


# ============================================================
# 10. TASK-AWARE GAIN
# ============================================================

def compute_taskaware_gain(
    raw_scores,
    taskaware_scores
):
    """
    Task-Aware Gain:

        TaskAware_SBERT - Raw_SBERT

    Positive:
        Task-aware feature extraction strengthened
        task grounding.

    Near zero:
        Task-aware feature extraction added little.

    Negative:
        Task-aware augmentation reduced task similarity.
    """

    return [
        taskaware - raw

        for raw, taskaware
        in zip(
            raw_scores,
            taskaware_scores
        )
    ]


# ============================================================
# 11. SUMMARY STATISTICS
# ============================================================

def safe_mean(values):

    arr = np.array(
        values,
        dtype=float
    )

    if np.all(
        np.isnan(arr)
    ):
        return np.nan

    return float(
        np.nanmean(arr)
    )


def safe_std(values):

    arr = np.array(
        values,
        dtype=float
    )

    valid = arr[
        ~np.isnan(arr)
    ]

    if len(valid) < 2:
        return np.nan

    return float(
        np.std(
            valid,
            ddof=1
        )
    )


# ============================================================
# 12. PLOTTING
# ============================================================

def plot_scores(
    bert_f1,
    raw_sbert,
    taskaware_sbert,
    bleu,
    rouge,
    output
):

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(11, 15),
        sharex=True
    )

    turn_numbers = np.arange(
        1,
        len(raw_sbert) + 1
    )

    # --------------------------------------------------------
    # BERTScore
    # --------------------------------------------------------

    axes[0].plot(
        turn_numbers,
        bert_f1,
        marker="o"
    )

    axes[0].set_title(
        "BERTScore F1 vs. Task Prototype"
    )

    axes[0].set_ylabel(
        "BERTScore F1"
    )

    axes[0].grid(
        True
    )

    # --------------------------------------------------------
    # Raw SBERT
    # --------------------------------------------------------

    axes[1].plot(
        turn_numbers,
        raw_sbert,
        marker="o"
    )

    axes[1].set_title(
        "Raw SBERT vs. Task Prototype"
    )

    axes[1].set_ylabel(
        "Cosine Similarity"
    )

    axes[1].grid(
        True
    )

    # --------------------------------------------------------
    # Task-Aware SBERT
    # --------------------------------------------------------

    axes[2].plot(
        turn_numbers,
        taskaware_sbert,
        marker="o"
    )

    axes[2].set_title(
        "Task-Aware SBERT vs. Task Prototype"
    )

    axes[2].set_ylabel(
        "Cosine Similarity"
    )

    axes[2].grid(
        True
    )

    # --------------------------------------------------------
    # Consecutive BLEU
    # --------------------------------------------------------

    axes[3].plot(
        turn_numbers,
        bleu,
        marker="o"
    )

    axes[3].set_title(
        "BLEU Between Consecutive Turns"
    )

    axes[3].set_ylabel(
        "BLEU"
    )

    axes[3].grid(
        True
    )

    # --------------------------------------------------------
    # Consecutive ROUGE
    # --------------------------------------------------------

    axes[4].plot(
        turn_numbers,
        rouge,
        marker="o"
    )

    axes[4].set_title(
        "ROUGE-L Between Consecutive Turns"
    )

    axes[4].set_ylabel(
        "ROUGE-L F1"
    )

    axes[4].set_xlabel(
        "Turn Number"
    )

    axes[4].grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved plot to: "
        f"{output}"
    )


# ============================================================
# 13. MAIN ANALYSIS
# ============================================================

def analyze_conversation(
    filepath,
    content_column="content",
    sbert_model="all-MiniLM-L6-v2",
    bertscore_model="roberta-large"
):

    # --------------------------------------------------------
    # Read input Excel
    # --------------------------------------------------------

    df = pd.read_excel(
        filepath
    )

    if content_column not in df.columns:

        raise ValueError(
            f"Column '{content_column}' was not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    turns = (
        df[content_column]
        .dropna()
        .astype(str)
        .tolist()
    )

    if len(turns) == 0:

        raise ValueError(
            "No conversational turns were found."
        )

    print(
        f"\nNumber of turns: "
        f"{len(turns)}"
    )

    # ========================================================
    # RAW + TASK-AWARE SBERT
    # ========================================================

    (
        raw_sbert,
        taskaware_sbert

    ) = compute_sbert_scores(

        turns,

        model_name=
            sbert_model
    )

    # ========================================================
    # BERTSCORE
    # ========================================================

    (
        bert_precision,
        bert_recall,
        bert_f1

    ) = compute_bertscore_scores(

        turns,

        model_type=
            bertscore_model
    )

    # ========================================================
    # CONSECUTIVE BLEU + ROUGE
    # ========================================================

    (
        consecutive_bleu,
        consecutive_rouge

    ) = compute_consecutive_scores(
        turns
    )

    # ========================================================
    # TASK-AWARE GAIN
    # ========================================================

    taskaware_gain = compute_taskaware_gain(
        raw_sbert,
        taskaware_sbert
    )

    # ========================================================
    # BUILD RESULT DATAFRAME
    # ========================================================

    result_df = pd.DataFrame({

        "turn_number":
            np.arange(
                1,
                len(turns) + 1
            ),

        "content":
            turns,

        # ----------------------------------------------------
        # Semantic metrics
        # ----------------------------------------------------

        "BERTScore_Precision":
            bert_precision,

        "BERTScore_Recall":
            bert_recall,

        "BERTScore_F1":
            bert_f1,

        "Raw_SBERT":
            raw_sbert,

        "TaskAware_SBERT":
            taskaware_sbert,

        "TaskAware_Gain":
            taskaware_gain,

        # ----------------------------------------------------
        # Consecutive lexical similarity
        # ----------------------------------------------------

        "Consecutive_BLEU":
            consecutive_bleu,

        "Consecutive_ROUGE_L":
            consecutive_rouge
    })

    return result_df


# ============================================================
# 14. PRINT SUMMARY
# ============================================================

def print_summary(
    result_df
):

    print(
        "\n"
        "========================================"
    )

    print(
        "SUMMARY"
    )

    print(
        "========================================"
    )

    metrics = [

        "BERTScore_Precision",
        "BERTScore_Recall",
        "BERTScore_F1",

        "Raw_SBERT",
        "TaskAware_SBERT",
        "TaskAware_Gain",

        "Consecutive_BLEU",
        "Consecutive_ROUGE_L"
    ]

    for metric in metrics:

        values = result_df[
            metric
        ].values

        mean = safe_mean(
            values
        )

        std = safe_std(
            values
        )

        print(
            f"{metric:25s} "
            f"mean={mean:.4f} "
            f"std={std:.4f}"
        )


# ============================================================
# 15. BATCH PROCESS ALL EXCEL FILES IN CURRENT DIRECTORY
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compute BERTScore, Raw SBERT, Task-Aware SBERT, "
            "Task-Aware Gain, consecutive-turn BLEU, and "
            "consecutive-turn ROUGE-L for every Excel file "
            "in the current directory."
        )
    )

    parser.add_argument(
        "--content-column",
        default="content",
        help=(
            "Column containing conversational turns "
            "(default: content)."
        )
    )

    parser.add_argument(
        "--sbert-model",
        default="all-MiniLM-L6-v2",
        help=(
            "SentenceTransformer model "
            "(default: all-MiniLM-L6-v2)."
        )
    )

    parser.add_argument(
        "--bertscore-model",
        default="roberta-large",
        help=(
            "Model used for BERTScore "
            "(default: roberta-large)."
        )
    )

    args = parser.parse_args()

    current_dir = Path.cwd()

    # Find all .xlsx files in the current directory.
    # Skip temporary Excel files and files generated by this script.
    excel_files = sorted(
        path for path in current_dir.glob("*.xlsx")
        if not path.name.startswith("~$")
        and not path.stem.endswith("_newresults")
    )

    if not excel_files:
        print(
            f"No input .xlsx files found in: {current_dir}"
        )
        return

    print(
        f"Found {len(excel_files)} Excel file(s) in: {current_dir}"
    )

    successful = 0
    failed = 0

    for file_index, input_path in enumerate(excel_files, start=1):

        print(
            "\n" + "=" * 70
        )
        print(
            f"Processing file {file_index}/{len(excel_files)}: "
            f"{input_path.name}"
        )
        print(
            "=" * 70
        )

        output_path = input_path.with_name(
            f"{input_path.stem}_newresults.xlsx"
        )

        plot_output = input_path.with_name(
            f"{input_path.stem}_newresults_plot.png"
        )

        try:
            result_df = analyze_conversation(
                filepath=input_path,
                content_column=args.content_column,
                sbert_model=args.sbert_model,
                bertscore_model=args.bertscore_model
            )

            result_df.to_excel(
                output_path,
                index=False
            )

            print(
                f"\nSaved results to: {output_path.name}"
            )

            print(
                "\n========================================"
            )
            print(
                "PER-TURN SCORES"
            )
            print(
                "========================================"
            )

            for _, row in result_df.iterrows():

                turn = int(row["turn_number"])

                print(f"\nTurn {turn}")
                print(
                    f"  BERTScore P   : "
                    f"{row['BERTScore_Precision']:.4f}"
                )
                print(
                    f"  BERTScore R   : "
                    f"{row['BERTScore_Recall']:.4f}"
                )
                print(
                    f"  BERTScore F1  : "
                    f"{row['BERTScore_F1']:.4f}"
                )
                print(
                    f"  Raw SBERT     : "
                    f"{row['Raw_SBERT']:.4f}"
                )
                print(
                    f"  TaskAware     : "
                    f"{row['TaskAware_SBERT']:.4f}"
                )
                print(
                    f"  TA Gain       : "
                    f"{row['TaskAware_Gain']:.4f}"
                )

                if pd.isna(row["Consecutive_BLEU"]):
                    print("  BLEU prev     : N/A")
                else:
                    print(
                        f"  BLEU prev     : "
                        f"{row['Consecutive_BLEU']:.4f}"
                    )

                if pd.isna(row["Consecutive_ROUGE_L"]):
                    print("  ROUGE prev    : N/A")
                else:
                    print(
                        f"  ROUGE prev    : "
                        f"{row['Consecutive_ROUGE_L']:.4f}"
                    )

            print_summary(result_df)

            plot_scores(
                bert_f1=result_df["BERTScore_F1"].tolist(),
                raw_sbert=result_df["Raw_SBERT"].tolist(),
                taskaware_sbert=result_df["TaskAware_SBERT"].tolist(),
                bleu=result_df["Consecutive_BLEU"].tolist(),
                rouge=result_df["Consecutive_ROUGE_L"].tolist(),
                output=str(plot_output)
            )

            successful += 1

        except Exception as exc:
            failed += 1
            print(
                f"\nERROR while processing {input_path.name}: {exc}"
            )
            print(
                "Skipping this file and continuing with the next one."
            )

    print(
        "\n" + "=" * 70
    )
    print(
        "BATCH PROCESSING COMPLETE"
    )
    print(
        "=" * 70
    )
    print(f"Successful files: {successful}")
    print(f"Failed files    : {failed}")


# ============================================================
# 16. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()