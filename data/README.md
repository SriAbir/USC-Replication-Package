# Dataset Documentation

This directory contains the conversation datasets and human stopping-point annotations used in our study of stopping conditions in multi-agent software engineering conversations.

The datasets support three research questions (RQ1–RQ3), covering human stopping judgments, conversational metric evaluation, and an external case study using ChatDev.

## 1. Directory Structure

```text
data/
├── README.md
├── primary_study/
│   ├── conversations/
│   ├── task_requirements/
│   │   └── task_requirements.csv
│   └── annotations/
│       ├── annotation_guidelines.pdf
│       └── human_stopping_annotations.xlsx
└── chatdev_case_study/
    ├── conversations/
    ├── annotations/
    └── provenance.csv
```

## 2. Primary Study Dataset

The primary dataset contains **40 multi-agent programming conversations** generated using our Designer–Programmer framework.

The conversations cover seven programming tasks:

1. Fibonacci Game
2. Caesar Cipher Game
3. Anagram Game
4. Palindrome
5. Two Sum
6. Linked List Sum
7. Median of Two Sorted Arrays

Each conversation records interactions between the Designer and Programmer agents as they attempt to solve the assigned programming task.

The original task descriptions are available in [`primary_study/task_requirements/task_requirements.csv`](primary_study/task_requirements/task_requirements.csv).

### Human Stopping-Point Annotations

Seven human annotators independently evaluated the conversations to determine appropriate stopping points and outcomes.

The annotation materials are available at:

- **[Annotation Guidelines](primary_study/annotations/annotation_guidelines.pdf):** The original instructions and decision criteria provided to annotators.
- **[Human Stopping Annotations](primary_study/annotations/human_stopping_annotations.xlsx):** The recorded stopping-point and outcome judgments.

The annotations distinguish three possible decisions:

- **Stop-Success (S):** The task has been satisfactorily completed, and further interaction is unnecessary.
- **Stop-Failure (F):** The task remains unsuccessful, and further interaction is unlikely to produce meaningful progress.
- **Continue:** Further interaction is warranted.

The annotators mark the stopping row followed by the stopping decision beside each file row. Example 4-S means the annotator decided the 4th conversation row to be the stopping row and the decision was a success i.e., the conversation successfully completed the task at row 4. 

The original conversation row identifiers should be preserved when interpreting stopping-point annotations.

These data support RQ1, which examines human stopping judgments, and RQ2, which evaluates their relationship with task-grounding and repetition metrics.

## 3. ChatDev Case Study Dataset

The external case study contains five conversations extracted from publicly available ChatDev execution logs.

| Project | Extracted Turns | Application Type |
|---|---:|---|
| BookBreeze | 35 | E-book reader |
| BMI Calculator | 12 | Utility application |
| Maze Generator | 16 | Maze generation |
| QR Code Generator | 13 | QR code utility |
| DesignDream | 49 | Interactive interior design |
| **Total** | **125** | |

The extracted conversations preserve the chronological agent interactions, generated code, agent roles, and development phases.

The original execution logs were processed to remove logging metadata and duplicate entries.

The corresponding human stopping-point annotations are stored in `chatdev_case_study/annotations/`.

Source information for the ChatDev projects is documented in `chatdev_case_study/provenance.csv`.

The original log files are also preserved

These data support RQ3, which investigates whether the stopping-related patterns observed in the primary study also appear in independently generated multi-agent development conversations.

## 4. Data Usage and Reproducibility

The datasets are intended to support the reproduction and extension of our human annotation and metric-based stopping analyses.

Researchers should:

- Preserve original conversation-turn and row identifiers when matching metric values to human annotations.
- Distinguish stopping outcomes from stopping positions, as annotators may agree on the outcome but disagree on when to stop.
- Consult the original annotation guidelines before interpreting the stopping judgments.
- Use the metric-calculation and statistical-analysis scripts provided in the repository to reproduce the reported results.

Metric outputs and statistical summaries are stored separately in the repository's `results/` directory.

## 5. Data Provenance and Reuse

The primary conversations were generated using the multi-agent framework provided in the repository's `framework/` directory.

The ChatDev conversations originate from the publicly available [ChatDev repository](https://github.com/OpenBMB/ChatDev). Their redistribution and reuse remain subject to applicable upstream licensing conditions.

Please consult the repository's license information before redistributing or modifying the datasets.