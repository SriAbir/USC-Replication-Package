# USC: Towards Measurement-Driven Conversation Termination in Multi-Agent Software Engineering

Replication package accompanying the research study **USC: Towards Measurement-Driven Conversation Termination in Multi-Agent Software Engineering**.

## Overview

This repository provides conversation traces, human stopping-point annotations, metric outputs, evaluation results, and analysis scripts for investigating when multi-agent software-engineering conversations should terminate. The study examines two complementary aspects of agent interaction:

- **Task grounding:** how closely agent responses remain aligned with the assigned programming task (BERTScore F1, Raw SBERT, and Task-Aware SBERT).
- **Conversational repetition:** how much content recurs across turns (BLEU, ROUGE-L, and Windowed Maximum SBERT Cosine Similarity, WMSC).

The analysis compares these measures with human-identified **Stop-Success**, **Stop-Failure**, and **Continue** judgments. The package also includes a separate exploratory case study based on five ChatDev projects. This repository provides evidence for studying stopping conditions; it does **not** provide a validated autonomous stopping controller.

## Repository organization

| Location | Contents |
| --- | --- |
| `CITATION.cff` | Citation metadata for this package. |
| `data/primary_study/conversations/` | Original Designer–Programmer conversation workbooks, organized by seven programming tasks. |
| `data/primary_study/annotations/AnnotationRules.pdf` | Human annotation instructions and stopping criteria. |
| `data/primary_study/task_requirements/task_requirements.csv` | Task-prompts used to interpret completion and divergence. |
| `data/chatdev_case_study/Raw_logs/` | Original ChatDev execution logs for five projects. |
| `data/chatdev_case_study/Processed_conversations/` | Extracted ChatDev conversation workbooks. |
| `data/chatdev_case_study/annotations/` | ChatDev stopping-point annotations. |
| `data/chatdev_case_study/provenance.csv` | Provenance information for ChatDev projects. |
| `framework/AgentFrameworkCode/` | Lightweight multi-agent framework source code. |
| `framework/agents/` | Model-pair and model-snapshot records. |
| `framework/configs/` | Experiment environment configuration. |
| `results/rq1/` | Primary-study human annotations and outcome summaries. |
| `results/rq2/MetricScore/` | Per-turn task-grounding and lexical repetition metric workbooks, grouped by task. |
| `results/rq2/WMSC/` | Separate WMSC/loop metric workbooks. |
| `results/rq2/MetricsEvaluation/` | AUROC, stop-to-post-stop difference, and stopping-point proximity outputs. |
| `results/rq3/Metrics/` | ChatDev metric workbooks. |
| `results/rq3/Evaluation/` | ChatDev evaluation outputs based on stopping references. |
| `scripts/metrics/` | Task-specific metric computation scripts and `WMSC.py`. |
| `scripts/EvaluationScripts_RQ1_3/` | Scripts for AUROC, stop-to-post-stop difference, and proximity analyses. |
| `scripts/preprocessing/` | ChatDev log-to-conversation extraction script. |
| `scripts/figures/` | Script for generating Results-section figures. |

The seven primary tasks are **Anagram game**, **Caesar Cipher game**, **Fibonacci game**, **Linked List Addition**, **Median of Two Sorted Arrays**, **Palindrome**, and **Two Sum**. The five ChatDev projects are **BMI Calculator**, **BookBreeze**, **DesignDream**, **Maze Generator**, and **QR Code Generator**.

## Study data and stopping references

The primary study uses seven human annotators and a corpus described in the manuscript as 40 selected conversations. The directory listing supplied for this package shows **39 primary conversation workbooks**; users should check this difference before attempting to reproduce results reported for 40 conversations. The `results/rq1/human_stoppingpointAnnotation.xlsx` workbook contains the human stopping judgments.

In the study protocol:

- **Stop-Success** identifies the earliest complete, compilable, executable, and functionally satisfactory solution meeting the task requirements.
- **Stop-Failure** identifies persistent abandonment of essential task requirements, according to the annotation rules; compilation errors alone are not sufficient.
- **Continue** indicates that no defensible stopping point was identified.

Two overlapping reference groups are used for the RQ2 evaluation:

1. **Strict agreement:** unanimous agreement on the stopping outcome, with annotated stopping rows differing by no more than two rows. The reference set consists of the distinct selected stopping rows.
2. **Majority stopping-row agreement:** at least five of seven annotators select the same stopping row, irrespective of outcome agreement. The modal stopping row is used as the reference.

A conversation can qualify for both groups. The two groups must **not** be added together as disjoint samples. Consult the annotation workbook and the paper for the exact study-specific classification.



## Replication Instructions

Follow the steps below to set up the multi-agent framework, generate Designer–Programmer conversations, and compute the conversational metrics.

### 1. Prerequisites

Before running the framework, ensure that the following requirements are satisfied:

1. **Python:** Python 3.9 or later is recommended.

2. **Ollama:** Download and install Ollama from [https://ollama.com/](https://ollama.com/). Ollama should be installed on both servers hosting the Designer and Programmer LLMs.

3. **LLM Models:** Pull the required models on their respective servers using Ollama. For example:

   ```bash
   ollama pull llama3
   ```

   Replace `llama3` with the model you intend to use. Ensure that the models are available and running on their respective servers before executing the framework.

### 2. Installation and Setup

**Step 1: Clone the repository**

```bash
git clone https://github.com/SriAbir/xxxx
```

**Step 2: Navigate to the project directory**

```bash
cd framework/AgentFrameworkCode
```

**Step 3: Create and activate a Python virtual environment**

Create the environment:

```bash
python -m venv venv
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

Or on Windows:

```bash
venv\Scripts\activate
```

Install the Python dependencies required by the framework. Refer to the imports in the source files and the experimental configuration for the required packages.

**Step 4: Configure the multi-agent framework**

Open `main_agentc.py` in the `framework/AgentFrameworkCode/` directory and update the following configurations:

- **Server addresses:** Replace the existing Ollama server addresses with the addresses of your own Designer and Programmer servers.

  ```python
  server_address="http://deeperthought.cse.chalmers.se"
  server_address="http://deepthought.cse.chalmers.se"
  ```

- **Maximum iterations:** Modify `MAX_ITERATIONS` to specify the maximum number of Designer–Programmer interaction cycles.

- **Output filename:** Specify the Excel file where the conversation transcript should be saved. For example:

  ```python
  agentProgrammer.save_to_excel("programmer_conversation_test_comp2.xlsx")
  ```

Ensure that the configured server addresses are reachable and the required Ollama models are available.

**Step 5: Execute the framework**

Navigate to the directory containing `main_agentc.py` and execute:

```bash
python main_agentc.py --prompt "Write a mathematical game of Fibonacci"
```

You can replace the example prompt with any programming task of your choice.

**Step 6: Retrieve the generated conversation**

After execution completes, the Designer–Programmer conversation transcript will be saved in the Excel file specified in Step 4.

The transcript can be used to examine the interaction history, generated code, revisions, and agent behavior across iterations.

### 3. Computing Conversational Metrics

Once the conversation files have been generated, the metrics can be calculated using the Python scripts available under:

```text
scripts/metrics/
```

The replication package includes scripts for calculating the following metrics:

- **Task-grounding metrics:** BERTScore, Raw SBERT, and Task-Aware SBERT.
- **Conversational repetition metrics:** BLEU, ROUGE-L, and Windowed Maximum SBERT Cosine Similarity (WMSC).

Select the appropriate task-specific metric script and configure its input and output paths according to the generated conversation files.

The resulting metric scores can subsequently be used with the evaluation scripts under `scripts/EvaluationScripts_RQ1_3/` to analyze AUROC, stop-to-post-stop differences, and stopping-point proximity using human-annotated stopping references.

### 4. Setup Complete

**Congratulations!** You have successfully set up a Designer–Programmer–Compiler multi-agent framework in which LLM agents collaborate iteratively to solve programming tasks.

You can now generate new conversation trajectories, analyze task grounding and conversational repetition, and investigate potential stopping conditions using the USC evaluation methodology.