# MiloNet: TMLR code artifact

This repository contains the code released with the accepted MiloNet paper:
the main system, prompts, corpus-construction scripts, baseline runners,
evaluation scripts, and inference-cost tools.

RAGBench data, the processed corpus used in the experiments, cached outputs,
API credentials, and private experiment logs are not included. Instructions
for downloading the evaluation split are provided below.

## Repository contents

- `MiloNet.py`, `UI.py`, `helper.py`, `core_prompts.py`, `parameters.ini`, and
  `Modes/`: MiloNet runtime, prompts, routing, agent hierarchy, cited
  synthesis, and final answer checks.
- `corpus_construction/`: staged utilities used to construct the fixed textual
  folder/document registry from the RAGBench HotpotQA records.
- `run_prepared_corpus_ui.py`: launcher for a separately prepared
  `Original_documents`, `Processed_documents`, and runtime registry.
- `vanilla_rag_runner.py`, `rankgpt_runner.py`, and `light_raptor.py`: baseline
  runners.
- `response_quality_runner.py`, `response_quality_diagnostics.py`,
  `baseline_milo_score.py`, `add_retrieval_precision.py`, and
  `compare_correctness.py`: evaluation and metric utilities.
- `cost_logging.py`, `summarize_cost_logs.py`, and
  `build_revision_cost_table.py`: per-call logging and inference-cost
  aggregation utilities.

## Corpus scope

The reported experiments used the plain-text HotpotQA documents in
`ragbench_test.jsonl`, which MiloNet writes to Markdown document stores during
corpus initialization. The shared conversion function is defined in
`local_pdf_parser_mac.py`; its PDF-specific branch was not used in the reported
experiments.

The reported evaluation used the default `Chat` operation mode for every
query. Only `Modes/Chat.txt` is included because the other interactive modes
were not used in the reported evaluation.

`MILONET_VARIANT` selects the core or full pipeline. The prepared-corpus
launcher runs MiloNet against an existing registry, and the cost hooks record
per-call usage.

## Environment

The code was developed with Python 3.10. From the repository root, create a
local environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

`requirements.txt` records the full Python dependency snapshot from the
development environment:

```bash
python -m pip install -r requirements.txt
```

This dependency list is broad and targets Python 3.10 on Apple Silicon macOS,
because the runtime still imports Apple MLX and PDF packages. Some pinned
legacy packages may produce installer warnings. Use a dedicated environment.
The corpus-construction utilities use a separate environment because their
pinned NumPy and OpenAI versions differ from the main MiloNet runtime:

```bash
python3.10 -m venv .venv-corpus
.venv-corpus/bin/python -m pip install --upgrade pip setuptools wheel
.venv-corpus/bin/python -m pip install -r corpus_construction/requirements.txt
```

Use `.venv/bin/python` for the MiloNet runtime and `.venv-corpus/bin/python`
for scripts under `corpus_construction/`.

Create a local credential file and set an OpenAI API key:

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

Do not commit `.env`. The other entries in `.env.example` configure optional
integrations.

## Dataset and fixed registry

The experiments use the 390-record HotpotQA test split from RAGBench. Download
the pinned dataset revision and write it to `ragbench_test.jsonl`:

```bash
python - <<'PY'
from datasets import load_dataset

data = load_dataset(
    "galileo-ai/ragbench",
    "hotpotqa",
    split="test",
    revision="97808f3e5fd16ede40bbff6c2949af8139b2eb7b",
)

assert len(data) == 390
data.to_json("ragbench_test.jsonl", orient="records", lines=True)
PY
```

The pinned source is available on the
[RAGBench dataset page](https://huggingface.co/datasets/galileo-ai/ragbench/tree/97808f3e5fd16ede40bbff6c2949af8139b2eb7b/hotpotqa).
Corpus construction reads the source documents. The inference and evaluation
scripts also read the query; some metric adapters require sentence-level
document representations and benchmark evidence keys.

[`corpus_construction/README.md`](corpus_construction/README.md) documents the
offline construction steps. The workflow:

1. extracts and initially groups the RAGBench source documents;
2. initializes document stores, indexes, summaries, and
   `processed_files_map.json`;
3. derives `doc_summary_dict.json` from exported Chroma metadata;
4. performs LLM-assisted folder organization and document-ID consistency
   checks; and
5. materializes the final folder/document registry used by MiloNet.

The processed-corpus registry is built before evaluation and reused for every
query. Query-time routing selects eligible document tools from this registry;
it does not move documents or change the hierarchy.

## Run MiloNet with a prepared registry

The launcher expects three existing directories:

- `Original_documents`: finalized structured text documents;
- `Processed_documents`: document stores, nodes, indexes, summaries, and
  `processed_files_map.json`; and
- `Runtime`: `global_id_map.json`, `file_structure.txt`,
  `tool_access_info.txt`, `all_doc_tool.txt`, and the Chroma stores.

Prepare a JSONL question file with one object per line. The batch reader accepts
`original_question` or `question`:

```json
{"original_question": "Question supported by the prepared corpus", "judged_quality": "Good"}
```

`judged_quality` is accepted for compatibility with the original question
files and is not used during answer generation.

Set paths and start with one query:

```bash
ORIGINAL=/absolute/path/to/Original_documents
PROCESSED=/absolute/path/to/Processed_documents
RUNTIME=/absolute/path/to/Runtime
QUESTIONS=/absolute/path/to/questions.jsonl
RESULTS=/absolute/path/to/results.jsonl

TOKENIZERS_PARALLELISM=false \
MILONET_STRICT_INITIALIZATION=1 \
MILONET_VARIANT=full \
MILONET_UI_BATCH_INPUT="$QUESTIONS" \
MILONET_UI_BATCH_OUTPUT="$RESULTS" \
MILONET_UI_BATCH_LIMIT=1 \
python run_prepared_corpus_ui.py \
  --structured-original "$ORIGINAL" \
  --processed-documents "$PROCESSED" \
  --runtime-dir "$RUNTIME" \
  --host 127.0.0.1 \
  --port 7861
```

After initialization, open `http://127.0.0.1:7861` and click `Send` to start
the JSONL batch specified by `MILONET_UI_BATCH_INPUT`. This batch interface
does not use text entered in the Gradio text box. It processes up to
`MILONET_UI_BATCH_LIMIT` rows, appends completed results to
`MILONET_UI_BATCH_OUTPUT`, and terminates after the batch is complete.

- `MILONET_VARIANT=core` stops after the cited synthesis response.
- `MILONET_VARIANT=full` continues through the final answer-checking routines.

Use a new output filename for a fresh run because the batch writer appends
completed rows.

## Baseline runners

Run each script with `--help` to see its options:

```bash
python vanilla_rag_runner.py --help
python rankgpt_runner.py --help
python light_raptor.py --help
```

`vanilla_rag_runner.py` uses a MiloNet retrieval-analysis file to obtain the
matched evidence count for each query. `rankgpt_runner.py` reads the RAGBench
JSONL records directly. `light_raptor.py` uses either the prepared summary
database or a saved RAPTOR output containing Level-1 summaries.

### Reported baseline settings

The RankGPT entry in Table 3 uses `o4-mini` for document ranking and answer
generation:

```bash
python rankgpt_runner.py \
  --ragbench-file /path/to/ragbench_test.jsonl \
  --model o4-mini \
  --output /path/to/rankgpt_o4mini.json
```

The RAPTOR entry uses the selected `c4_k12` configuration and `o4-mini`. Its
Level-1 summaries are read from the saved output of that selected
configuration:

```bash
python light_raptor.py \
  --benchmark-file /path/to/ragbench_test.jsonl \
  --level1-summary-source /path/to/raptor_run_ragbench_c4_k12.jsonl \
  --cluster-select 4 \
  --global-top-k 12 \
  --llm-model o4-mini \
  --output /path/to/raptor_c4_k12_o4mini.jsonl
```

The Appendix B configuration sweep used `gpt-4o-mini`. For Table 3, the
selected Level-1 summaries were held fixed while the cluster-summary and
answer-generation calls used `o4-mini`. The saved Level-1 summary source is
not included in this repository.

Convert a RAPTOR output to the sentence-key format used by the evaluation
scripts:

```bash
python raptor_evaluation_adapter.py \
  --raptor-output /path/to/raptor_c4_k12_o4mini.jsonl \
  --benchmark-file /path/to/ragbench_test.jsonl \
  --output /path/to/evaluation_data_raptor.json
```

These runners call model-provider APIs. Model availability and provider-side
behavior may have changed since the experiments were run.

## Evaluation

Run answer-quality evaluation on a prepared retrieval-analysis JSONL file:

```bash
python response_quality_runner.py \
  --input /path/to/retrieval_analysis.jsonl \
  --output /path/to/response_quality_evaluation.jsonl
```

Pass that output to the diagnostic evaluator:

```bash
python response_quality_diagnostics.py \
  --input /path/to/response_quality_evaluation.jsonl \
  --output /path/to/response_quality_diagnostics.jsonl
```

Both evaluators write one JSON object per line. Some saved files use a `.json`
suffix even though their contents are JSONL.

Answer-quality scoring uses `o4-mini`, and diagnostic scoring uses
`gpt-5-mini`. Faithfulness scoring uses the default `gpt-4o-mini` model from
the pinned `ragas==0.2.14` release.

Use `python SCRIPT.py --help` to inspect the adapter and metric options in
`baseline_milo_score.py`, `add_retrieval_precision.py`, and
`compare_correctness.py`. These scripts require system outputs and benchmark
records that are not included here.

## Inference cost

Cost logging records API calls as JSONL when the relevant runner or MiloNet
environment variables are enabled. Existing logs can be aggregated with:

```bash
python summarize_cost_logs.py \
  --log /path/to/system_cost_log.jsonl \
  --output revision_cost_inputs.json

python build_revision_cost_table.py \
  --input revision_cost_inputs.json \
  --output revision_cost_table.md
```

`revision_cost_inputs.template.json` lists the expected table fields. The raw
cost logs used for the paper are not included.

## Experimental provenance

This artifact preserves the active prompts used in the reported experiments.
Several fixed demonstrations and entity-specific rules overlap at the case
level with the evaluated RAGBench HotpotQA split. This overlap occurs in prompt
components shared by MiloNet-core and MiloNet-full, as well as in final-answer
checks used only by MiloNet-full. Four question-and-answer demonstrations in
the MiloNet-full scope validator overlap with four queries in the evaluation
split.

The prompt templates were fixed across queries. At run time, MiloNet did not
load benchmark reference responses, `judged_quality` labels, or gold sentence
keys. The evaluation split was not fully isolated from prompt development. The
original prompts are retained unchanged so that the artifact matches the
reported experiments.

The diagnostic evaluator receives the answer-quality summary as an auxiliary
hint. Its diagnostic labels are therefore conditioned on that summary and
should not be interpreted as measurements independent of the answer-quality
scores.

## Excluded files

This repository does not contain:

- `.env` and all API credentials;
- RAGBench records and source documents;
- generated `Original_documents`, `Processed_documents`, and runtime stores;
- Chroma databases, model caches, and serialized agent state;
- cached system outputs, evaluation outputs, cost logs, and internal review
  materials; and
- source PDFs and the later PDF convenience pipeline.

Obtain the benchmark data under its applicable terms. Keep generated corpora,
outputs, and credentials outside version control.
