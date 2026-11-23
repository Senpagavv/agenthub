# AgentHub

AgentHub runs a batch of text tasks, sends selected items to a second model for review, and compares the answers with reference labels. Use it to see which cases improve and how much an extra review costs. The Python command-line tool has no package dependencies.

[中文使用说明](docs/guide.zh.md) · [Four-item example](examples/items.jsonl) · [Use terms](USE.md)

## Try it without an API key

Use Python 3.10 or later. From the repository root, run:

```sh
python examples/demo.py
```

The question is whether a text explicitly reports a **train** delay. The first pass mistakes a delayed bus for a delayed train. A second pass reviews that one item. Both runs use the same four reference labels.

```text
First pass: 3/4 correct, CNY 0.00100 per input
First pass + one review: 4/4 correct, CNY 0.00175 per input
```

The demo replays illustrative answers and costs through the evaluator. It runs locally and does not call a provider. Inspect the [first pass](examples/first-pass.jsonl), [review](examples/review.jsonl) and [reference labels](examples/gold.jsonl).

## Run your own task

Save one JSON object per line with a unique string `id` and a `text` field. Set the environment variable for the provider you want to use in your shell. DeepSeek reads `AGENTHUB_DEEPSEEK_KEY`. TypeSafe Jev reads `AGENTHUB_JEV_KEY`. Live requests use your provider account.

```sh
python hub.py batch --task trains --input examples/items.jsonl --json --prompt "Return JSON with boolean label. Does this text explicitly report a train delay? {text}" --limit 4 --workers 1 --budget 0.20
python hub.py eval --task trains --gold examples/gold.jsonl --field label
python hub.py sample --task trains -n 4
```

For structured yes/no decisions, use the same records with Jev:

```sh
python hub.py jbatch --task trains-jev --input examples/items.jsonl --questions-file examples/question.json --limit 4 --workers 1 --budget 0.20
python hub.py eval --task trains-jev --gold examples/gold.jsonl --field label
```

For selective review, give the second model a JSONL containing only the items to recheck, preserving the original IDs. Then pass its task name to the evaluator:

```sh
python hub.py eval --task trains --gold examples/gold.jsonl --field label --fallback trains-review
```

A successful review replaces that item’s first answer. Missing or failed predictions count against the reference set. Costs are shown in CNY per reference item. The real cost of live use depends on provider pricing, retries and any already-running requests when a batch stops. Check `config.json` and your provider’s account controls before running a large batch.

## Where results go

The runner writes task records, model responses, prompts and accounting logs under `runs/`. Set `AGENTHUB_WORKSPACE` to choose a different directory. Task records are local and may contain the input text you supplied. The sample data bundled here is illustrative.

The public runner includes the local project’s single-request, batch, review, evaluation, sample and recipe commands. Run `python hub.py --help` for the list, or use `--help` after a command. The optional `--via defapi` setting selects a third-party relay only when requested. The default chat path is DeepSeek, and the default Jev path is TypeSafe.
