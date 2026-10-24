# Direct RAG vs. agent — trade-off memo (Week 6, Sunday)

**Question.** For a document Q&A product, when is a tool-calling agent worth its cost over
a single retrieval + generation?

**Short answer.** Default to direct RAG. Reach for the agent only when a question *cannot*
be answered by one retrieval — it needs arithmetic, structured/metadata lookups, or several
dependent searches. Everywhere else the agent adds latency, cost, variance and attack
surface for no gain in answer quality.

## The three modes in this repo

| Mode | Endpoint | What happens |
|---|---|---|
| Direct | `POST /ask/langchain` | retrieve top-k → one prompt → one answer; citations verified afterwards |
| Structured | `POST /ask/structured` | same retrieval; model must return validated JSON; invalid output is fed back and retried |
| Agent | `POST /agent` | the model chooses tools (`document_search`, `calculator`, `query_documents`) over several steps, then answers |

## Structural comparison (reasoned from the code, not measured)

| | Direct | Structured | Agent |
|---|---|---|---|
| Model calls per question | 1 | 1 … 1+`LLM_VALIDATION_RETRIES` (default ≤ 3) | 1 … `max_steps`+1 (default ≤ 5) |
| Worst case incl. transport retries | 1 × (1+`LLM_MAX_RETRIES`) | 3 × 3 = 9 | 5 × 3 = 15 |
| Who decides what to retrieve | the code (always the user's question) | the code | **the model** (query rewriting, multiple searches, or none) |
| Input tokens | one prompt | one prompt (+ error feedback on retry) | the **whole history is re-sent every step**, so input grows roughly with steps² |
| Determinism / reproducibility | high (same retrieval every time) | high | lower: the path itself varies run to run |
| Failure modes | provider error; bad citation | + invalid output after retries | + loops, wrong tool, tool misuse, budget exhaustion, tool errors |
| Attack surface | prompt injection can bend the *answer* | same | prompt injection can bend the *actions* (tool calls) — needs scope, allowlists, sanitizing (see AGENT_THREAT_MODEL.md) |
| Debuggability | read one prompt | read one prompt + retry log | read a trace of N steps |
| Latency shape | retrieval + 1 generation | + retries | N × (generation + tool) and unpredictable: a p95 driver |

## Results

Run on your machine against the running API (your Groq key, your corpus):

```
python eval/run_compare.py --email you@example.com --password '...'
```

Paste the table it prints here, with the date and model name:

```
(not run yet — this memo deliberately contains no invented numbers)
```

**Cost accounting.** All three modes now report provider usage in their response
(`usage: {input_tokens, output_tokens, model_calls}`), so the table's *tokens* column is
comparable across modes. Two caveats to keep in the memo: (1) `0 tokens` with `model_calls > 0`
means the provider did not report usage, not that the call was free; (2) structured output's
tokens are summed over **every attempt**, but if all attempts fail the endpoint returns 502
with no usage, so a failing request's spend is not in the table.

How to read it: **source-correct** is the grounding signal (answered *and* cited the gold
document); **correct-refusal** is the honesty signal on the no-answer questions; **p95** is
what users feel; **model/tool calls and tokens** are the cost. It is a 30-question smoke
test on a 3-document corpus — a difference of one or two questions is noise, and
"source-correct" does not catch a wrong claim that cites the right file. Spot-check by hand.

### Findings (write these AFTER pasting the table; delete any line the numbers do not support)

- **Quality:** on single-passage lookups, direct vs agent source-correct was ___ vs ___ (n = ___).
  On tool-shaped questions (calculation / metadata / multi-hop) the agent was ___ vs direct ___.
- **Honesty:** correct-refusal on the 8 no-answer questions: direct ___, structured ___, agent ___.
  (Week 5 found `n3` retrieves deceptively close; check whether any mode answered it.)
- **Cost:** average tokens per question: direct ___, structured ___ (___ avg attempts), agent ___
  (___ avg model calls). Agent / direct token ratio: ___x. Cost per *correct* answer matters more
  than cost per question.
- **Latency:** p95 direct ___ s, structured ___ s, agent ___ s.
- **Surprise:** one thing the numbers showed that the structural table above did not predict: ___.

## Decision rules

Use **direct RAG** when:
- the answer lives in one or two passages (policy lookups, FAQ, "what does the doc say about X");
- latency or cost is tight, or you need predictable behaviour and easy debugging;
- you cannot justify the security work an agent requires.

Use **structured output** (still direct) when a downstream system consumes the answer and
must trust its shape, or you need `answerable` / `cited_sources` as machine-readable fields.

Use the **agent** when the question genuinely needs tools, e.g.
- *"What is 25 days × 12 employees?"* — calculation, not retrieval;
- *"How many documents do we have, and which mention refunds?"* — metadata SQL **and** search;
- multi-hop questions where the second search depends on the first result.

**Do not** use an agent to "make RAG smarter" by default. If retrieval quality is the
problem, fix retrieval (chunking, top-k, hybrid search, reranking — see the Week 5 experiments);
an agent mostly hides a weak retriever behind more model calls.

## Recommendation for this project

Ship direct/structured RAG as the default path for document questions. Expose the agent
as an opt-in capability for tool-shaped questions, behind the same authentication,
tenant scoping and budgets (`max_steps`, `max_tool_calls`) as everything else, and watch
`budget_exhausted` and denied-call rates as health signals. If the measured table shows
the agent gaining nothing on plain lookups, route by question type rather than sending
everything through it.

## 60-second interview answer

> An agent is a loop where the model chooses tools. That buys flexibility but costs
> latency, tokens (the history is re-sent every step), reproducibility and security
> surface — a hijacked model can *act*, not just answer. So I default to direct RAG, which
> is one retrieval and one generation, and add an agent only for questions that need
> calculation, structured lookups or dependent searches. I measure it rather than assume:
> source-correct rate, correct refusals, p95 latency and calls/tokens per question, on the
> same question set for each mode.
