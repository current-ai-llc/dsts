# DSTS — Dynamic Self‑improving TypeScript

DSTS is a minimal, AI SDK–aligned prompt optimizer for TypeScript. It optimizes prompts for both generateObject (with Zod schemas) and generateText with the latest GEPA optimizer (soft G like “giraffe”). GEPA evolves a prompt along a Pareto frontier across multiple objectives that matter in practice: task performance, latency, and cost.

- AI Gateway–aligned: pass model ids as strings; generateObject with a Zod schema for a structured object or generateText with a string.
- Minimal abstractions: one default adapter to call `generateObject`/`generateText` and one optimizer.
- Multi‑objective first: correctness + latency (and cost) tracked per iteration; Pareto front and hyper‑volume (2D) reported.
- Persistence & budgets: checkpoint/resume, per‑call cost estimation (via tokenlens) and budget caps, seeded minibatching.

## Install

```bash
npm i @currentai/dsts zod
```

## Quick start

```ts
import { z } from "zod";
import { optimize, DefaultAdapterTask } from "@currentai/dsts";

// Define schema (generateObject)
const Item = z.object({ title: z.string(), url: z.string().url() });

// Training data: use schema ⇒ generateObject; provide expectedOutput and/or a scorer
const trainset: DefaultAdapterTask<z.infer<typeof Item>>[] = [
  {
    input: "link to TS docs",
    expectedOutput: {
      title: "TypeScript",
      url: "https://www.typescriptlang.org",
    },
    schema: Item,
  },
];

const result = await optimize({
  seedCandidate: { system: "Extract a title and a valid URL from the text." },
  trainset,
  // Optional valset (defaults to trainset)
  taskLM: "openai/gpt-5-nano",
  reflectionLM: "openai/o3",
  maxIterations: 5,
  maxMetricCalls: 200,
  maxBudgetUSD: 50,
  reflectionMinibatchSize: 3,
  candidateSelectionStrategy: "pareto",
  componentSelector: "round_robin",
  logger: {
    log: (lvl, msg, data) => {
      if (lvl === "info") console.log(`[${lvl}] ${msg}`, data || "");
    },
  },
  persistence: {
    dir: "runs/quickstart",
    checkpointEveryIterations: 1,
    resume: true,
  },
});

console.log("Best system prompt:", result.bestCandidate.system);
```

## Examples

- Email extraction to a rich object schema: [`examples/email-extraction.ts`](file:///home/swiecki/coding/dsts/examples/email-extraction.ts)
- Message spam classification: [`examples/message-spam.ts`](file:///home/swiecki/coding/dsts/examples/message-spam.ts)

Each example:

- Loads `.env` locally (AI Gateway by default),
- Prints total iterations, metric calls, cost (USD), and duration (ms),
- Enables persistence to `runs/...`.

Run:

```bash
npm run example              # email extraction
npm run example:message-spam # spam classification
```

## How it works

- Default adapter decides generateObject vs generateText based on `schema` presence in each task; collects per‑instance scores, latency_ms, and cost_usd (via tokenlens when usage is available).
- GEPA optimizer maintains a candidate archive, runs minibatch reflection, and accepts improving children. It computes per‑candidate metrics:
  - correctness = average(score[])
  - latency = −avg(latency_ms) (stored negative so higher is better)
  - cost is tracked cumulatively and enforced via `maxBudgetUSD`.
- Pareto front and 2D hyper‑volume (when exactly two objectives) are logged per iteration and at the end.

Key files:

- Optimizer: [`src/gepa.ts`](file:///home/swiecki/coding/dsts/src/gepa.ts#L1-L495)
- Adapter: [`src/adapters/default-adapter.ts`](file:///home/swiecki/coding/dsts/src/adapters/default-adapter.ts#L1-L267) (default maxConcurrency = 10)
- Pareto utilities: [`src/pareto-utils.ts`](file:///home/swiecki/coding/dsts/src/pareto-utils.ts#L1-L241)
- Types: [`src/types.ts`](file:///home/swiecki/coding/dsts/src/types.ts#L1-L158)
- Persistence: [`src/persistence.ts`](file:///home/swiecki/coding/dsts/src/persistence.ts#L1-L63)

## Design choices

- No custom LLM classes: pass model ids as strings (Gateway format, e.g., `openai/gpt-5-nano`). The adapter uses AI SDK directly.
- Minimal knobs: set budgets (`maxMetricCalls`, `maxBudgetUSD`), minibatch size, and selectors. Concurrency defaults to 10.
- Multi‑objective by default: we optimise for correctness and latency together; add cost as an explicit objective later if desired.

## Environment

- AI Gateway by default. Set `AI_GATEWAY_API_KEY` in `.env` or export it in your shell.
- If you prefer provider‑direct, swap to `@ai-sdk/openai` models and pass model objects; the adapter will forward them.

## Roadmap

- Centralized eval helper for exact metric‑call counting and pre‑call budget gates.
- Parent minibatch result reuse to avoid duplicate evaluations.
- Extend hyper‑volume and objectives (e.g., cost as a third dimension) with explicit reference points.
- Reflection concurrency (optional) and parent/child evaluation parallelism.
