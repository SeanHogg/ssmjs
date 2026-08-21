# BuilderForce Agent Memory

A framework for giving **any AI agent** persistent, browser-trainable SSM memory. It is provider-neutral — swappable LLM bridges (Anthropic / OpenAI / Fetch), an inference router, online distillation, and a persistent memory store — so any agent or app can depend on it, not just BuilderForce.

**[BuilderForce.ai](https://builderforce.ai) is the flagship deployment**, not the boundary: it trains a custom SSM in the browser, exports a model, and pushes it to an agent runtime where it runs as **Evermind — the model itself**, not just memory bolted onto someone else's LLM. Evermind is the whole brain: its own shared-expert generator (the *cortex* that does reasoning and language), a write-through knowledge memory (the *hippocampus*), and a trainable affective layer (the *limbic* system). A request can be served by Evermind directly — `evermind/<ref>` traffic is generated on-device, not forwarded to Claude/GPT — and external frontier models stay an *optional* routing choice rather than a hard dependency. The same framework is reusable by other agents off the shelf.

This monorepo consolidates two packages that previously lived in separate repos (`mambacode.js` and `ssmjs`) whose names described the *technique* rather than the *role*. They are renamed and unified here so that "what they do" is legible: they are Agent Memory.

## Packages

| Package | Layer | Was | Responsibility |
|---|---|---|---|
| [`@seanhogg/builderforce-memory-engine`](packages/memory-engine) | **Engine** | `@seanhogg/mambacode.js` | WGSL/WebGPU Mamba SSM kernels, model blocks (Mamba1/2/3 + attention), autograd, trainer, BPE tokenizer, quantization. Zero runtime deps. |
| [`@seanhogg/builderforce-memory`](packages/memory) | **Runtime** | `@seanhogg/ssmjs` | SSM execution, Transformer orchestration (Anthropic/OpenAI/Fetch bridges), online distillation, inference router, sessions, the persistent `MemoryStore`, and **Write-Through Cognition** (`EvermindCognition`). Depends on `memory-engine`. |

The two-package split is deliberate: the engine is zero-dep and WebGPU-pure and can be consumed standalone; the runtime pulls in LLM-vendor bridges. Flattening them would force engine-only consumers to drag in vendor code and vice versa. They release in lockstep from one pipeline, which kills the publish-drift bug class that the separate-repo setup suffered (bumping one version without publishing + regenerating the consumer lockfile).

## Cutting token cost

Agent Memory ships three layers that reduce LLM spend, in increasing power. All are **portable** — the same code runs in the browser (WebGPU SSM) and in Node (the agent's `@webgpu/node` SSM) because the embedder and storage are injected, never hard-wired.

| Layer | What it does | Saves |
|---|---|---|
| **Prompt caching** ([`AnthropicBridge`](packages/memory/src/bridges/AnthropicBridge.ts) `cacheSystem`) | Marks the stable system prompt as an Anthropic `cache_control` block | ~90% on the cached **input** prefix (cost, not count) |
| **Exact-match cache** ([`CachingBridge`](packages/memory/src/bridges/CachingBridge.ts) + [`ResponseCache`](packages/memory/src/bridges/ResponseCache.ts)) | Reuses byte-identical completions | Eliminates duplicate calls (retries, identical fan-out) |
| **Semantic cache** ([`SemanticCache`](packages/memory/src/cache/SemanticCache.ts) + [`SemanticCachingBridge`](packages/memory/src/bridges/SemanticCachingBridge.ts)) | Reuses a prior answer when the new prompt is within a cosine **threshold** of one already answered — catches paraphrases | Avoids frontier calls entirely on semantically-repeated prompts |

The semantic cache is the real lever. It is **read-through with two tiers**, mirroring an L1-Map / L2-KV cache:

- **L1** — an in-process vector list, scanned locally with on-device SSM embeddings (free, offline-capable).
- **L2** — an optional shared backend ([`FetchSemanticCacheBackend`](packages/memory/src/cache/FetchSemanticCacheBackend.ts) → the BuilderForce.ai gateway), so a paraphrase answered by the **web app** is reusable by an **agent** and vice-versa.

```ts
import { SemanticCache, FetchSemanticCacheBackend, AnthropicBridge } from '@seanhogg/builderforce-memory';

const cache = new SemanticCache({
  embed: (t) => runtime.embed(t),                                   // on-device SSM (free)
  l2: new FetchSemanticCacheBackend({ baseUrl, apiKey }),           // shared via the gateway
  threshold: 0.92,
});

const { response, cached, tier } = await cache.getOrGenerate(
  prompt,
  () => bridge.generate(prompt),                                    // only runs on a miss
);
```

Memory-backed fact injection is semantic too: [`SSMAgent`](packages/memory/src/agent/SSMAgent.ts) defaults to `factSelection: 'semantic'`, injecting only the top-`maxFacts` embedding-relevant facts each turn (paraphrase-robust, smaller prompts) instead of an exact key-substring match.

## Write-Through Cognition (Evermind)

Caching keeps *answers* fresh; **cognition keeps *knowledge* fresh.** A frozen model — or an append-only memory — drifts: stale and current facts pile up under different keys until something reconciles them by hand. [`EvermindCognition`](packages/memory/src/cognition/EvermindCognition.ts) closes that gap. It is the model-knowledge analogue of a write-through cache with a conflict resolver, so a belief is **replaced on write**, never appended into a reconciliation backlog. This is what lets Evermind's knowledge stay current without a manual reconcile step.

Every candidate fact flows through one pipeline:

> Canonicalize (stable subject key) → recall incumbent → evaluate evidence → reconcile (**augment | confirm | supersede | reject**) → write-through

- **Stable subject key** — facts about the same subject collide and replace, instead of accumulating under per-run ids. This is the anti-drift fix.
- **Evidence-gated** — a *conflicting* fact only supersedes the incumbent when ground-truth evidence favours it. The [`EvidenceGatherer`](packages/memory/src/cognition/types.ts) is injected, so the evidence source is surface-specific (IDE file tools, a DB probe, an HTTP check); `workspacePresenceGatherer` ships for the common "is it still on disk?" case.
- **Write-through recall** — `recall()` is served from a version-token cache that invalidates the instant knowledge changes — the same invalidate-on-write rule as the semantic cache, applied to beliefs.

```ts
import { EvermindCognition, workspacePresenceGatherer, MemoryStore } from '@seanhogg/builderforce-memory';

const cog = new EvermindCognition({ store: new MemoryStore() });

// Seed a (soon-to-be) stale belief.
await cog.commit({ subjectKey: 'pkg:ssm-stack', content: 'SSM stack = MambaKit + SSMjs' });

// Re-observe: evidence from the real workspace decides the conflict.
const r = await cog.commit(
  { subjectKey: 'pkg:ssm-stack', content: 'SSM stack = builderforce-memory monorepo' },
  workspacePresenceGatherer({
    list: () => listWorkspace(),                  // e.g. the IDE `list_files` control tool
    mustExist:   ['builderforce-memory/'],
    mustBeAbsent: ['MambaKit/', 'SSMjs/'],
  }),
);
// r.verdict === 'supersede' → exactly one belief held, the stale one retired (replace, not append)
```

`EvermindCognition` is store- and surface-agnostic — the `CognitionFactStore` interface is satisfied structurally by `MemoryStore` (no adapter) — so the same loop runs in the IDE, on-prem, cloud, and the browser.

## Enterprise architecture

Caching cuts the bill and cognition keeps knowledge current, but neither answers the questions an enterprise buyer actually asks before signing: *can it read our data, will it leak across tenants, what did that answer cost, and how do we know it got better?* Those four questions are what this layer exists to answer. Each is a port with adapters behind it, so a customer's existing stack is an adapter choice rather than a rewrite.

| Layer | Entry point | Subpath export | Answers |
|---|---|---|---|
| **Ingestion** | `IngestionPipeline` | `@seanhogg/builderforce-memory/ingest` | Can it read our data — structured *and* unstructured — and stay current? |
| **Vector store** | `VectorStore` port + adapters | `…/vectorstore` | Can it run against the database we already have, with our tenancy rules? |
| **Retrieval** | `EnterpriseRetriever` | `…/rag` | Are the answers grounded, scoped, and citable? |
| **Orchestration** | `AgentGraph` + patterns | `…/orchestration` | Can agents coordinate, pause for a human, and resume after a crash? |
| **Telemetry** | `Tracer` + `MetricsRegistry` | `…/telemetry` | What did it cost, how fast was it, and where did it go wrong? |
| **Evaluation** | `EvalHarness` | `…/eval` | How do we know it is good enough to launch — and still is? |

### Ingestion — structured and unstructured through one path

A support-ticket export (rows, typed columns) and a policy document (prose) reach the index through the same pipeline; the difference is which parser ran, not which system was built. Structure that survives parsing becomes **filterable metadata**, which is what makes *"summarise open P1 tickets about billing"* answerable — `status` and `priority` are database predicates rather than something the embedding has to imply.

```ts
import { IngestionPipeline, InMemoryIngestManifest } from '@seanhogg/builderforce-memory/ingest';
import { MemoryVectorStore } from '@seanhogg/builderforce-memory/vectorstore';

const pipeline = new IngestionPipeline({
  store   : new MemoryVectorStore(),
  embed   : (texts) => embedder.embedBatch(texts),
  manifest: new InMemoryIngestManifest(),   // enables `diff` sync
  tracer,
});

await pipeline.ingest([
  { id: 'policy.md', tenantId: 'acme', title: 'Retention Policy',
    content: { kind: 'text', text: markdown, mediaType: 'text/markdown' } },
  { id: 'tickets',   tenantId: 'acme', acl: ['support'],
    content: { kind: 'rows', rows: ticketRows, rowIdField: 'id' } },
]);
```

What makes it a pipeline rather than a loader script is everything that happens on the **second** run: chunk ids are deterministic (a re-run overwrites instead of duplicating), content hashes mean only changed chunks are re-embedded, vanished chunks are deleted, one malformed document fails alone and is reported by id, and `forget(sourceId)` erases a source completely. Builtin parsers cover markdown (keeping the full heading breadcrumb), HTML, JSON, CSV/TSV and typed rows; a new format is a `ParserRegistry` entry, not a branch.

### Vector store — a port, not a vendor

Enterprises rarely get to choose the vector database; it is already pgvector in their Cloud SQL, Vertex AI Vector Search because they are a GCP shop, or Qdrant because platform standardised on it. So the store is a port and every adapter is infrastructure:

- **`MemoryVectorStore`** — in-process, HNSW-backed, with BM25 keyword search. The default, the test double, and the honest answer for edge deployments.
- **`RestVectorStore`** — one adapter over a **dialect registry**. `evermind` (the contract this package defines), `qdrant`, `pinecone` and `vertex-ai` ship; `registerDialect()` adds your own.

Filters are **data**, not predicates, because a `(record) => boolean` cannot be pushed down to a remote database — it forces fetch-everything-then-filter, which is both slow and a leak. When a dialect cannot express a clause, `translateFilter` returns it as a **residual** that the store applies locally after over-fetching. A clause is never silently dropped, which on an ACL clause would be a cross-tenant leak.

**Tenancy is compiled, not remembered.** An `AccessScope` becomes a filter that is AND-ed into every read, and it fails *closed*: ingestion always writes a non-empty `acl` (defaulting to `['*']`), so a record written without one matches nothing rather than everything.

```ts
const hits = await store.query({
  vector, topK: 5,
  scope: { tenantId: 'acme', principals: ['sec-team'], maxSensitivity: 3 },
});
```

### Retrieval — hybrid, scoped, cited

`EnterpriseRetriever` runs dense and lexical arms concurrently, fuses them with RRF, reranks with MMR for diversity, and optionally hands the result to a reranker. Dense retrieval alone misses exact tokens — error codes, SKUs, clause numbers; lexical alone misses paraphrase.

It is also honest about degradation: against a store with no lexical index it reports `mode: 'dense-only'` rather than quietly returning worse answers. And with no passages in scope it **refuses**, because a confident answer with no evidence is the exact failure that kills pilots.

```ts
const { answer, result } = await retriever.answer(question, bridge, {
  scope: { tenantId: 'acme', principals: ['support'] },
});
// answer cites passages as [1], [2] …; result.passages carries title + uri for each
```

### Orchestration — multi-agent as a stateful graph

`AgentGraph` executes in **supersteps**: a frontier of nodes runs concurrently, their partial updates merge through declared channel reducers, and the next frontier comes from the edges. The three hard parts of multi-agent work fall out of that model — concurrent fan-out is just a bigger frontier, a merge conflict has a declared answer instead of a race, and the state between supersteps fully describes the run, so a checkpoint written there resumes it exactly.

That last property is why crash recovery, human-in-the-loop approval and time-travel debugging are one mechanism here rather than three features:

```ts
const graph = new AgentGraph({ interruptBefore: ['publish'], checkpointer, tracer });
const paused = await graph.invoke(input, { threadId });   // reason: 'interrupted'
await graph.resume(threadId, { draft: humanEditedDraft }); // edits merge through the same reducers
```

Three patterns ship on top, and they compose — a supervisor's worker can be a ReAct agent:

- **`createReactAgent`** — interleaved reasoning and tool use. The iteration cap is enforced in the routing rather than asked for in the prompt, and a malformed action or a failing tool comes back as an *observation* the model can recover from, never an exception that ends the run.
- **`createReflectionAgent`** — generate → critique → revise, bounded. Reflection pays only when the critic may be harsh *and* the loop may stop; both limits are code, not prompt.
- **`createSupervisor`** — hierarchical delegation. Workers return a **result**, not their transcript, which keeps the supervisor's context from growing into the sum of everything its workers read.

### Telemetry — LLM-native metrics on one span stream

Metrics are a *projection* of the trace stream rather than a second instrumentation path that can drift out of agreement with it. Attach `MetricsRegistry` as a `Tracer` exporter and tokens/sec, cost-per-request, TTFT and cache-hit rate fall out of traffic that is already traced.

```ts
const registry = new MetricsRegistry();
const tracer   = new Tracer({ exporter: registry });
const bridge   = new InstrumentedBridge(new AnthropicBridge({ apiKey }), { tracer });

registry.llm().tokensPerSecond;         // throughput the user feels
registry.llm().avgCostPerRequestUsd;    // the unit that ties spend to a decision
registry.llm().localCacheHitRate;       // the lever that moves cost
registry.llm().unpricedCalls;           // honesty: never billed at a silent zero
```

Every bridge reports what its last call actually consumed, so cost is **measured, not estimated** — and Anthropic's three-way input split (fresh / cache read / cache write) is priced at three different rates, because folding them together overstates spend by ~10× on exactly the cache-heavy traffic this package is tuned to produce. Spans export to any OTLP/HTTP collector (Cloud Trace, Datadog, Honeycomb, Tempo) using GenAI semantic conventions, over `fetch` rather than the OpenTelemetry SDK, so the package still runs in a browser and in a Worker.

### Evaluation — the launch gate

Projects die either because nobody can say whether the system is good enough, or because it regressed silently after launch. Both are the same missing artefact: a dataset, comparable scores, and a threshold a build can fail on.

```ts
const report = await new EvalHarness({
  graders: [noError, retrievalRecall, citationValidity, substringChecks],
  gate   : { minPassRate: 0.9, minGraderScore: { 'retrieval-recall': 0.85 }, maxMeanCostUsd: 0.01 },
  tracer,
}).run(dataset, target);

console.log(formatReport(report));
if (!report.gate.passed) process.exit(1);
```

Two graders matter most and generic LLM evals omit both. **`retrievalRecall`** asks whether the right source came back at all — a generation score cannot distinguish "reasoned badly" from "was handed nothing", and those have opposite fixes. **`citationValidity`** catches an answer citing `[7]` when six passages were supplied; that scores fine on similarity and fails an audit. Cost and latency are graded alongside quality, because a quality-only harness quietly ships an unaffordable system. A gate with no thresholds reports itself as **vacuous** rather than green.

### Google Cloud

`VertexAIBridge` and `VertexAIEmbedder` serve GCP deployments, and the `vertex-ai` dialect maps this filter algebra onto Vertex `restricts` / `numericRestricts`. Auth is **injected, never owned** — supply `getAccessToken` and the same code runs under Application Default Credentials locally, Workload Identity on GKE, and an impersonated token in CI, without this package taking on a credential lifecycle or a Node-only vendor SDK.

Vertex AI Vector Search stores vectors and restricts but **not text**, so its dialect declares `storesText: false` and the store hydrates chunk text through a `textResolver` (Firestore, BigQuery or GCS beside the index). Pretending otherwise would return matches with empty text and an answer with no evidence.

> **Validation status.** The `qdrant`, `pinecone` and `vertex-ai` dialects and the Vertex bridge are covered by unit tests that assert the emitted request bodies and parse recorded response shapes. They have **not** been exercised against live services from this repository, which has no credentials for them — see the Gap Register.

### Discovery

[`docs/enterprise-discovery.md`](docs/enterprise-discovery.md) is the technical discovery guide: the questions to ask a customer, which answer maps to which port, and the decisions that must be made before a line of integration code is written.

## Technical report & peer review

The architecture is written up as a formal, peer-review-grade technical report, with the SSM cortex, Write-Through Cognition, and the limbic layer specified mathematically (including a monoid-scan proof and the single-incumbent invariant). It lives in [`publication/evermind/`](publication/evermind):

- **[`evermind-architecture.pdf`](publication/evermind/evermind-architecture.pdf)** (source: [`.tex`](publication/evermind/evermind-architecture.tex), readable [`.html`](publication/evermind/evermind-architecture.html)) — the manuscript. It deliberately scopes the "beats a frozen frontier model" advantages as **falsifiable hypotheses with a measurement protocol**, not as benchmark results. The language-model **benchmarking harness** those metrics need (held-out perplexity, bits-per-token, accuracy, throughput, model A/B) now ships in the engine (`packages/memory-engine/src/bench/`) and on-device in the Studio, so the protocol is an instrument away from numbers, not a skeleton.
- **[`PEER-REVIEW.md`](publication/evermind/PEER-REVIEW.md)** — an adversarial scientific referee report on both the paper *and this implementation*, grounded in `file:line` evidence. It is candid about where the shipped code is first-generation: notably that recall is an `O(N)` scan with no ANN index, that the "stable subject key" above is **caller-supplied and not yet canonicalized** (so the single-incumbent guarantee currently rests on an unenforced precondition), and that the MCP HTTP surface lacks tenant isolation. These findings are tracked as `EVM-1…EVM-8` in the Builderforce.ai roadmap's Consolidated Gap Register. Those Major items have since been resolved in the v2026.6.34 hardening (HNSW ANN index, a subject-key canonicalizer inside `commit()`, MCP tenant isolation, fenced recall, a forgetting guard); see the resolution addendum at the top of that review.
- **[`SUBMISSION.md`](publication/evermind/SUBMISSION.md)** — venue guide and the TechRxiv/arXiv submission kit.

Read the review before depending on any single guarantee in the report at production scale.

## Develop

```bash
pnpm install          # links workspace packages; memory-engine auto-builds via prepare
pnpm build            # builds memory-engine, then memory (order matters — runtime needs core's .d.ts)
pnpm test             # full suite across engine (jest) + runtime (jest, incl. cognition) + mcp (node:test)
pnpm lint
```

`@seanhogg/builderforce-memory` consumes `@seanhogg/builderforce-memory-engine` via the pnpm `workspace:^` protocol; on publish, pnpm rewrites it to a real `^<version>` range.

## Release

All three packages (`memory-engine`, `memory`, `memory-mcp`) version in lockstep (`YYYY.M.D[-beta.N]`).

1. Bump `version` in every `packages/*/package.json` (and root) to the same value, plus the `@seanhogg/builderforce-memory` range in `memory-mcp`'s `peerDependencies`.
2. `pnpm release:check` (build + test).
3. Commit, then `git tag vYYYY.M.D`.
4. Push the tag — `.github/workflows/release.yml` runs `pnpm -r publish`, which publishes `memory-engine`, `memory`, and `memory-mcp` to npm with provenance (dependency order is resolved automatically; `workspace:` specifiers are rewritten to real ranges at pack time).

## Migration / redirect plan for the old packages

The old packages (`@seanhogg/mambacode.js`, `@seanhogg/ssmjs`) are superseded. To migrate consumers cleanly:

1. **Publish the new scope** from this repo (`@seanhogg/builderforce-memory-engine`, `@seanhogg/builderforce-memory`).
2. **Deprecate the old names** pointing at the new scope:
   ```bash
   npm deprecate "@seanhogg/mambacode.js" "Moved to @seanhogg/builderforce-memory-engine"
   npm deprecate "@seanhogg/ssmjs"        "Moved to @seanhogg/builderforce-memory"
   ```
3. **Update consumers' imports**:
   - `@seanhogg/mambacode.js` → `@seanhogg/builderforce-memory-engine`
   - `@seanhogg/ssmjs`        → `@seanhogg/builderforce-memory`
   The primary in-repo consumer to update is the BuilderForce.ai agent-runtime (which loads the SSM as Evermind — generation plus write-through memory).
4. The old `SeanHogg/mambacodejs` and `SeanHogg/SSMjs` repos can be archived once consumers are cut over. Full git history from both was preserved into this repo via subtree merge — no history was lost.

## Consolidated Gap Register

Roadmap items identified during the consolidation but intentionally not closed in this pass:

- **[DONE 2026-06-18] Published under `@seanhogg/builderforce-memory*`; old names deprecated; mistakes cleaned up.** Final published names: `@seanhogg/builderforce-memory-engine` / `@seanhogg/builderforce-memory` / `@seanhogg/builderforce-memory-mcp` at `2026.6.19` (the `@builderforce` org doesn't exist on npm, so the scope is `@seanhogg` but the `builderforce-` name prefix is kept). Earlier wrong-name publishes are gone: `@builderforce/*` never landed (404), and the mistaken `@seanhogg/memory{,-engine,-mcp}@2026.6.18` were unpublished. Superseded legacy packages deprecated on npm via the `deprecate-old-packages.yml` workflow: `@seanhogg/ssmjs` → `@seanhogg/builderforce-memory`, `@seanhogg/mambakit` → `@seanhogg/builderforce-memory` (`@seanhogg/mambacode.js` was never on npm — engine was git-only).
- **[DONE 2026-06-18] BuilderForce.ai consumers repointed to the published packages.** frontend now depends on `@seanhogg/builderforce-memory-engine` (the `mambacode.js` git dep + `global.d.ts` shim removed; `tsc` clean, tests green); agent-runtime's SSM service imports `@seanhogg/builderforce-memory`. Open: agent-runtime can't add the `optionalDependencies` entry until `2026.6.19` is >48h old (its `pnpm.minimumReleaseAge: 2880` guard); add it after 2026-06-20, plus wire the frontend browser `SemanticCache` (needs the runtime dep + WebGPU bundle verify).
- **[DONE 2026-06-18] Multi-host installer — memory now extends to any MCP-capable agent, not just Claude.** New `builderforce-memory-install` bin + `src/install/` module (host registry + shared launch spec) register the stdio server into Claude Code, Claude Desktop, Cursor, Windsurf, VS Code, Cline, Gemini CLI, and Codex CLI, idempotently with `.bak` backups, all pointing at one shared `~/.builderforce-memory/memory.json` store. Auto-detects installed hosts (`--host=auto`), or `--host=all|<ids>`. 7 unit tests over an in-memory fs (`tests/install.test.mjs`). `buildServerSpec`/`installMemoryServer` exported for reuse.
- **builderforce.ai's own agents — memory wiring status.** (1) **On-prem agent hosts** run on a machine with an MCP host → covered by the multi-host installer above. (2) **[DONE 2026-06-18] Cloud Node/container agents** now load memory in-process via `ssm-memory-service.ts` — `@seanhogg/builderforce-memory`(+`-engine`) added to agent-runtime `optionalDependencies`; the 48h `minimumReleaseAge` cooldown was bypassed for first-party packages only via `minimumReleaseAgeExclude` (third-party guard intact). Verified: recall/remember loads in the agent-runtime env; `server-startup` boots `initSsmMemoryService`, the orchestrator injects `recallSimilar` into task prompts, and `KnowledgeLoopService` remembers/learns per run. (3) **[Node DONE 2026-06-18] Active recall/remember TOOLS** — instead of the MCP-into-custom-loop path, the `memory` capability in `@builderforce/agent-tools` was completed with `memory_recall`/`memory_remember` tool definitions, backed on-prem by the SSM service; on-prem + cloud-container (Node) agents now actively recall/remember mid-run. The server-side `getCachedOrGenerate` semantic cache was also wired into the cortex call. (4) **[DONE 2026-06-18] Cloud Worker/DO (V2 durable) active memory** — backed by Postgres `agent_memory` (migration 0200, tenant-scoped key→fact, lexical ILIKE recall, read-through cached) wired into the api's `buildCloudProvider` + `'memory'` in `CLOUD_SURFACE_CAPS`; graceful-degrades pre-migration, activates on deploy. (5) **[DONE 2026-06-20] Web (browser) `SemanticCache` (L1)** — `frontend/src/lib/semantic-cache.ts` fronts the cloud-inference path with a read-through cache backed by an on-device WebGPU SSM embedder (`SSM.create`→`runtime.embed`), dynamic-imported and gracefully `null` without WebGPU/assets. tsc + 4 vitest cases green; browser smoke + L2 endpoint wiring are the residuals. Every builderforce memory surface is now wired (on-prem, cloud container, cloud Worker/DO, web).
- **Lint not enforced in CI and likely dirty.** CI runs build + test only. The two packages had divergent ESLint setups (runtime had `@typescript-eslint`, engine did not) now unified into one root config; `pnpm lint` has not been run to green and is absent from the CI gate. Closing this makes lint a real signal instead of decoration.
- **`tsconfig` not wired for project references / incremental.** Root `tsconfig.json` lists references but the packages aren't `composite: true`, so `tsc -b` incremental builds and editor cross-package go-to-def rely on the built `dist` rather than source. Making the packages composite would speed builds and tighten editor UX.
- **README claims unverified at runtime.** The "browser-trained SSM → exported model → pushed to agent as Evermind (the model itself, generation + write-through memory)" loop is documented as the product strategy but there is no end-to-end test in this repo exercising export→load→generate/recall across the two packages. An integration test would convert the narrative into a guarantee.
- **Local working-copy folder still named `SSMjs`.** This monorepo lives in the former `SSMjs` checkout (its native git history is the base); the on-disk rename to `builderforce-memory` is blocked by VS Code holding a handle on the folder (the open `vscode.git.Git.log` tab) — ruled out tooling shells, the IDE is the locker. Cosmetic only — the GitHub repo, local remote URL, repo identity, and `package.json` name are all already `builderforce-memory`. Close the git-log tab/window, then run `Rename-Item C:\code\agentic\SSMjs builderforce-memory`.
- **`mambacodejs` retired.** Local checkout removed and `SeanHogg/mambacodejs` archived on GitHub (history preserved here via subtree merge into `packages/memory-engine`). Its archive description points to `SeanHogg/builderforce-memory`, which now exists (rename landed) — the pointer resolves.
- **Coverage excludes the WebGPU session layer (`src/session/**`) and barrels (`src/**/index.ts`).** `pnpm test:coverage` enforces a 100% global threshold, but `collectCoverageFrom` skips `src/session/**` — `MambaSession`, the WGSL/WebGPU `complete`/`completeStream`/`evaluate`/`embed` orchestration, `streaming.ts` sampling, `presets.ts`, and IDB weight `persistence.ts`. These bind to `@seanhogg/builderforce-memory-engine` + a GPU and can't be unit-tested without a WebGPU/`MambaModel` mock harness; the runtime/consumption surface that wraps them is at 100%. Closing this = a GPU-or-mock integration harness that drives `MambaSession` end-to-end (also closes the unverified export→load→recall loop noted above). Until then the engine boundary is covered only by the build + the consumers' own tests.
- **Fact injection still uses exact-substring match, not semantic recall.** `SSMAgent._buildPrompt` selects facts via `input.includes(f.key)` (`packages/memory/src/agent/SSMAgent.ts`). The store already exposes `recallSimilar()` (SSM-embedding top-K), which would inject fewer, more-relevant facts and shrink the escalated prompt further. Switching injection to top-K similarity (capped to N facts) closes this — deferred from the token-cost pass that added system-prefix caching.
- **`@seanhogg/builderforce-memory-mcp` zod peer skew (v3 vs v4).** The new MCP package pins `zod@^3.23`, but `@anthropic-ai/claude-agent-sdk@0.3.163` requires `zod@^4` (peer warning at install). The SDK transport (`src/transports/sdk.ts`) dynamically imports the SDK and hands it zod-v3 raw shapes; basic shapes are cross-compatible so it compiles and runs, but this is unverified against v4-only schema behaviour and a consumer installing the Agent SDK inherits the mismatch. Closing = bump `memory-mcp` to `zod@^4`, re-verify the MCP SDK `registerTool` + tool shapes still type-check, then drop this note.
- **`memory-mcp` headless bins fall back to lexical (Jaccard) recall.** The `builderforce-memory-mcp` stdio/HTTP bins don't load the SSM checkpoint/GPU, so `recall` uses Jaccard word-overlap. Smoke-verified mis-rank: query "what language does the user like?" ranked a `project.*` entry above `user.preferred-language`. Closing = let the bins optionally load an SSM runtime (or an embedding provider) and pass it to `createLocalMemoryStoreBackend({ runtime })`; until then, production recall should use the in-process transport with the agent-runtime's hippocampus, or accept lexical recall. Unblocks trustworthy headless recall.
- **`memory-mcp` has no automated tests / not in CI.** Verified by an ad-hoc smoke script (since removed); there is no jest suite for `buildMemoryTools` caps (top-K / truncation / read-only gating) and no MCP stdio round-trip integration test, and the package has no `test` script so `pnpm -r test` skips it. Closing = add unit + stdio-handshake tests and wire into the CI gate; converts the token-saving caps from "intended" to "guaranteed".
- **`memory-mcp` HTTP transport is stateless and tenant-blind.** `createMemoryHttpHandler` builds one `McpServer` per request against a single fixed backend with only bearer-token gating — no per-tenant backend routing, no rate limiting, and no read-through cache. The multi-tenant story (builderforce.ai hosting) needs a tenant→backend resolver and `getOrSetCached` on recall. Closing = add tenant resolution + caching when the remote builderforce.ai memory store lands (see next item).
- **No remote builderforce.ai memory backend yet.** `MemoryBackend` is the seam for "coordination flows through builderforce.ai", but only the local `MemoryStoreBackend` (per-process IndexedDB) ships. Cross-product/cross-machine shared memory needs a networked adapter implementing `recall/get/recallByTag/remember/forget` against builderforce.ai. Closing = build the remote read/write API + a `RemoteMemoryBackend` adapter; the tools and all three transports already consume it unchanged.
- **agent-runtime should consume `@seanhogg/builderforce-memory-mcp` instead of hand-rolling memory exposure.** `Builderforce.ai/agent-runtime/src/infra/ssm-memory-service.ts` wraps `MemoryStore` with its own `remember`/`recallSimilar` and would separately need to surface those to its own Claude Agent SDK loop. The in-process SDK transport (`createMemoryMcpServer`, passing `ssmMemoryService.runtime` as the embedding backend) is the canonical seam for that. Closing = refactor agent-runtime to depend on this package, removing the parallel wrapper — a DRY consolidation, deferred as a cross-repo change out of scope for shipping the package.
- **Facts share the cached system block, so any fact write busts the prefix.** Prompt caching now marks `System + Facts` as one ephemeral block. A fact update invalidates the whole cached prefix on the next turn. Acceptable (facts change slowly), but splitting the system prompt and the fact block into two `cache_control` breakpoints would let the static system prompt stay cached across fact writes. Deferred.
- **OpenAI/Fetch bridges rely on provider-side auto-caching, not an explicit marker.** Only `AnthropicBridge` emits `cache_control`; OpenAI-compatible endpoints cache prefixes automatically when the stable system message comes first (it now does, via the runtime `system` split). No marker API exists to assert it — undocumented and unverifiable from this side. Note in bridge docs.
