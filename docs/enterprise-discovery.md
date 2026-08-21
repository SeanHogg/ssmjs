# Enterprise technical discovery

A guide for running the first two technical sessions with a customer deploying Evermind. It exists because most enterprise AI engagements fail on decisions made — or quietly not made — before any integration code is written, and because a discovery session that produces a wish list rather than a set of *decisions with owners* has to be run again.

Every question below maps to a port in `@seanhogg/builderforce-memory`. That mapping is the point: the customer's answer selects an adapter, and an answer we cannot get is a **risk we name out loud** rather than an assumption we bury in a design doc.

---

## How to run the session

**Two sessions, not one.** Session 1 establishes what exists and what "good" means. Session 2 walks the proposed architecture back to them and closes the decisions. A single long session reliably produces agreement that dissolves on contact with their security team.

**Bring the right people or reschedule.** You need someone who owns the data, someone who owns the identity model, and someone who can approve spend. Missing the identity owner is the most common cause of a pilot that works and can never ship — tenancy and ACLs are not something to retrofit.

**Ask what breaks, not what they want.** "What would make you turn this off in week two?" surfaces more architecture than any feature list. The answer is almost always one of: it leaked something, it was wrong and nobody could tell, or it cost more than the thing it replaced.

**Write down the kill condition.** A pilot without a number that could fail it is a launch with extra steps. Get the number in session 1 and put it in the eval gate in week 1.

---

## 1. Data — what has to be readable

| Ask | Why it decides something | Maps to |
|---|---|---|
| What are the top three sources users will ask about? | Scopes the pilot to something provable in weeks. | `SourceDocument` set |
| For each: how much, what format, how often does it change? | Volume picks the store; change rate picks the sync strategy. | `IngestionPipeline`, `SyncStrategy` |
| Which sources are **structured** (tickets, CRM rows, tables)? | Typed columns must survive as filterable metadata, or precision questions are unanswerable. | `rowsParser`, `csvParser` |
| What identifiers do users type verbatim? (error codes, SKUs, clause numbers) | Confirms the lexical arm is load-bearing; dense-only will miss these. | `keywordSearch` |
| Is there a system of record we must not duplicate? | Decides whether text lives in the index or beside it. | `textResolver` |
| Who may see what, and where does that live today? | The ACL model must come from *their* groups, not a new one. | `AccessScope.principals` |
| Is there content that must be deleted on request? | Erasure must be a supported operation, not a migration. | `IngestionPipeline.forget()` |

**Red flag:** "we'll just index everything and figure out permissions later." Permissions retrofitted after indexing means re-indexing, and in the meantime every demo is a potential incident. Model tenancy and ACLs in week 1 — they are cheap then and structural later.

**Red flag:** a source nobody owns. If no one can say whether a document is current, retrieval quality is capped by a data-governance problem that no model will fix.

---

## 2. Retrieval — what "a good answer" means

| Ask | Why it decides something | Maps to |
|---|---|---|
| Show me five real questions and their ideal answers. | This *is* the eval dataset. Without it there is no launch gate. | `EvalDataset` |
| For each, which document should the answer come from? | Retrieval recall becomes measurable independently of generation. | `expectedSources` |
| Must answers cite their source? | Almost always yes in an enterprise. Decides the prompt and a grader. | `citationValidity` |
| What must an answer **never** say? | Turns compliance requirements into failing tests. | `mustNotContain` |
| What should happen when the answer is not in the corpus? | Refusal must be designed; the default of confident invention is what kills pilots. | `NO_EVIDENCE_ANSWER` |
| Is there an existing search system to beat? | Gives a baseline, which is the only honest framing of "is this better". | eval baseline run |

**The five questions are the deliverable of session 1.** If the customer cannot produce them, the project is not ready — and saying so early is worth more than a quarter spent discovering it.

---

## 3. Infrastructure — where it runs

| Ask | Why it decides something | Maps to |
|---|---|---|
| Which cloud, and can data leave it? | Decides bridge and store adapters, and whether a hosted model is even permitted. | `VertexAIBridge`, dialect choice |
| Do you already run a vector database? | Adopting theirs removes a procurement cycle. | `registerDialect()` |
| How does a workload get credentials? | Auth must be injected from their identity system, never stored by us. | `authorize`, `getAccessToken` |
| Where do traces and metrics go today? | Reuse their observability stack; a second dashboard nobody opens is not observability. | `OtlpHttpSpanExporter` |
| What is the data-residency requirement? | Constrains region, model availability, and sometimes the whole design. | region config |
| Is there a model allow-list? | Some regulated customers permit only specific models or providers. | `PriceBook`, bridge choice |

**Red flag:** "we'll give you a service-account key." Prefer workload identity. A long-lived key in a config file is a finding in their next audit and a conversation you will have anyway — better to have it now.

---

## 4. Agents — how much autonomy

| Ask | Why it decides something | Maps to |
|---|---|---|
| Does the system only answer, or does it act? | Acting changes the risk model entirely. | `createReactAgent` vs `EnterpriseRetriever` |
| Which actions need a human approval? | Approval is a graph interrupt, designed in, not a UI afterthought. | `interruptBefore` |
| What is the worst thing a wrong action could do? | Sets the tool surface and where the gates go. | tool design |
| How long may one task run? | Anything over a request timeout needs durable checkpoints. | `Checkpointer` |
| Who is accountable when an agent is wrong? | If nobody, the project has an ownership problem, not a technical one. | — |

**Start without autonomy.** Retrieval with citations proves value in weeks and carries a fraction of the risk. Add tools once the eval suite exists to catch regressions — an agent without a gate is a system whose quality nobody can measure until a user complains.

---

## 5. Economics — what it may cost

| Ask | Why it decides something | Maps to |
|---|---|---|
| What does this task cost today (person-minutes x loaded rate)? | The only credible ROI framing. | — |
| What is an acceptable cost per request? | Becomes a gate threshold, not a hope. | `maxMeanCostUsd` |
| What latency do users tolerate? | Streaming and TTFT design; p95 hides perceived latency. | TTFT metric |
| Expected query volume at steady state? | Cost model and cache sizing. | `MetricsRegistry` |
| How repetitive are the questions? | High repetition makes the semantic cache the single biggest lever. | `SemanticCache` |

Costs to name before they are discovered: **embedding the initial corpus** (one-off, can be large), **re-embedding on every full re-sync** (why the manifest exists), and **reranking**, which is per-query and easy to leave on by default.

---

## 6. What to leave the room with

A discovery session succeeded if all six exist and each has a name against it:

1. **Three sources**, with an owner and a change frequency for each.
2. **Five questions with ideal answers and expected sources** — the seed eval dataset.
3. **The tenancy and ACL model**, expressed in *their* group names.
4. **The store and cloud decision**, or a named blocker and who resolves it.
5. **A gate**: minimum pass rate, cost ceiling, latency ceiling.
6. **The kill condition** — what result in week 6 means stopping.

Anything unresolved is written down as a risk with an owner and a date. An open question with a name against it is a project; an open question without one is how a pilot quietly becomes a pilot forever.

---

## First-week architecture (the default proposal)

Absent a reason to deviate, this is what to propose. It is deliberately small: everything here is measurable within a fortnight.

```
Sources ──► IngestionPipeline ──► VectorStore (theirs, via a dialect)
              │  parsers               │  scoped: tenantId + acl + sensitivity
              │  chunk + hash          │
              └─ manifest → diff sync  ▼
                                  EnterpriseRetriever  ──► answer with citations
                                       dense + lexical → RRF → MMR
                                            │
                            Tracer ─────────┴──────────► their OTLP collector
                               └─► MetricsRegistry (tokens/sec, $/request, TTFT)
                                            │
                                       EvalHarness ──► gate in CI
```

Sequence it so value and evidence arrive together:

- **Week 1** — ingest one source; stand up the eval dataset from the five questions; wire the tracer to their collector. Retrieval quality is measurable before anyone tunes a prompt.
- **Week 2** — add the remaining sources and the lexical arm; set the gate thresholds from the first real numbers rather than from a guess.
- **Week 3** — grounded answers with citations, behind their access scope. This is the demo that ends the "will it leak" conversation.
- **Week 4+** — only now consider agents, and only for actions with a named owner and an approval gate.
