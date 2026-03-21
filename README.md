# SSM.js

> **JavaScript-native AI runtime** — SSM execution + Transformer orchestration + online distillation + persistent agent memory.

[![npm](https://img.shields.io/npm/v/ssmjs)](https://www.npmjs.com/package/ssmjs)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

SSM.js sits as an orchestration layer on top of [MambaKit](https://www.npmjs.com/package/mambakit), extending it into a complete AI runtime.

---

## Layer Stack

```
MambaCode.js  →  WebGPU kernels (WGSL, Mamba-1/2/3 SSM math)
MambaKit      →  Model / session abstraction (MambaSession.create())
SSM.js        →  Runtime orchestration (this package)
```

---

## What's New Over MambaKit

| Capability                   | MambaKit | SSM.js |
|------------------------------|----------|--------|
| Simple session API           | ✅       | ✅     |
| WebGPU execution             | ✅       | ✅     |
| SSM variants (1/2/3/hybrid)  | ✅       | ✅     |
| Transformer bridge           | ❌       | ✅     |
| Intelligent routing          | ❌       | ✅     |
| Online distillation          | ❌       | ✅     |
| Persistent semantic memory   | ⚠️       | ✅     |
| Agent workflows              | ❌       | ✅     |

---

## Quick Start

```ts
import { SSM, AnthropicBridge, SSMAgent, MemoryStore } from 'ssmjs';

const ai = await SSM.create({
  session: { modelSize: 'small', mambaVersion: 'mamba2' },
  bridge : new AnthropicBridge({ apiKey: 'sk-ant-...' }),
});

// Generate — routes to SSM or transformer automatically
const answer = await ai.generate('What is a state space model?');

// Streaming — always SSM for low-latency output
for await (const token of ai.stream('function fibonacci(')) {
  process.stdout.write(token);
}

// Fine-tune on your content (browser, no data leaves)
await ai.adapt(myCodebase);
```

---

## Node.js / Server-Side Usage

SSM.js and MambaKit are browser-first, but can run natively in Node.js by injecting a WebGPU adapter and an IndexedDB factory. This is how [CoderClaw](https://coderclaw.ai) loads user-trained `.bin` models as a local hippocampus.

**Install the Node.js shims:**

```bash
npm install @webgpu/node fake-indexeddb
```

**Usage:**

```ts
import { create as createGPU } from '@webgpu/node';
import { IDBFactory }          from 'fake-indexeddb';
import { SSM, MemoryStore }    from 'ssmjs';

// Obtain a WebGPU adapter from @webgpu/node (Google's Dawn implementation)
const gpuAdapter = await createGPU().requestAdapter({ powerPreference: 'high-performance' });
const idbFactory = new IDBFactory();

const runtime = await SSM.create({
  session: {
    gpuAdapter,                        // injected — skips navigator.gpu entirely
    idbFactory,                        // injected — skips globalThis.indexedDB
    checkpointUrl: '.coderClaw/model.bin',  // path to a user-trained .bin
    modelSize    : 'small',
    mambaVersion : 'mamba2',
  },
  bridge: myFrontierLLMBridge,         // cortex: Claude / GPT-4 / Ollama
});

// MemoryStore also accepts idbFactory for Node.js
const memory = new MemoryStore({ idbFactory });

for await (const token of runtime.stream('Explain this codebase:')) {
  process.stdout.write(token);
}
```

**Requirements:**
- Node.js 18+
- `@webgpu/node` — Dawn-based WebGPU for Node.js; supports all WGSL compute kernels used by MambaKit
- `fake-indexeddb` — in-memory IndexedDB compatible with the IDB spec; already used in SSM.js tests

**What works in Node.js:**
- ✅ Model inference (`generate`, `stream`, `streamHybrid`)
- ✅ Fine-tuning / WSLA adaptation (`adapt`)
- ✅ Distillation (`DistillationEngine`)
- ✅ `MemoryStore` (fact storage + weight persistence via `fake-indexeddb`)
- ❌ `storage: 'download'` — browser Blob URL download not available
- ❌ `storage: 'fileSystem'` — File System Access API not available; use `checkpointUrl` with a local file path instead

---

## Core Concepts

### SSMRuntime

The central runtime object. Wraps a `MambaSession` and adds routing, bridging, and lifecycle management.

```ts
import { SSMRuntime } from 'ssmjs';

const runtime = await SSMRuntime.create({
  session: {
    modelSize    : 'small',
    mambaVersion : 'mamba2',
    checkpointUrl: '/models/checkpoint.bin',
  },
  bridge          : new OpenAIBridge({ apiKey: 'sk-...' }),
  routingStrategy : 'auto',         // 'auto' | 'ssm' | 'transformer'
});

const response = await runtime.generate('Explain recursion');
await runtime.save();    // persist weights to IndexedDB
runtime.destroy();       // release GPU
```

---

### Transformer Bridges

Plug in any LLM as a teacher or fallback.

```ts
import { OpenAIBridge, AnthropicBridge, FetchBridge } from 'ssmjs';

// OpenAI
const gpt = new OpenAIBridge({ apiKey: 'sk-...', model: 'gpt-4o' });

// Anthropic
const claude = new AnthropicBridge({ apiKey: 'sk-ant-...' });

// Local / self-hosted (Ollama, LM Studio, vLLM)
const local = new FetchBridge({ baseUrl: 'http://localhost:11434/v1', model: 'llama3' });

// All bridges support streaming
for await (const token of claude.stream('Write a poem')) {
  output += token;
}
```

---

### InferenceRouter

Automatically routes each request to the right model.

```ts
import { SSMRuntime, InferenceRouter } from 'ssmjs';

// Default auto-routing (built into SSMRuntime.create)
const runtime = await SSM.create({
  session: { modelSize: 'nano' },
  bridge : claude,
  routingStrategy   : 'auto',
  longInputThreshold: 1200,     // chars before preferring transformer
  perplexityThreshold: 80,      // SSM perplexity threshold for fallback
});
```

**Auto-routing heuristics (cheapest first):**
1. **Complexity patterns** — detects "step by step", "analyze", "compare" → transformer
2. **Input length** — over threshold → transformer
3. **SSM perplexity** — async probe; high perplexity = novel topic → transformer

---

### DistillationEngine

Teach the SSM using a transformer — runs entirely in the browser.

```ts
import { DistillationEngine } from 'ssmjs';

const distiller = new DistillationEngine(runtime, claude);

// Single pass: claude generates → SSM adapts on output
const result = await distiller.distill('Explain WebGPU compute shaders');
console.log('Teacher:', result.teacherOutput);
console.log('Loss after:', result.adaptResult.losses.at(-1));

// Batch distillation
const batchResult = await distiller.distillBatch([
  'What is a Mamba block?',
  'Explain WSLA adaptation.',
  'How does SSD differ from S6?',
], { adapt: { wsla: true, epochs: 5 } });

console.log(`${batchResult.totalEpochs} epochs in ${batchResult.totalMs}ms`);
```

---

### MemoryStore

Persistent semantic key-value memory, separate from model weights.

```ts
import { MemoryStore } from 'ssmjs';

// Browser
const memory = new MemoryStore({ dbName: 'my-app' });

// Node.js — inject fake-indexeddb
import { IDBFactory } from 'fake-indexeddb';
const memory = new MemoryStore({ dbName: 'my-app', idbFactory: new IDBFactory() });

// Store facts
await memory.remember('author', 'Sean Hogg');
await memory.remember('project', 'MambaCode.js WebGPU SSM library');

// Retrieve
const entry = await memory.recall('project');
console.log(entry?.content);  // 'MambaCode.js WebGPU SSM library'

// List all
const all = await memory.recallAll();

// Persist and restore model weights
await memory.saveWeights(runtime);       // saves to IndexedDB under 'ssmjs-weights'
const loaded = await memory.loadWeights(runtime);   // returns false if none saved
```

---

### SSMAgent

High-level orchestration: conversation history, routing, memory injection.

```ts
import { SSMAgent, MemoryStore } from 'ssmjs';

const memory = new MemoryStore();
const agent  = new SSMAgent({
  runtime      : runtime,
  memory,
  systemPrompt : 'You are a senior TypeScript engineer.',
  maxHistoryTurns: 20,
});

// Store project context
await agent.remember('stack', 'React 18, TypeScript 5, Vite');
await agent.remember('goal', 'Build a real-time collaborative editor');

// Multi-turn conversation — facts matching input keys are auto-injected
const reply1 = await agent.think('What stack should I use?');
const reply2 = await agent.think('How do I handle concurrent edits?');

// Streaming
for await (const token of agent.thinkStream('Show me a WebSocket hook')) {
  output += token;
}

// Teach the agent from content
await agent.learn(myCodebase, { wsla: true, epochs: 3 });

console.log(agent.turnCount);  // 2
agent.clearHistory();
```

---

## Full Example

```ts
import {
  SSM,
  AnthropicBridge,
  DistillationEngine,
  MemoryStore,
  SSMAgent,
} from 'ssmjs';

// 1. Create runtime with hybrid routing
const runtime = await SSM.create({
  session: { modelSize: 'small', mambaVersion: 'mamba3' },
  bridge : new AnthropicBridge({ apiKey: process.env.ANTHROPIC_KEY! }),
  routingStrategy: 'auto',
});

// 2. Distill knowledge from the transformer into the SSM
const distiller = new DistillationEngine(runtime, runtime.bridge!);
await distiller.distillBatch([
  'What are the key differences between Mamba-1 and Mamba-2?',
  'How does WSLA fast-adaptation work?',
  'Explain the ET discretisation in Mamba-3.',
]);

// 3. Set up persistent memory
const memory = new MemoryStore();
await memory.saveWeights(runtime);  // save distilled weights

// 4. Build an agent
const agent = new SSMAgent({
  runtime,
  memory,
  systemPrompt: 'You are an expert on the Mamba SSM family.',
});

// 5. Interact
const answer = await agent.think('Compare Mamba-2 SSD and Mamba-3 ET');
console.log(answer);
```

---

## API Reference

### `SSM.create(opts)` / `SSMRuntime.create(opts)`

| Option | Type | Default | Description |
|---|---|---|---|
| `session` | `MambaSessionOptions` | required | Forwarded to `MambaSession.create()` |
| `bridge` | `TransformerBridge` | — | Transformer backend for routing/distillation |
| `routingStrategy` | `'auto'\|'ssm'\|'transformer'` | `'auto'` | Routing strategy |
| `longInputThreshold` | `number` | `1200` | Chars before auto-routing prefers transformer |
| `perplexityThreshold` | `number` | `80` | SSM perplexity cutoff |
| `callbacks` | `CreateCallbacks` | — | Progress callbacks |

### `runtime.generate(input, opts?)`
Generates a full response. Routes to SSM or transformer per strategy.

### `runtime.stream(input, opts?)`
`AsyncIterable<string>` — always uses SSM path for consistent latency.

### `runtime.streamHybrid(input, opts?)`
`AsyncIterable<string>` — routes like `generate()`, streams via bridge if available.

### `runtime.adapt(data, opts?)`
Pass-through to `session.adapt()`. Returns `AdaptResult`.

### `runtime.evaluate(text)`
Returns SSM perplexity. Used internally by auto-routing.

### `runtime.save(opts?)` / `runtime.load(opts?)`
Weight persistence pass-throughs to `MambaSession`.

### `runtime.destroy()`
Releases GPU device and all buffers.

---

## Error Handling

```ts
import { SSMError, SSMErrorCode } from 'ssmjs';
import { MambaKitError }          from 'mambakit';

try {
  const runtime = await SSM.create({ session: { modelSize: 'nano' } });
  await runtime.generate('hello');
} catch (err) {
  if (err instanceof SSMError) {
    // SSM.js-level error
    console.error(err.code);  // 'BRIDGE_REQUEST_FAILED' | 'RUNTIME_DESTROYED' | ...
  }
  if (err instanceof MambaKitError) {
    // Propagated from MambaSession (GPU / tokenizer failure)
    console.error(err.code);  // 'GPU_UNAVAILABLE' | 'TOKENIZER_LOAD_FAILED' | ...
  }
}
```

---

## File Structure

```
src/
├── index.ts                          ← package entry + SSM namespace
├── runtime/
│   └── SSMRuntime.ts                 ← core runtime, owns MambaSession
├── bridges/
│   ├── TransformerBridge.ts          ← interface
│   ├── OpenAIBridge.ts               ← OpenAI chat completions
│   ├── AnthropicBridge.ts            ← Anthropic Messages API
│   └── FetchBridge.ts                ← generic OpenAI-compatible endpoint
├── router/
│   └── InferenceRouter.ts            ← SSM ↔ transformer routing
├── memory/
│   └── MemoryStore.ts                ← IndexedDB fact store + weight helpers
├── distillation/
│   └── DistillationEngine.ts         ← online teacher→student distillation
├── agent/
│   └── SSMAgent.ts                   ← orchestration: history + routing + memory
└── errors/
    └── SSMError.ts                   ← typed error class
```

---

## Professional Platform

**SSM.js patterns are the architectural foundation of [Builderforce.ai](https://builderforce.ai)'s Agent Runtime.**

Builderforce.ai implements the SSM.js runtime model natively in the browser — the same `step()` → inference → confidence scoring → cloud escalation pipeline runs inside the IDE's `agent-runtime.ts`:

| SSM.js concept | Builderforce.ai equivalent |
|---|---|
| `SSMRuntime` | `AgentRuntime` (browser-native, ties to IDE project) |
| `DistillationEngine` | LLM-assisted dataset generation → in-browser LoRA training |
| `MemoryStore` | IndexedDB `MambaAgentState` + `AgentPackage` embedding |
| `SSMAgent` | Published workforce agent (Workforce Registry) |
| `TransformerBridge` | Cloudflare Workers AI / OpenRouter fallback |
| Confidence threshold → escalation | Auto-escalation to Workers AI when local score < threshold |

**What Builderforce.ai adds on top:**

- **Visual training panel** — configure LoRA rank, epochs, batch size, learning rate with a loss curve and live log console; no code required
- **Team collaboration** — real-time Yjs + Durable Objects CRDT editing; multiple users on the same project
- **Workforce Registry** — publish `SSMAgent`-equivalent specialists; discoverable by skills; hirable by the community
- **CoderClaw mesh** — agents deploy as self-hosted coding agents via [CoderClaw](https://coderclaw.ai) and receive task assignments from Builderforce

Use SSM.js to build custom agent runtimes in your own applications. Use Builderforce.ai for the full managed experience — IDE, training infrastructure, agent publishing, and enterprise orchestration.

---

## License

MIT
