/**
 * orchestration — stateful multi-agent graphs (supersteps, declared channel
 * merges, checkpointing) plus the three patterns enterprise work reduces to:
 * ReAct, self-reflection, and hierarchical delegation.
 */

export {
    AgentGraph,
    appendReducer,
    mergeReducer,
    sumReducer,
    unionReducer,
} from './AgentGraph.js';
export type { AgentGraphOptions, RunOptions } from './AgentGraph.js';

export { InMemoryCheckpointer, KeyValueCheckpointer } from './checkpoint.js';
export type {
    InMemoryCheckpointerOptions,
    KeyValueCheckpointerOptions,
    KeyValueLike,
} from './checkpoint.js';

export {
    createReactAgent,
    createReflectionAgent,
    createSupervisor,
    asWorker,
    parseReactTurn,
    renderToolCatalog,
    renderTranscript,
    REACT_PROTOCOL,
    DEFAULT_CRITIC_PROMPT,
    FINISH,
} from './patterns.js';
export type {
    AgentTool,
    AgentTurn,
    AgentTurnRole,
    ParsedAction,
    ReactAgentOptions,
    ReactState,
    ReflectionAgentOptions,
    ReflectionState,
    SupervisorOptions,
    SupervisorState,
    Worker,
} from './patterns.js';

export { START, END } from './types.js';
export type {
    ChannelReducer,
    ChannelSpec,
    ChannelSpecs,
    Checkpoint,
    Checkpointer,
    EdgeCondition,
    GraphEndReason,
    GraphEvent,
    GraphRunResult,
    GraphState,
    NodeContext,
    NodeFn,
} from './types.js';
