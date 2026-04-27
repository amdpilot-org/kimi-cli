Spawn a subagent to perform a specific task. The subagent starts with a fresh context — it cannot see your conversation history or prior results.

**Delegation Patterns**

Use the Task tool whenever work benefits from context isolation or parallelism. Two primary patterns:

1. **Supervisor delegation** — You act as a planner and verifier, delegating implementation phases to specialized subagents. This is the preferred pattern for complex multi-phase projects (e.g., porting a codebase, profiling, iterative optimization). Delegate entire phases, verify results yourself, then proceed to the next phase.

2. **Narrow delegation** — You are doing implementation work yourself and want to offload a specific subtask (fixing a build error, exploring unfamiliar code, searching the web) to keep your own context clean and focused.

Both patterns are valid. Choose based on the user's instructions and the complexity of the work.

**Writing Effective Task Prompts**

The subagent has a **completely fresh context** — your prompt must be self-contained. Include:

- **Objective:** What specifically needs to be done.
- **Context:** Relevant file paths, architecture details, and current state (e.g., current best latency, what's already been done, branch name).
- **Skills/references:** If the subagent should read a specific skill file or follow a protocol, say so explicitly and provide the path or content.
- **Expected output:** What the subagent must report back (e.g., "report all files modified, full benchmark stdout including `[BENCHMARK]` lines, and whether the benchmark script was changed").
- **Constraints:** What the subagent must NOT do (e.g., "do not modify benchmark measurement logic").

A vague prompt produces vague results. Be specific.

**Verifying Results**

When a subagent reports measurements or benchmark results, verify them yourself before committing or building on top of them. This is especially important in optimization workflows where correctness is cumulative — a fabricated or misreported number early on corrupts everything downstream.

**Iterating on Failures**

If a subagent's work fails or produces no improvement, do not simply give up. Spawn a new task with:
- The error message or regression details as context
- What was already tried and why it failed
- A request to diagnose the root cause and try an alternative approach

Make at least 2 attempts per optimization category before moving on.

**Parallel Multi-Tasking**

When subtasks are independent, call Task multiple times in a single response to run subagents in parallel:

- Exploring different parts of a large codebase simultaneously
- Implementing independent optimizations targeting different modules
- Running multiple web searches in parallel
- Porting multiple independent files or subsystems concurrently

**Available Subagents:**

${SUBAGENTS_MD}
