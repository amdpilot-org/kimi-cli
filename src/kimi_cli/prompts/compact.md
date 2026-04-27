
---

The above is a list of messages in an agent conversation. You are now given a task to compact this conversation context according to specific priorities and rules.

**Compression Priorities (in order):**
1. **Current Task State**: What is being worked on RIGHT NOW, including the exact goal and target metric
2. **Optimization/Fix Progress**: What approaches have been tried and their outcomes (metric values, errors, pass/fail). This is CRITICAL — the agent must not repeat failed approaches
3. **Errors & Solutions**: All encountered errors and their resolutions (full stack traces for unresolved errors)
4. **Profiling Data**: Any kernel-level profiling results — kernel names, time percentages, hotspot rankings. These are expensive to regenerate and must be preserved verbatim
5. **Code Evolution**: Final working versions only (remove intermediate attempts). Which files were modified and what changes worked
6. **System Context**: Project structure, dependencies, environment setup, container state
7. **Key File Locations**: Paths to important files discovered during exploration (config files, kernel source, benchmark scripts)
8. **Design Decisions**: Architectural choices and their rationale
9. **TODO Items**: Unfinished tasks, known issues, and promising unexplored directions

**Compression Rules:**
- MUST KEEP: Error messages, stack traces, working solutions, current task, all benchmark metric values, profiling kernel breakdowns, file paths that were modified
- MUST KEEP: The contents of any workspace state files the agent has been maintaining (e.g. optimization_state.json, learned_insights.md, bench_config.env) — summarize their key fields
- MERGE: Similar discussions into single summary points
- REMOVE: Redundant explanations, verbose code reading output, repeated benchmark runs with identical results
- CONDENSE: Long code blocks → keep signatures + key logic only
- PRESERVE: All numeric results (latency values, percentages, kernel timings) — these are irreplaceable

**Special Handling:**
- For code: Keep full version if < 20 lines, otherwise keep signature + key logic
- For errors: Keep full error message + final solution
- For discussions: Extract decisions and action items only
- For benchmarks: Keep the metric value and what config produced it
- For profiling: Keep the full kernel breakdown (kernel name → percentage) — this is the most valuable data

**Required Output Structure:**

<current_focus>
[What we're working on now, including target metric and current best]
</current_focus>

<environment>
- [Key setup/config points]
- [Container state: Python version, venv path, key packages installed]
- [Key file paths discovered]
- ...more...
</environment>

<approaches_tried>
- [Approach]: [Metric result or error] — [Keep/Revert]
- [Approach]: [Metric result or error] — [Keep/Revert]
- ...more (list ALL, never omit any)...
</approaches_tried>

<profiling_data>
[Kernel breakdown if available — kernel names and time percentages]
[Profiling tool used and any raw data paths]
</profiling_data>

<completed_tasks>
- [Task]: [Brief outcome]
- ...more...
</completed_tasks>

<active_issues>
- [Issue]: [Status/Next steps]
- ...more...
</active_issues>

<code_state>

<file>
[filename]

**Summary:**
[What this code file does]

**Key elements:**
- [Important functions/classes]
- ...more...

**Latest version:**
[Critical code snippets in this file]
</file>

...more files...
</code_state>

<important_context>
- [Any crucial information not covered above]
- [Workspace state file contents summary (optimization_state.json, learned_insights.md)]
- [Supervisor hints and nudge guidance received]
- ...more...
</important_context>
