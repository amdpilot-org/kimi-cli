Consult a senior advisor for guidance when you are stuck or facing a difficult decision.

The advisor is a frontier-class model that can help with:
- **Diagnosis**: understanding what's going wrong and why
- **Direction**: which hypothesis to pursue next
- **Stop/go decisions**: whether to continue current approach or pivot

## When to use this tool

Use `ConsultAdvisor` when you encounter one of these situations:
- You have tried multiple approaches and none are working (hypothesis exhaustion)
- You see contradictory evidence and cannot determine which interpretation is correct
- You are unsure whether an issue is an infrastructure/environment problem or the target bug
- You have been reading code for many steps without running a benchmark or making progress

## When NOT to use this tool

Do NOT use `ConsultAdvisor` for:
- Simple questions you can answer by reading code or running a command
- Asking for exact code patches or implementations (the advisor gives direction, not code)
- Routine decisions that don't require senior guidance

## Limits

You have a maximum of **${MAX_CONSULTS} consults per trial**. Use them wisely at genuine decision points, not for routine questions. The advisor will respond with structured guidance including a diagnosis, recommended next action, and things to avoid.

## How it works

When you call this tool, your request is sent to a senior advisor who reviews your situation and responds with structured guidance. The response typically arrives within 1-2 minutes.
