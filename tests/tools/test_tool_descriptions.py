from __future__ import annotations

# ruff: noqa

import platform
import pytest
from inline_snapshot import snapshot

from kimi_cli.tools.multiagent.create import CreateSubagent
from kimi_cli.tools.shell import Shell
from kimi_cli.tools.dmail import SendDMail
from kimi_cli.tools.file.glob import Glob
from kimi_cli.tools.file.grep_local import Grep
from kimi_cli.tools.file.read import ReadFile
from kimi_cli.tools.file.read_media import ReadMediaFile
from kimi_cli.tools.file.replace import StrReplaceFile
from kimi_cli.tools.file.write import WriteFile
from kimi_cli.tools.multiagent.task import Task
from kimi_cli.tools.think import Think
from kimi_cli.tools.todo import SetTodoList
from kimi_cli.tools.web.fetch import FetchURL
from kimi_cli.tools.web.search import SearchWeb


def test_task_description(task_tool: Task):
    """Test the description of Task tool."""
    assert task_tool.base.description == snapshot(
        """\
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

- `mocker`: The mock agent for testing purposes.
"""
    )


def test_create_subagent_description(create_subagent_tool: CreateSubagent):
    """Test the description of CreateSubagent tool."""
    assert create_subagent_tool.base.description == snapshot(
        """\
Create a custom subagent with specific system prompt and name for reuse.

Usage:
- Define specialized agents with custom roles and boundaries
- Created agents can be referenced by name in the Task tool
- Use this when you need a specific agent type not covered by predefined agents
- The created agent configuration will be saved and can be used immediately

Example workflow:
1. Use CreateSubagent to define a specialized agent (e.g., 'code_reviewer')
2. Use the Task tool with agent='code_reviewer' to launch the created agent
"""
    )


def test_send_dmail_description(send_dmail_tool: SendDMail):
    """Test the description of SendDMail tool."""
    assert send_dmail_tool.base.description == snapshot(
        """\
Send a message to the past, just like sending a D-Mail in Steins;Gate.

This tool is provided to enable you to proactively manage the context. You can see some `user` messages with text `CHECKPOINT {checkpoint_id}` wrapped in `<system>` tags in the context. When you feel there is too much irrelevant information in the current context, you can send a D-Mail to revert the context to a previous checkpoint with a message containing only the useful information. When you send a D-Mail, you must specify an existing checkpoint ID from the before-mentioned messages.

Typical scenarios you may want to send a D-Mail:

- You read a file, found it very large and most of the content is not relevant to the current task. In this case you can send a D-Mail immediately to the checkpoint before you read the file and give your past self only the useful part.
- You searched the web, the result is large.
  - If you got what you need, you may send a D-Mail to the checkpoint before you searched the web and put only the useful result in the mail message.
  - If you did not get what you need, you may send a D-Mail to tell your past self to try another query.
- You wrote some code and it did not work as expected. You spent many struggling steps to fix it but the process is not relevant to the ultimate goal. In this case you can send a D-Mail to the checkpoint before you wrote the code and give your past self the fixed version of the code and tell yourself no need to write it again because you already wrote to the filesystem.

After a D-Mail is sent, the system will revert the current context to the specified checkpoint, after which, you will no longer see any messages which you can now see after that checkpoint. The message in the D-Mail will be appended to the end of the context. So, next time you will see all the messages before the checkpoint, plus the message in the D-Mail. You must make it very clear in the message, tell your past self what you have done/changed, what you have learned and any other information that may be useful, so that your past self can continue the task without confusion and will not repeat the steps you have already done.

You must understand that, unlike D-Mail in Steins;Gate, the D-Mail you send here will not revert the filesystem or any external state. That means, you are basically folding the recent messages in your context into a single message, which can significantly reduce the waste of context window.

When sending a D-Mail, DO NOT explain to the user. The user do not care about this. Just explain to your past self.
"""
    )


def test_think_description(think_tool: Think):
    """Test the description of Think tool."""
    assert think_tool.base.description == snapshot(
        "Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed.\n"
    )


def test_set_todo_list_description(set_todo_list_tool: SetTodoList):
    """Test the description of SetTodoList tool."""
    assert set_todo_list_tool.base.description == snapshot(
        """\
Update the whole todo list.

Todo list is a simple yet powerful tool to help you get things done. You typically want to use this tool when the given task involves multiple subtasks/milestones, or, multiple tasks are given in a single request. This tool can help you to break down the task and track the progress.

This is the only todo list tool available to you. That said, each time you want to operate on the todo list, you need to update the whole. Make sure to maintain the todo items and their statuses properly.

Once you finished a subtask/milestone, remember to update the todo list to reflect the progress. Also, you can give yourself a self-encouragement to keep you motivated.

Abusing this tool to track too small steps will just waste your time and make your context messy. For example, here are some cases you should not use this tool:

- When the user just simply ask you a question. E.g. "What language and framework is used in the project?", "What is the best practice for x?"
- When it only takes a few steps/tool calls to complete the task. E.g. "Fix the unit test function 'test_xxx'", "Refactor the function 'xxx' to make it more solid."
- When the user prompt is very specific and the only thing you need to do is brainlessly following the instructions. E.g. "Replace xxx to yyy in the file zzz", "Create a file xxx with content yyy."

However, do not get stuck in a rut. Be flexible. Sometimes, you may try to use todo list at first, then realize the task is too simple and you can simply stop using it; or, sometimes, you may realize the task is complex after a few steps and then you can start using todo list to break it down.
"""
    )


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
def test_shell_description(shell_tool: Shell):
    """Test the description of Shell tool."""
    assert shell_tool.base.description == snapshot(
        """\
Execute a bash (`/bin/bash`) command. Use this tool to explore the filesystem, edit files, run scripts, get system information, etc.

**Output:**
The stdout and stderr will be combined and returned as a string. The output may be truncated if it is too long. If the command failed, the exit code will be provided in a system tag.

**Guidelines for safety and security:**
- Each shell tool call will be executed in a fresh shell environment. The shell variables, current working directory changes, and the shell history is not preserved between calls.
- The tool call will return after the command is finished. You shall not use this tool to execute an interactive command or a command that may run forever. For possibly long-running commands, you shall set `timeout` argument to a reasonable value.
- Avoid using `..` to access files or directories outside of the working directory.
- Avoid modifying files outside of the working directory unless explicitly instructed to do so.
- Never run commands that require superuser privileges unless explicitly instructed to do so.

**Guidelines for efficiency:**
- For multiple related commands, use `&&` to chain them in a single call, e.g. `cd /path && ls -la`
- Use `;` to run commands sequentially regardless of success/failure
- Use `||` for conditional execution (run second command only if first fails)
- Use pipe operations (`|`) and redirections (`>`, `>>`) to chain input and output between commands
- Always quote file paths containing spaces with double quotes (e.g., cd "/path with spaces/")
- Use `if`, `case`, `for`, `while` control flows to execute complex logic in a single call.
- Verify directory structure before create/edit/delete files or directories to reduce the risk of failure.

**Commands available:**
- Shell environment: cd, pwd, export, unset, env
- File system operations: ls, find, mkdir, rm, cp, mv, touch, chmod, chown
- File viewing/editing: cat, grep, head, tail, diff, patch
- Text processing: awk, sed, sort, uniq, wc
- System information/operations: ps, kill, top, df, free, uname, whoami, id, date
- Network operations: curl, wget, ping, telnet, ssh
- Archive operations: tar, zip, unzip
- Other: Other commands available in the shell environment. Check the existence of a command by running `which <command>` before using it.
"""
    )


def test_read_file_description(read_file_tool: ReadFile):
    """Test the description of ReadFile tool."""
    assert read_file_tool.base.description == snapshot(
        """\
Read text content from a file.

**Tips:**
- Make sure you follow the description of each tool parameter.
- A `<system>` tag will be given before the read file content.
- The system will notify you when there is anything wrong when reading the file.
- This tool is a tool that you typically want to use in parallel. Always read multiple files in one response when possible.
- This tool can only read text files. To read images or videos, use other appropriate tools. To list directories, use the Glob tool or `ls` command via the Shell tool. To read other file types, use appropriate commands via the Shell tool.
- If the file doesn't exist or path is invalid, an error will be returned.
- If you want to search for a certain content/pattern, prefer Grep tool over ReadFile.
- Content will be returned with a line number before each line like `cat -n` format.
- Use `line_offset` and `n_lines` parameters when you only need to read a part of the file.
- Use negative `line_offset` to read from the end of the file (e.g. `line_offset=-100` reads the last 100 lines). This is useful for viewing the tail of log files. The absolute value cannot exceed 1000.
- The tool always returns the total number of lines in the file in its message, which you can use to plan subsequent reads.
- The maximum number of lines that can be read at once is 1000.
- Any lines longer than 2000 characters will be truncated, ending with "...".
"""
    )


def test_read_media_file_description(read_media_file_tool: ReadMediaFile):
    """Test the description of ReadMediaFile tool."""
    assert read_media_file_tool.base.description == snapshot(
        """\
Read media content from a file.

**Tips:**
- Make sure you follow the description of each tool parameter.
- A `<system>` tag will be given before the read file content.
- The system will notify you when there is anything wrong when reading the file.
- This tool is a tool that you typically want to use in parallel. Always read multiple files in one response when possible.
- This tool can only read image or video files. To read other types of files, use the ReadFile tool. To list directories, use the Glob tool or `ls` command via the Shell tool.
- If the file doesn't exist or path is invalid, an error will be returned.
- The maximum size that can be read is 100MB. An error will be returned if the file is larger than this limit.
- The media content will be returned in a form that you can directly view and understand.

**Capabilities**
- This tool supports image and video files for the current model.
"""
    )


def test_glob_description(glob_tool: Glob):
    """Test the description of Glob tool."""
    assert glob_tool.base.description == snapshot(
        """\
Find files and directories using glob patterns. This tool supports standard glob syntax like `*`, `?`, and `**` for recursive searches.

**When to use:**
- Find files matching specific patterns (e.g., all Python files: `*.py`)
- Search for files recursively in subdirectories (e.g., `src/**/*.js`)
- Locate configuration files (e.g., `*.config.*`, `*.json`)
- Find test files (e.g., `test_*.py`, `*_test.go`)

**Example patterns:**
- `*.py` - All Python files in current directory
- `src/**/*.js` - All JavaScript files in src directory recursively
- `test_*.py` - Python test files starting with "test_"
- `*.config.{js,ts}` - Config files with .js or .ts extension

**Bad example patterns:**
- `**`, `**/*.py` - Any pattern starting with '**' will be rejected. Because it would recursively search all directories and subdirectories, which is very likely to yield large result that exceeds your context size. Always use more specific patterns like `src/**/*.py` instead.
- `node_modules/**/*.js` - Although this does not start with '**', it would still highly possible to yield large result because `node_modules` is well-known to contain too many directories and files. Avoid recursively searching in such directories, other examples include `venv`, `.venv`, `__pycache__`, `target`. If you really need to search in a dependency, use more specific patterns like `node_modules/react/src/*` instead.
"""
    )


def test_grep_description(grep_tool: Grep):
    """Test the description of Grep tool."""
    assert grep_tool.base.description == snapshot(
        """\
A powerful search tool based-on ripgrep.

**Tips:**
- ALWAYS use Grep tool instead of running `grep` or `rg` command with Shell tool.
- Use the ripgrep pattern syntax, not grep syntax. E.g. you need to escape braces like `\\\\{` to search for `{`.
- Hidden files (dotfiles like `.gitlab-ci.yml`, `.eslintrc.json`) are always searched. To also search files excluded by `.gitignore` (e.g. `node_modules`, build outputs), set `include_ignored` to `true`. Sensitive files (such as `.env`) are still skipped for safety, even when `include_ignored` is `true`.
"""
    )


def test_write_file_description(write_file_tool: WriteFile):
    """Test the description of WriteFile tool."""
    assert write_file_tool.base.description == snapshot(
        """\
Write content to a file.

**Tips:**
- When `mode` is not specified, it defaults to `overwrite`. Always write with caution.
- When the content to write is too long (e.g. > 100 lines), use this tool multiple times instead of a single call. Use `overwrite` mode at the first time, then use `append` mode after the first write.
"""
    )


def test_str_replace_file_description(str_replace_file_tool: StrReplaceFile):
    """Test the description of StrReplaceFile tool."""
    assert str_replace_file_tool.base.description == snapshot(
        """\
Replace specific strings within a specified file.

**Tips:**
- Only use this tool on text files.
- Multi-line strings are supported.
- Can specify a single edit or a list of edits in one call.
- You should prefer this tool over WriteFile tool and Shell `sed` command.
"""
    )


def test_search_web_description(search_web_tool: SearchWeb):
    """Test the description of MoonshotSearch tool."""
    assert search_web_tool.base.description == snapshot(
        "WebSearch tool allows you to search on the internet to get latest information, including news, documents, release notes, blog posts, papers, etc.\n"
    )


def test_fetch_url_description(fetch_url_tool: FetchURL):
    """Test the description of FetchURL tool."""
    assert fetch_url_tool.base.description == snapshot(
        "Fetch a web page from a URL and extract main text content from it.\n"
    )
