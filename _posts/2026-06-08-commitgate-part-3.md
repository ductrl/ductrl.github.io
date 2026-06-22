---
layout: post
title: "Building CommitGate - Part 3: Git Hooks Installation"
date: 2026-06-08
description: Part 3 of building CommitGate - a personal project I’m working on for the Splunk Agentic Ops Hackathon
toc:
    sidebar: left
tags: commitgate hackathon
giscus_comments: true
categories: projects
---

Today is the day I’ve been looking forward to, as we’ll finally get started on implementation! Upcoming blogs will be a bit more technical as I talk about our code as we implement CommitGate.

## 1. CLI Foundation

Before implementing any actual functionality, I needed to give users a way to interact with CommitGate. During the planning phase, I quickly decided that CommitGate should be a command-line tool since Git itself is mostly used in CLI. As CommitGate is to be integrated into the Git workflow, CLI felt like a natural choice for me.

At least for now, I think users should be able to use these three commands:

```
commitgate scan
commitgate install-hook
commitgate version
```

Upon some research on Python libraries to build CLI tools, I - like many - spent hours contemplating between `argparse`, `Click`, and `Typer`. I don’t want to go into too much details about the three, so I will refer those interested to [this well-written blog post comparing the three](https://medium.com/@mohd_nass/navigating-the-cli-landscape-in-python-a-comparative-study-of-argparse-click-and-typer-480ebbb7172f). I ended up going with Typer for its user-friendliness, which works well with CommitGate’s simplicity.

### Typer

Typer was relatively quick and easy to learn. The `@app.command()` decorator registers a function as a CLI command (if a fucntion name has `_`, it will convert `_` into `-` for the command name). This simple example creates the `commitgate version` command:

```python
import typer

app = typer.Typer()

@app.command()
def version():
    print("CommitGate 0.1.0")
```

Typer is also specifically helpful as it automatically generates help menus and argument parsing, thus allowing me to focus more on implementing other functionalities.

### Rich

If you watch any Youtube tutorial on Typer, you will more than likely see it being used with `Rich`, a library for colorful terminal output and formatting. The `print()` function is more than enough for my use case. Having that said, saying Rich is powerful is absolutely an understatement, as it provides so many great functionalities that can be very useful for larger CLI tools, which you can further explore from [this blog post on how to get start with Rich](https://medium.com/@ahmedharabi/get-started-with-the-python-rich-library-2736b1b57941).

### Initial Commands

For now, since we haven’t implemented any functionalities yet, I will just put some placeholders in the commands:

```python
import typer
from rich import print

app = typer.Typer()

@app.command()
def scan():
    # TODO: Implement
    print("[green]CommitGate scanned![/green]")

@app.command()
def install_hook():
    # TODO: Implement
    print("Installing Git hook...")

@app.command()
def version():
    print("CommitGate 0.1.0")
```

We will work on this as we implement other functionalities.

## 2. Git Hook Installation

Upon planning the idea of CommitGate, one of its core functionalities that I envisioned was that it needed to automatically run a security check every time the developer uses `git commit`. It is vital as the entire reason why I wanted to build CommitGate is that developers, especially inexperienced ones like me, make mistakes. We get lazy and don’t go through our code before committing. We get careless and will forget to run a security scan manually before pushing the code. 

Thus, we will need to set up a **Git pre-commit hook** for CommitGate. I discovered this concept a few months ago when I accidentally pushed my MongoDB Atlas API key and had to set up a pre-commit hook for Gitleaks. 

### What is a Git hook?

A Git hook is simply a script that Git automatically executes every time a specific event triggers it. For example, we can give Git scripts to run every time a commit is created, after a commit is created, before a push, after a merge, and many other points in the workflow. As mentioned above, CommitGate will utilize the pre-commit hook, which (intuitively) executes before Git creates a commit. The flow looks like the chart below:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://i.postimg.cc/635Q1J6Z/image.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

By running before a commit goes through, CommitGate can detect things like leaked secrets or security vulnerabilities and stop them before they become part of Git history. This helps lessen the hassle as even if the secret is deleted later, malicious actors can still find it in previous commits, thus requiring additional cleanup.

### Implementation

Git stores hooks inside the `.git/hooks/` directory of a repository. For a pre-commit hook, Git will look for an executable file named `pre-commit` and automatically execute it whenever `git commit` is run. Because of that, all we need to implement is to have CommitGate create a file at `.git/hooks/pre-commit` and write a simple shell script into it:

```
#!/bin/sh
commitgate scan
```

With this, whenever a developer runs `git commit`, Git will automatically execute `commitgate scan` before the commit is created.

This can be easily implemented using Python’s built-in `subprocess` module, which allows us to execute external commands and manage external processes. You will also see me using the built-in `pathlib` module a lot, as it helps us handle filepaths across different platforms.

```python
hook_path = Path(".git/hooks/pre-commit")
    
subprocess.run(
    f"echo '#!/bin/sh' > {hook_path}",
    shell=True,
    check=True
)

subprocess.run(
    f'echo \'commitgate scan\' >> {hook_path}',
    shell=True,
    check=True
)
```

Using the `run` function, the recommended approach to run external shell commands, a subprocess is created and returns a `CompletedProcess` object that has the output and exit code when it is finished. `shell=True` makes sure that the command is executed through the shell, and `check=True` makes it so that a `CalledProcessError` will be raised when the process exits with a non-zero exit code. 

As seen in the code above, the two `subprocess.run` function calls write `#!/bin/sh` to the `.git/hooks/pre-commit` file and then adds `commitgate scan` to the end of it.

One interesting thing I discovered upon implementing was that Git will silently ignore hook files that don’t have execute permission. As such, even though the hook file is successfully created, the scan is never actually run, which took me a good minute to debug.

An easy fix is to add execute permission to the file:


```python
subprocess.run(
    f"chmod +x {hook_path}",
    shell=True,
    check=True
)
```

And voila, the install hook function is complete! Only thing left to do is to utilize this function in the `install-hook` command: 

```python
@app.command()
def install_hook():
    hook_path = install_pre_commit_hook()

    print(f"Installed pre-commit hook at {hook_path}")
```

> ##### WARNING
>
> During a later stage of implementation, my teammate and I discovered two bugs about this implementation: 
> 1. This implementation assumes a Unix-like system (as we use `chmod`) and thus would not work on Windows
> 2. If a user’s git repo already have existing pre-commit hooks, this would completely overwrite it
> I will revisit this implementation and address these bugs in another blog post. 
{: .block-warning }

## 3. Determining What to Scan

During the planning process, we quickly came to the conclusion that having CommitGate scan the entire repository was too inefficient, especially if we want to have an LLM going through the code. This is especially important as CommitGate is designed to be integrated into the Git workflow and thus should absolutely not create too much friction to it.

Because of that, by default, we think it makes the most sense for CommitGate to only focus on the changes about to be committed.

### The Staging Area

The staging area is the most useful place for us to find such changes, it is the last place before these codes really get committed. When a developer runs `git add file.py`, Git puts the changes into the **staging area**, and by using this, we can get the staged files and code differences.

### Staged Files Retrieval

We can get a list of stage files by using:

```
git diff --cached --name-only
```

The `--cached` option tells Git to compare the staged changes to the last commit. Without it, Git would show the changes that hasn’t been staged yet. The `--name-only` flag tells Git to only output the filenames.

Using this, we can implement the `get_staged_files` function:

```python
def get_staged_files() -> list[str]:

    res = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True
    )

    return res.stdout.split()
```

### Staged Diff Retrieval

Knowing which files changed is useful, but obviously we will also need to get the code diff for CommitGate to scan. Similar to the previous function, we can get the staged diff using:

```
git diff --cached
```

And we can implement `get_staged_diff` like this:

```python
def get_staged_diff() -> str:
    res = subprocess.run(
        ["git", "diff", "--cached"],
        capture_output=True,
        text=True
    )

    return res.stdout
```

It is important that we get these two functions going, as they help CommitGate avoid unnecessary work and reduce AI costs. 

## Conclusion

That wraps it up for today! The next blog post will be dedicated to Gitleaks integration, as there is a little more code for that compared to what we’re doing today. The functions and foundations we built today will definitely come in handy for later stages of the project. After the Gitleaks integration, we will be able to actually use CommitGate to scan our commits and start using it as we intended, so I’m definitely looking forward to it!
