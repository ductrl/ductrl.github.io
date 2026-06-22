---
layout: post
title: "Building CommitGate - Part 4: Gitleaks Integration & Git Hook Install Fix"
date: 2026-06-12
description: Part 4 of building CommitGate - a personal project I’m working on for the Splunk Agentic Ops Hackathon
toc:
    sidebar: left
tags: commitgate hackathon
giscus_comments: true
categories: projects
--- 

Welcome back to part 4 of building CommitGate! Today, we will work on integrating Gitleaks into the tool. Also, there is a major bug with the Git hook installation process, which I will go into more detail about later. Let’s get started!

<br>

## 1. Gitleaks Integration
--- 

Rather than implementing our own deterministic secret scanning, I think it’s more efficient to utilize an available tool like Gitleaks, as it is specifically built and maintained for this purpose (and is probably better implemented than whatever my version would be). 

For Gitleaks integration, we will implement three functions: `is_gitleaks_installed()`, `parse_gitleaks_findings()`, and `run_gitleaks_scan()`. Only `run_gitleaks_scan()` will be used by the CLI.

<br>

#### is_gitleaks_installed

This is a function returning a bool indicating whether Gitleaks is installed. The implementation is quite straightforward by using `shutil.which()`, which returns the file path of an executable application. By checking if the function returns `None` for Gitleaks, the function does exactly what we want:


```python
def is_gitleaks_installed() -> bool:
    return shutil.which("gitleaks") is not None
```

<br>

#### parse_gitleaks_findings

This function is used to parse Gitleaks JSON report and returns a list of dicts. Among the keys that Gitleaks gives us, the ones we will use are `description`, `start_line`, `end_line`, `start_column`, `end_column`, and `file` (So the description of the leaked secret and its location).

First, we will check if the report exists, if it is a file, and if its type is JSON and raise an error if any of these are not true:

```python
report_path = Path(report_path)

    if not report_path.exists():
        raise FileNotFoundError(f"Gitleaks file report not found: {report_path}")
    
    if not report_path.is_file():
        raise ValueError(f"Expected a file but received: {report_path}")
    
    if report_path.suffix.lower() != ".json":
        raise ValueError(f"Expected a .json report file, got: {report_path}")
```

A Gitleaks JSON report file will look something like this:

```json
[
  {
    "Description": "AWS Access Key ID",
    "StartLine": 12,
    "EndLine": 12,
    "StartColumn": 16,
    "EndColumn": 36,
    "Match": "AKIAIOSFODNN7EXAMPLE",
    "Secret": "REDACTED",
    "File": "config/production.json",
    "SymlinkTarget": "",
    "Commit": "b3f71c4d9a8b2c5e6f1a3d5e7f9a0b2c4d6e8f0a",
    "Author": "Jane Doe",
    "Email": "jane.doe@example.com",
    "Date": "2026-05-14T14:23:11Z",
    "Message": "feat: add production environment variables",
    "Tags": ["key", "AWS"],
    "RuleID": "aws-access-key-id",
    "Fingerprint": "b3f71c4d9a8b2c5e6f1a3d5e7f9a0b2c4d6e8f0a:config/production.json:aws-access-key-id:12"
  },
  {
    "Description": "GitHub Personal Access Token",
    "StartLine": 45,
    "EndLine": 45,
    "StartColumn": 12,
    "EndColumn": 52,
    "Match": "ghp_1234567890abcdefghijklmnopqrstuvwxyz",
    "Secret": "REDACTED",
    "File": "scripts/deploy.sh",
    "SymlinkTarget": "",
    "Commit": "7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b",
    "Author": "John Smith",
    "Email": "john.smith@example.com",
    "Date": "2026-06-02T09:15:44Z",
    "Message": "fix: resolve deployment script auth issues",
    "Tags": ["token", "GitHub"],
    "RuleID": "github-pat",
    "Fingerprint": "7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b:scripts/deploy.sh:github-pat:45"
  }
]
```

Using Python’s built-in `json` library, we can easily convert this report into a list of dictionaries:

```python
with open(report_path, "r", encoding="utf-8") as f:
    raw_findings = json.load(f)
```

Then, we will run a for loop to extract only the needed keys and append them to our own list of dictionaries:

```python
findings = []

for item in raw_findings:
    findings.append(
        {
                "description": item.get("Description"),
                "start_line": item.get("StartLine"),
                "end_line": item.get("EndLine"),
                "start_column": item.get("StartColumn"),
                "end_column": item.get("EndColumn"),
                "file": item.get("File"),
                "rule": item.get("RuleID")
        }
    )
```

<br>

#### run_gitleaks_scan

Now the hard part! We will implement the `run_gitleaks_scan()` function, the one that the CLI will call to run a Gitleaks scan on the staged files.

One conscious choice that we’re making in this implementation is that Gitleaks will run on all staged files instead of staged diffs, as having it scan staged diffs will make it quite challenging to find the location of the leaked secret. We will create an issue to have it scan staged diff instead in the future.

We will use this command to run Gitleaks on a file of our choice:

```bash
gitleaks dir path_to_file --report-format json --no-banner --redact --report-path path_to_report_file
```

`gitleaks dir path_to_file` lets us scan whatever directories or files we input. `--report-format json` tells Gitleaks to output the report as a JSON file into whatever file we provided through `--report-path path_to_report_file`. `--no-banner` removes the unnecessary Gitleaks banner from the output, and `--redact` prevents Gitleaks from printing the secret into the report.

First, we get a list of staged file paths from the `get_staged_files()` function we implemented in [part 3](https://ductrl.github.io/blog/2026/commitgate-part-3/) and then loopthrough each file:

```python
staged_files = get_staged_files()

for file_path in staged_files:
    path = Path(file_path)
```

For CommitGate, we will use the built-in `tempfile` library to create a temporary file path for the Gitleaks JSON report through the `NamedTemporaryFile` function:

```python
with tempfile.NamedTemporaryFile(mode="w+t", suffix=".json", delete=True) as report_file:
    report_path = report_file.name
```

And then run the mentioned Gitleaks command and have it output the report into the temporary file:

```python
command = [
    "gitleaks",
    "dir",
    file_path,
    "--report-format",
    "json",
    "--no-banner",
    "--redact",
    "--report-path",
    report_path
]

result = subprocess.run(
    command, 
    capture_output=True,
    text=True
)
```

> ##### WARNING
>
> We discovered a bug in this implementation that prevents it from working on Windows. As this is the second time we have faced cross-platform issues, it might be wise for me to set up a Windows virtual machine to prevent these type of bugs from occuring in the future.
{: .block-warning }

After that, after checking Gitleaks’s return code for success, we parse the findings using the `parse_gitleaks_findings()` function and add them to our list of findings:

```python
if result.returncode == 126:
    raise RuntimeError(f"Gitleaks failed while scanning {file_path}:\n{result.stderr}")

if result.returncode not in (0, 1):
    raise RuntimeError(f"Gitleaks failed while scanning {file_path}:\n{result.stderr}")
            
findings.extend(parse_gitleaks_findings(report_path=report_path))
```

And this is the complete function:

```python
def run_gitleaks_scan() -> list[dict]:
    if not is_gitleaks_installed():
        raise RuntimeError("Gitleaks is not installed. Please install it before running CommitGate")

    staged_files = get_staged_files()

    findings = []

    for file_path in staged_files:
        path = Path(file_path)

        if not path.exists() or not path.is_file():
            continue

        with tempfile.NamedTemporaryFile(mode="w+t", suffix=".json", delete=True) as report_file:
            report_path = report_file.name
        
            command = [
                "gitleaks",
                "dir",
                file_path,
                "--report-format",
                "json",
                "--no-banner",
                "--redact",
                "--report-path",
                report_path
            ]

            result = subprocess.run(
                command, 
                capture_output=True,
                text=True
            )

            if result.returncode == 126:
                raise RuntimeError(f"Gitleaks failed while scanning {file_path}:\n{result.stderr}")

            if result.returncode not in (0, 1):
                raise RuntimeError(f"Gitleaks failed while scanning {file_path}:\n{result.stderr}")
            
            findings.extend(parse_gitleaks_findings(report_path=report_path))
    
    return findings
```

<br>

## Git Hook Installation Bug Fix
---

Like I mentioned in part 3, we discovered two crucial bugs that need to be fixed:

1. This implementation assumes a Unix-like system (as we use `chmod`) and thus would not work on Windows
1. If a user’s git repo already have existing pre-commit hooks, this would completely overwrite it

Problem one comes in from these lines:

```python
subprocess.run(
    f"chmod +x {hook_path}",
    shell=True,
    check=True
)
```

We needed this because all Git hook files, including pre-commit, require executable permissions to run. Without that, Git will silently ignore and the scan won’t run. However, `chmod` is not a native Windows command, and thus would result in an error when a user tries to use the `commitgate install-hook` command on Windows.

To fix this, instead of having us using the shell directly, we will have Python handle it instead using the `pathlib` library. We will create a simple `_write_commitgate_hook()` function:

```python
def _write_commitgate_hook(hook_path: Path) -> None:
    hook_path.write_text(
        f"#!/bin/sh{COMMITGATE_HOOK_BLOCK}",
        encoding="utf-8",
    )

    hook_path.chmod(0o755)
```

Now, we will avoid overwriting users’ existing pre-commit hooks by considering 3 different cases.

The first case is if there is no pre-commit hook yet, we will just simply create the hook using the function we just implemented:

```python
if not hook_path.exists():
    _write_commitgate_hook(hook_path=hook_path)
    return hook_path

existing_content = hook_path.read_text(encoding="utf-8")
```

Note that we have to consider an edge case here, where there exists a pre-commit file but it’s empty:

```python
if not existing_content.strip():
    _write_commitgate_hook(hook_path=hook_path)
    return hook_path
```

The second case is if the CommitGate hook is already installed, then we basically don’t do anything:

```python
if "commitgate scan" in existing_content:
    hook_path.chmod(0o755)
    return hook_path
```

Finally, the last case is if there is a pre-commit hook already installed, then we will add the hook to the end of the file.

```python
hook_path.write_text(
    existing_content.rstrip() + COMMITGATE_HOOK_BLOCK,
    encoding="utf-8",
)
hook_path.chmod(0o755)
```

`COMMITGATE_HOOK_BLOCK` is just a global variable we created: `COMMITGATE_HOOK_BLOCK = "\n\n# CommitGate hook\ncommitgate scan\n"`

Note that there is another edge case where the existing pre-commit hook is non-shell, then we will raise an error telling the user to install the hook manually.

<br>

## Conclusion
---

And that is plenty of coding for today! Next up in part 5, my teammate Phuong will talk about how he implemented the AI reviewer, and then, it will be my responsibility to integrate both Gitleaks and the AI reviewer into the CLI. Stay tuned!
