---
layout: post
title: Building CommitGate - Part 5: CLI Integration
date: 2026-07-10
description: Part 5 of building CommitGate - a personal project I’m working on for the Splunk Agentic Ops Hackathon
toc:
    sidebar: left
tags: commitgate hackathon
giscus_comments: true
categories: projects
--- 

Welcome back to part 5 of building CommitGate! It has been a while, as unfortunately I have not been able to allocate the appropriate time to continue these blog posts while working on my other personal projects and hackathons. Because of that, I plan to end this dev journey blog series within the next few blog posts. My teammate and I have been continuously working on this project even after the hackathon (from which we should hear back about the results in one week). In the upcoming blog posts I want to briefly talk about what we have changed and some of the thinking behind them.

<br>

## 1. CLI 
--- 

We have made a lot of changes to the CLI. Let’s take a look at some of the changes.  

First, let’s look at how the security scan is run:

```python
gitleaks_findings = run_gitleaks_scan(file_paths=file_paths)

if ai_enabled:
    ai_findings, ai_review_ok = review(diff=diff, staged_files=file_paths, timeout=timeout)
else:
    ai_findings, ai_review_ok = [], True

all_findings = remove_dup(gitleaks_findings + ai_findings)
```

Gitleaks is run by default, and we haven’t provided (nor intend to) any options to turn off Gitleaks scan yet. The reason we have to have an `ai_review_ok` variable is to know whether the AI scan was run successfully or not, as it will return `[]` in `ai_findings` both when no findings are found and when there is an error during the scanning process.

The findings are then passed into the `removed_dup` function implemented in the `report_generator` module. We will then print the appropriate message to the terminal based on the findings.  

When there are no findings:  

```python
if not all_findings:
    if not ai_enabled:
        print("[yellow]AI review disabled by config.[/yellow]")
    
    print("[green]No security findings found![/green]")
    print("[green]CommitGate scan completed![/green]")
    raise typer.Exit(code=0)
```

When there are findings:  

```python
decision = decide(all_findings)
log_decision(decision)
action = decision["action"]

color = "yellow" if action == "warn" else "red"
print(f"[{color}]CommitGate detected {len(all_findings)} security finding(s):[/{color}]")

for index, finding in enumerate(all_findings):
    severity = finding.get("severity", "").lower()
    sev_color = severity_color(severity=severity)

    print(
        f"[{sev_color}]"
        f"[{severity.upper()}] Finding #{index + 1}"
        f"[/{sev_color}]"
    )

    finding_output = format_finding(finding=finding, fields=fields)

    print(finding_output)
    print()
```

We also added an `init` command that installs the git hook and creates the config file:  

```python
@app.command()
def init():
    config_file = create_default_config()
    hook_path = install_git_hook()

    print(f"[green]Created config file:[/green] {config_file}")
    print(f"[green]Installed {hook_path.name} hook:[/green] {hook_path}")
```

<br>

## 2. Config System
---  

We implemented a config system that allows users to configure CommitGate. The configurations are stored in a `commitgate.yaml` file so that users within a project can share the same configurations (or they can simply add it to `.gitignore`).  

Creating the default config file is very straightforward:

```python
DEFAULT_CONFIG = {
    "enabled": True,
    "ai": {
        "enabled": True,
        "provider": "deepseek",
        "timeout": 20,
    },
    "policy": {
        "block_severity": "high",
    },
    "reporting": {
        "min_severity": "medium",
        "fields": {
            "source": True,
            "category": True,
            # "severity": True,
            # "file": True,
            # "location": True,
            "description": True,
            "suggestions": True,
        },
    },
}

DEFAULT_CONFIG_YAML = """\
# Enable or disable CommitGate for this repository.
enabled: true

ai:
  # Enable AI-powered security review.
  enabled: true

  # AI provider to use.
  # Option 1: (AI_KEY in .env): openai, gemini, deepseek, kimi, groq (Tip: groq offers a free API key - at https://console.groq.com)
  # Option 2: Claude Code or Codex (no API key): claude-cli, codex-cli
  provider: deepseek

  # Maximum time (seconds) allowed for AI review.
  timeout: 20

policy:
  # Findings at or above this severity block the commit/push.
  # Options: low, medium, high, critical
  block_severity: high

reporting:
  # Minimum severity shown in CommitGate output.
  # Must be <= block_severity, so a blocking finding is never hidden
  # Options: low, medium, high, critical
  # Example: medium shows medium, high, and critical findings, but hides low findings.
  # Raising this also speeds up the AI review (fewer findings to generate).
  min_severity: medium

  # Control which optional fields are displayed for each finding.
  # Turning off description and suggestions also speeds up the AI review.
  fields:
    source: true
    category: true
    description: true
    suggestions: true
"""
```

We’ll use `DEFAULT_CONFIG` when we need to load the default config and `DEFAULT_CONFIG_YAML` when creating the default file. This is done so that the file comes with helpful comments to let the user know how to use the different configuration options. To create the file:

```python
path = get_config_path()

if path.exists() and not overwrite:
    return path

path.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
return path
```

To load user’s configurations, we merge it with the default configs and then validate them:

```python
path = get_config_path()

if not path.exists():
    config = deepcopy(DEFAULT_CONFIG)
else:
    with open(path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) 
    
    if not user_config:
        config = deepcopy(DEFAULT_CONFIG)
    elif not isinstance(user_config, dict):
        raise ValueError("commitgate.yaml must contain a YAML dictionary")
    else:
        config = merge_with_defaults(user_config)

validate_config(config)
return config
```

`merge_with_defaults` uses the `merge_dicts` function to recursively merge the user configuration dict with the default one. It needs to be recursive since different configuration options might have further subdicts.  `merge_dicts` is implemented like this:

```python
def merge_dicts(default_dict: dict, user_dict: dict) -> dict:
    """
    Helper function: Recursive dictionary merge
    """
    result = deepcopy(default_dict)

    for key, value in user_dict.items():
        # if the current subfield is also a dictionary, then we call the merge function
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result
```

I will not show the validation function since it is pretty straightforward. We check the user config against a pre-defined set of valid options and raise a ValueError if it is not valid.

<br>

## Conclusion
---

Unfortunately, this will be the last post talking about CommitGate’s implementation. I might write some more posts in the future discussing its updates but nothing is decided yet. In the next post (which will be the last of the whole series), I will talk about the lessons I learned creating CommitGate, and also give some final words on the project. See yall then!
