---
layout: page
title: CommitGate
description: An AI-powered Git pre-commit security gate
img: assets/gif/commitgate-demo.gif
importance: 1
category: personal
giscus_comments: true
related_publications: false
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/gif/commitgate-demo.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    CommitGate blocking a vulnerable commit containing hardcoded API keys and command injections.
</div>

---
<br>

## Links
---

* [GitHub repo](https://github.com/ductrl/CommitGate)
* [Devpost project page](https://devpost.com/software/commitgate)
* [Blog series on CommitGate development](https://ductrl.github.io/blog/tag/commitgate/)
* [Demo video](https://youtu.be/ZYe5vWFRTus?si=kVy7t30MgZUi-6p0)

<br>

## Overview  
---

CommitGate is an AI-powered Git pre-commit security gate that scans staged changes before a commit is created. The tool combines deterministic secret detection with AI-based semantic review to catch both known patterns and contextual vulnerabilities. 

The tool integrates directly into the existing Git workflow by installing a pre-commit hook. Every time the user runs `git commit`, CommitGate automatically analyzes the staged changes before deciding whether the commit should proceed. By stopping vulnerabilities before they enter the Git repository history, CommitGate can catch issues earlier and significantly reduce the cost of remediation.

> ##### WARNING
>
> CommitGate is intended to be the first layer of DevSecOps, and is not supposed to be a comprehensive code scanner tool.
{: .block-warning }

<br>

## The Problem
---

Most security tools come in too late in the development lifecycle, as developers usually rely on PR reviews, CI/CD security scans, GitHub Push Protection, manual code review, etc. While effective, they only occur after code has already been committed, sometimes even pushed.

I myself experienced this firsthand after I naively committed a MongoDB Atlas API key to GitHub. I was confused as GitHub and Atlas immediately detected the leak and warned me about it via email, yet there was no automatic prevention or block mechanism for it. Rotating credentials quickly fixed the issue, but I could not help but notice that such security checks were happening after the mistake rather than before.

<br>

## Solution 
---

CommitGate was designed to address that gap by moving the security check to commit time to prevent such mistakes before they can even enter Git history.

We designed the flow so that when the user runs `git commit`:

1. A Git pre-commit hook runs a CommitGate scan
1. CommitGate analyzes only the **staged changes**
1. Gitleaks searches for known secret patterns
1. An AI reviewer searches for contextual vulnerabilities
1. Findings are combined and evaluated by a decision engine
1. Decision engine compares the finding severities against a configurable policy and decides whether the commit is allowed, warned, or blocked
1. (Optional) Audit events are sent to Splunk and can be visually viewed on a dashboard

<br>

## Key Features
---

<br>

#### Git Integration

CommitGate is directly integrated into the developer’s workflow through Git pre-commit hooks.

<br>

#### Deterministic Secret Detection

[Gitleaks](https://github.com/gitleaks/gitleaks) is used to scan for secret leaks, which can detect API keys, access tokens, passwords, cloud credentials, etc. with known patterns.

<br>

#### AI-Powered Security Semantic Analysis

An AI reviewer will be used to analyze code to detect contextual vulnerabilities to capture findings that Gitleaks might miss, including command injection, data leakage risks, and unsafe deserialization. CommitGate supports any OpenAI-compatible model provider.

<br>

#### Configurable Policies

The user can configure block severity thresholds, AI provider, and reporting options. We are currently working to expand these options.

<br>

## Architecture 
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://i.postimg.cc/m2F4hb1D/Commit-Gate-architecture-diagram.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Components:

* **Git pre-commit hook**: Intercepts commit and runs the scan
* **Staged Changes Retriever**: Extracts only the staged changes, avoid scanning the whole repo for efficiency
* **Gitleaks Runner**: Performs deterministic secret detection
* **AI Reviewer**: Uses an OpenAI-compatible model to evaluate contextual security vulnerabilities
* **Decision Engine**: Merges findings and determines final action for the commit
* **Report Generator**: Formats findings using Rich to produce a readable security report to the terminal
* **Splunk Logger**: Exports audit events for monitoring and analysis

<br>

## Technical Challenges
---

<br>

#### Cross-Platform

One of the biggest problem we faced was to build CommitGate to be cross-platform. This was especially apparent when implementing Git pre-commit hook installation and Gitleaks integration.

For the pre-commit hook, we don't want to just blindly overwrite user's pre-commit file, but to also check if the file already exists and append the command to the end if suitable. Based on the OS, we have to also handle executable permissions and avoid disrupting existing workflows, which makes implementing the hook installation and running Gitleaks difficult. Finding the balance point for complex and potentially unsafe implementation for the sake of user setup simplicity was more challenging than I expected.

<br>

#### Reliable AI Output

Another problem we faced was building a reliable AI reviewer. LLMs can produce inconsistent outputs, false positives, and findings in different formats. Because of that, we spent significant time refining our prompts, constraining the model's output structure, and validating responses to ensure that security findings could be processed automatically by the rest of the pipeline.

<br>

#### Splunk Integration

Integrating CommitGate with Splunk also presented a learning curve. We had to understand how to send structured events through the HTTP Event Collector (HEC), design a schema for security findings, and ensure that telemetry collection remained non-blocking so that developers could continue committing code even if Splunk was temporarily unavailable.

<br>

## What We Learned
---

This was my first personal project. Building CommitGate has provided valuable experience across multiple domains for me, such as:

* The importance of careful planning
* A deeper understanding on how Git works and hook systems
* Using AI smartly to boost productivity but also as a great learning tool
* Security tooling integration
* Collaborative software development

I will discuss my lessons learned in an upcoming detailed blog post and will link it here.

<br>

## Future Work
---

Here are some improvements that we are working on (I will constantly update this section as more features/fixes come to mind):

* AI CLI subscriptions support (Codex, Claude, Antigravity, etc.)
* Pre-push hook option (instead of pre-commit)
* More configuration options
* Improve scan time

<br>

## Team
---

This was a two-person project. I want to dedicate this section to credit my teammate **Phuong (Mark) Hoang**, who was responsible for:

* AI reviewer implementation
* OpenAI-compatible model integration
* Parsing AI outputs
* AI prompt design
* Setting up Splunk integration and connecting CommitGate with its infrastructure
* Contributed to the blog series, documentation, and hackathon submission

Without his help, this project definitely would not have been possible. It was truly endless fun spending countless hours coding with him on this project!

As for my part, I was the project lead and was responsible for:

* Planning project architecture and design
* Git integration
* Pre-commit hook installation
* Gitleaks integration
* CLI development
* Documentation
* Demo and submission

<br>

## Development Journey
---

Interested in the technical details of this project?

Read the full build log (More coming in the future):

* [Building CommitGate - Part 1](https://ductrl.github.io/blog/2026/commigate-part-1/): Planning & Architecture
* [Building CommitGate - Part 2](https://ductrl.github.io/blog/2026/commitgate-part-2/): Workflow & Repository Setup
* [Building CommitGate - Part 3](https://ductrl.github.io/blog/2026/commitgate-part-3/): Git Hooks Installation
* [Building CommitGate - Part 4](https://ductrl.github.io/blog/2026/commitgate-part-4/): Gitleaks Integration & Git Hook Install Fix

