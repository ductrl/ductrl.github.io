---
layout: post
title: Building CommitGate - Part 1
date: 2026-06-06
description: Part 1 of building CommitGate - a personal project I’m working on for the Splunk Agentic Ops Hackathon
toc:
    sidebar: left
tags: commitgate
categories: projects
---

Today, I am starting a new project that I have kept in my notes for a while now - a security gate that automatically blocks a Git commit if there is a vulnerability. 

This idea came to me last winter, when I was learning backend using NodeJS. As a (still) naive and inexperienced programmer, I committed and pushed my MongoDB Atlas API key to my public GitHub repository. I panicked. Atlas and GitHub threw dozens of emails at me, making me feel like the world was ending. Luckily, it was a harmless mistake in a repo no one knew about, and was also a great opportunity for me to learn how to remediate such problems. 

After that grave mistake, I went to find a tool that could prevent this from happening in the future. Discovering more about Gitleaks, TruffleHog, and GitHub Push Protections, etc. I realized that they all share the same limitation: None of them were specifically built for pre-commit prevention (at least not by default). It didn’t make sense to me - the commit is the stage where the vulnerability gets ingrained into Git history, it is where developers will have to spend valuable time removing and fixing it, where the mistake will only become increasingly costly as it moves along the development process. Moreover, I want to build on tools like Gitleaks that only have pattern recognition by adding an AI-powered vulnerability analysis code-scanning.

As such, I set out to create the tool myself: An AI-powered security gate that automatically blocks your commit if there is a vulnerability detected. Quite frankly, I don’t want it to be a comprehensive tool, as all I intended with this project is to have it as the first layer of security for those inexperienced or simply forgot about these mistakes like me. I am submitting this project to the [Splunk Agentic Ops Hackathon](https://splunk.devpost.com/), which means I’ll have until June 15th to have at least an MVP.

## 1. The Plan

As I plan for this project. Here are some things that came to mind.

### Key Flow

This is what I would imagine as a sensible and smooth flow for the project. This might change as we go deeper into development:

1. Tool is installed
2. User runs initialization (like init) to setup the tool as a pre-commit hook
3. ‘git commit’ triggers the tool
4. Run through staged changes (code diff) to determine which code should be analyzed
5. Analyze least-privileged code:
6. Secret scanning (API keys, access tokens, database credentials, cloud credentials, private keys, etc.)
7. Vulnerability analysis (SQL injection, command injection, authentication bypass, authorization flaws, insecure deserialization, debug mode exposure, weak cryptography, dangerous dependency changes, security misconfigurations, etc.)
8. Decide if commit can go through (e.g. warn-only for low to medium vulnerabilities, block otherwise) → low/medium/high
9. Generate security report

### Goals

With that flow in mind, here are some goals that I think is important for the tool to achieve:

- Detect hardcoded secrets before committing
- Detect common security vulnerabilities
- Prevent detected vulnerabilities from being committed to Git history (**Note**: User should have the option to override the stoppage)
- Generate vulnerability report & recommend actions
- Easy to setup & should run in <5s to minimize workflow disruption

Also, I think it is important that the tool doesn’t take too long to run (at least by default), as I do not want it to increase the development friction by too much.

## 2. Tech Stack

To be honest, this is a pretty simple tool, with very little code and should be implementable within a really short time frame (as the hackathon deadline is on June 15th). It is also a CLI tool, so here is the tech stack that me and my teammate Phuong have in mind:

- **Language**: Python
- **CLI**: Typer
- **Secret detection**: Gitleaks
- **LLM**: DeepSeek (simply for the low cost)
- **Security report**: Markdown
- **Repo**: GitHub
- **Testing**: Pytest

## Progress

Here is the progress of the project by the time of this blog:
- [x] Project planning
- [x] Repository setup
- [ ] Git hook installation
- [ ] Gitleaks integration
- [ ] AI vulnerability reviewer
- [ ] Security report generation
- [ ] Splunk integration

In the upcoming days, I will further discuss the project idea with my teammate to see if there’s anything to change, and hopefully we’ll have some components implemented!

## Conclusion

This project should be fun. It is the first time that I came up with a complete idea and have the opportunity to actually implement it. I will be documenting the journey of developing this tool during the hackathon timeframe. We’ll see how it goes!
