---
layout: post
title: Building CommitGate - Part 2
date: 2026-06-07
description: Part 2 of building CommitGate - a personal project I’m working on for the Splunk Agentic Ops Hackathon
toc:
    sidebar: left
tags: commitgate hackathon
categories: projects
---

This is part 2 of building CommitGate - an AI-powered pre-commit Git security gate I’m building for the Splunk Agentic Ops Hackathon. I don’t have too much time today so we’ll just further focus on planning and setting up the project. You can visit the project repository on Github [here](https://github.com/ductrl/CommitGate).

## Setting Up The Workflow

In order to have a smooth workflow, I decided to set up some components so that my teammate and I can work on this project seamlessly. I tried not to overcomplicate it so that it doesn’t take up too much time and have unnecessary information for a 1-week project with only 2 people working on it.

### To-Do List and Kanban Board

If there’s one thing I learned from past projects and work experience, it’s that just blindly jumping into coding and remembering what needs to be implemented only works until it suddenly doesn’t. I remember my first hackathon, when my teammates just kept asking what to do, and I just made up random things for them to do on the spot. It was messy and unorganized. My teammates felt like they weren’t really contributing. No one had any idea what was going on but me, and we ended up with nothing done.

Even though this is a two-person project, since that experience, I have always had the habit of setting up a to-do list and a kanban board. To keep this simple, I went with GitHub Projects as the kanban board, with GitHub Issues essentially acting as my to-do list (any added issues are automatically added to the board’s backlog). For now, I’ll add the main features as the issues first and will add any necessary subtasks later:

* Project Setup and CLI Foundation
* Git Hook Installation
* Staged diff retrieval
* Gitleaks Implementation
* AI Reviewer
* Decision Engine
* Report Generator

I have also created an Issue Template to make it easier to create issues.

### pyproject.toml

One of the first things I set up was also `pyproject.toml`. This is important as the file acts as a central configuration file for Python projects, allowing my teammate and me to easily set the project up on our local machines.

### CONTRIBUTING.md

Going back to the story of my first hackathon, although unbelievable, but my team’s workflow was genuinely having my teammates coding with me using VS Code Live Share. We “used Git” by having them using `git add .`, `git commit -m` and `git push` themselves. There were no code reviews, no PRs, no GitHub issues, no branching. We just committed whatever we want, bugs or not, with whatever message we want, all to main. Thinking back, it was insane how we went with that, which makes a lot of sense, seeing that we got nothing done. Thus, it is absolutely vital that we actually use the common Git workflow for this project.

Although it’s just 2 people and I can go through the Git development workflow with my teammates, I think it’s really helpful if I create a file just so that we can have a reference point and consensus on how we work on the project. The file just details how to set up the project on the local machine and the basic Git workflow. I tried my best to keep it short and concise so that it doesn’t take up too much time.

### architecture.md

I believe the most important thing for my teammate and I to have a consensus on is to understand the key workflow and the components of the tool. As such, I am also creating an `architecture.md` file in the `docs/` directory listing the different modules and describing them. For now, the file looks like this:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://i.postimg.cc/DZNtZxw3/image.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

It is helpful to plan out the architecture, since it helped me realize that the project naturally breaks into different components that can be worked on independently. Thus, this allows my teammates and I to allocate work between ourselves based on our strengths. For example, since my teammate specializes in AI Engineering, he will be working on the AI reviewer module while I, with more experience in cybersecurity and understanding more about git hooks, will be implementing things like Gitleaks integration and Git hooks installation.

Throughout this whole process, I found this [blog on documentation](https://johnjago.com/great-docs/) that I found extremely helpful. If I have more time before the deadline of this project, I will circle back to this blog to improve the documentation of this project as I think this will be a very valuable skill to learn.

## Conclusion

We still haven’t done any coding yet, but I think having a clear plan and workflow is extremely important for actual development and would greatly improve efficiency if done well. Next time, we’ll definitely get started on the coding and start implementing components of the app. Let’s see how it goes!
