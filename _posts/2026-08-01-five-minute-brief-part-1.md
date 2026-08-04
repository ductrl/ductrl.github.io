---
layout: post
title: "Building Five-Minute-Brief - Part 1: Planning"
date: 2026-08-01
description: "Part 1 of building Five-Minute-Brief"
toc:
    sidebar: left
tags: five-minute-brief
giscus_comments: true
categories: projects
--- 

Welcome welcome! Today, I am starting on a new project that I have been wanting to do for quite a while now. It is called **Five-Minute-Brief**. Essentially, it is a minimalist Chrome extension that summarizes global & US news from the past 24 hours within 3-7 headlines. I planned the project around the core idea that is for lazy readers like me to spend 5 minutes everyday to catch up on the news, just enough so that we can converse with others about the news if need be.

I spent all of my time today just focusing on planning, with the goal deliverables to be a 90% completed PRD and TDD. Here is the gist of some of my decisions and the reasoning behind them.

<br>

## 1. Product Principles
---

There are several core principles that I want the product to follow.

First is a hard limit of 3-7 stories/headlines per day. I might change this range later on, although I think this should be good but I will have to try out the product itself to see, but I believe that a range needs to be established so that we can reliably make sure the daily news recap remains short enough for one to skim through and move on with their day. 

Next, there should be no account or any kind of personalization. I want this to be a uniform news window for everyone. There are two reasons for this. One is that I want this product to be as light and as easy to use as possible: One click to download the extension and that’s it - you can open the window to read the daily news recap any time. Two is to minimize unnecessary technical complexity and cost. With this design, I can take on the cost of the summarization generation myself, as it should be a very small number of tokens use every day, and also I won’t have to implement a user account system with a backend and a database.

The principle that’s really a headache for me right now is news gathering, as I want it to pick out the most important news for summarization based on the consensus from major news outlets. This is one part of the PRD that I will have to spend more time working on since we will need to come up with a system that can reliably (and **legally**) find a consensus among news outlets.

<br>

## 2. Editorial Scope
---

For the MVP, I want the product to strictly recap 3-7 major headlines that are reported among news outlets within the past 24 hours (I will probably base this on the ET timezone). The product does not update the news based on live or breaking-news updates and will only change the content every 24 hours (with the exception of minor fixes if the news recapped is factually incorrect or grammar mistakes or other similar stuff).

As mentioned, I want the product to produce only 3-7 headlines, and ideally an average of 4-5 stories every day. Headlines and summaries are AI-generated (and I will **definitely display a very visible warning saying that the contents are all AI-generated**) and should not be copied from news outlets in any shape or form (or else I will be doomed legally). Ideally, the content should:

* Describe the central event clearly.
* Avoid clickbait and emotional language.
* Remain understandable without opening the summary.
* Normally fit on one or two lines.
* Avoid overstating certainty or significance.

The summaries will be one- to two-sentence long with a structure of **what happened** and **necessary context to understand it**. Under each headline summary, there should be up to 3 buttons displaying publication names, which will open up the source article for users to read on the topic. I understand that users will prefer certain news sources based on their political standpoint, but I do not intend to spend too much time on worrying about this.

<br>

## 3. Tech stack
---

This is the tech stack I plan to use to implement this product:

1) **We will use WXT as this will be a Chrome extension**. This was suggested by ChatGPT and after my own research, it is the industry-leading framework to develop web extensions. As you can tell, I am unfamiliar with the tool and will have to learn how to use this for the project.
1) **I will use React for UI**. I have been spending the past 2 years self-studying JavaScript, and so I think it is time I use it in a personal project. I want to become more proficient in JavaScript through this project too, so you will see I use a JS-central tech stack.
1) **Bootstrap for frontend styling**. I initially intended to use Tailwind, but like many others, I hate styling my product the most, and since this is just a minimalist app that I don’t really intend to scale up too much, I will just lazily go with Bootstrap.
1) **For AI I will use OpenAI models since I just recently received free credits from hackathons**, which is embarrassingly a part of why I decided to get started on this project right now. 
1) **I will also use GitHub Actions for daily scheduling for news generation**.
1) **Maybe, just maybe, I will use GitHub Pages for static JSON hosting**.

<br>

## Conclusion
---

This is plenty of planning for the day already. I have also set up the [repo](https://github.com/ductrl/five-minute-brief), created some issues, and set up a Kanban board to track to-dos a while ago but nothing much to talk about. With the last few days of summer coming to an end, hopefully I will be able to finish this project by the start of the semester. Really excited!!!
