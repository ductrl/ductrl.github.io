---
layout: post
title: Checkout my new project CommitGate!
date: 2026-06-18
inline: false
related_posts: false
---

My teammate Phuong and I just finished building [CommitGate](https://github.com/ductrl/CommitGate) for the [Splunk Agentic Ops Hackathon](https://devpost.com/software/commitgate?)!  

It is an AI-powered tool that automatically scan your staged codes for security vulnerability every time you run `git commit` (Check out the demo GIF below). I truly believe that this would be a useful tool for any developer and am extremely proud to be able to build something like this. As we're both very passionate about this project, we are still working to improve and add more features to it even after the hackathon has ended.

Checkout the [project page](https://ductrl.github.io/projects/commitgate/) and [give it a star on GitHub](https://github.com/ductrl/CommitGate) if you find the tool helpful!

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/gif/commitgate-demo.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    CommitGate blocking a vulnerable commit containing hardcoded API keys and command injections.
</div>