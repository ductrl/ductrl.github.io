---
layout: post
title: "Building Five-Minute-Brief - Part 2: Initialize Chrome extension project"
date: 2026-08-07
description: Part 2 of building Five-Minute-Brief
toc:
    sidebar: left
tags: five-minute-brief
giscus_comments: true
categories: projects
--- 

Welcome back! This is part 2 of building Five Minute Brief, a daily news recap Chrome extension that I’m working on. In this blog, I will talk about what I have completed to solve our first item of the project.  

Today was a pretty light day, all we did was set up the project with Vite, install Bootstrap, and created a placeholder page for now. I actually dropped WXT from the tech stack and decided to just go with a normal Vite + React project since that will allow me to practice what I have learned from Full Stack Open. After running `vite build`, we will then just load the `dist` folder into Chrome.

Like for all Chrome extensions, we first created a `manifest.json` file (in the `public` directory since we want Vite to copy the file into `dist`):

```json
{
  "manifest_version": 3,
  "name": "Five Minute Brief",
  "version": "0.1.0",
  "description": "A five-minute daily briefing of major US and world news.",
  "action": {
    "default_popup": "index.html"
  }
}
```

After doing all of that, we created a simple page in `App.jsx`:

```jsx
const App = () => {
  return (
    <div className="container p-4">
      <h1 className="fs-4 fw-bold mb-2">Five-Minute Brief</h1>

      <p className="text-secondary mb-3">
        Your daily briefing of the news that matters.
      </p>

      <div className="alert alert-light border mb-0">
        Today's brief is coming soon.
      </div>
    </div>
  )
}

export default App
```

And that would look something like this:  


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://i.postimg.cc/pd8rkbRy/image.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<br>

## Conclusion
---

Like I said, chill day! Thank you so much for reading. For part 3, we will really work on the popup and have it look like an actual product (with fake news). See you soon!
