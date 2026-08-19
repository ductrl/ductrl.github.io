---
layout: post
title: "Building Five-Minute-Brief - Part 3: Static Popup UI"
date: 2026-08-08
description: Part 3 of building Five-Minute-Brief
toc:
    sidebar: left
tags: five-minute-brief
giscus_comments: true
categories: projects
--- 

Welcome back everyone! Today the coding start (though nothing close to challenging yet). Today, we will implement the barebone static UI page, just React components with absolutely no styling yet. Let’s get started!

<br>

## 1. The Design
---

Sorry, I lied. As much as I wanted to jump straight into coding, we need an actual design. However, Five Minute Brief is supposed to be a minimalist popup, it should be pretty straightforward. As much as I hate to say it, AI was pretty helpful for this part since I was too lazy and unskilled to come up with a good UI myself. I used Canva AI to generate a “minimalist Chrome extension news digest accordion-style” for me and to my surprise, one prompt was good enough. The product is definitely nowhere near good enough for a professional setting, but for a personal one-person project, this is plenty for me (it also forgot to generate the headlines). The generated UI looks like this:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://i.postimg.cc/j2dJ8xxj/image.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>  

Now, we can hop into the coding part!

<br>

## 2. Mock Data
---
First item of the day is to have some mock data that we can use for the popup. I used an LLM for this since it is pretty handy and looks more realistic. Totally could have lorem ipsum-ed this, but I already paid for ChatGPT plus so might as well.

The mock data looks like this:

```js
// FAKE MOCK DATA

export const mockBrief = {
  date: 'August 7, 2026',
  publishedAt: '8:00 a.m. ET',

  stories: [
    {
      id: 'story-1',
      headline: 'Global markets steady as investors assess new economic data',
      summary:
        'Global markets closed largely unchanged after fresh data pointed to cooling price pressures alongside uneven growth. Investors are now watching closely for the next signal from central banks.',
      sources: [
        {
          publisher: 'Reuters',
          url: 'https://www.reuters.com',
        },
        {
          publisher: 'BBC',
          url: 'https://www.bbc.com',
        },
        {
          publisher: 'AP',
          url: 'https://apnews.com',
        },
      ],
    },

    {
      id: 'story-2',
      headline: 'Major international leaders meet for new round of talks',
      summary:
        'Officials met Thursday for negotiations focused on several unresolved international disputes.',
      sources: [
        {
          publisher: 'Reuters',
          url: 'https://www.reuters.com',
        },
        {
          publisher: 'AP',
          url: 'https://apnews.com',
        },
      ],
    },
    ...
  ],
}

```

> **Note**: I have not decided on the specific shape for the data yet. I will leave this a problem for when I work on the news generation part. For now, we will just roll with this.

<br>

## 3. Header
---

The first component to create is the Header component, which would be this part:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://i.postimg.cc/BZY8SGZX/image.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

This should be pretty straightforward:

```jsx
const Header = ({ date, storyCount }) => (
  <header>
    <p>The essential read</p>
    <h1>Five Minute Brief</h1>
    <p>
      {date} 
      <i className="bi bi-dot"></i>
      {storyCount} stories
    </p>
  </header>
)

export default Header;
```

> ##### TIP
>
> I will be using Bootstrap icons for this project as you can see above
{: .block-tip }

<br>

## 4. News summarization
---

Next, the ‘*body*’ where the news headlines and summaries lie. That would be this part:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://i.postimg.cc/d0fsMJjk/image.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Here, you will see that I decided to handle the accordion list with React rather than Bootstrap. This is because I think this will be handier when we need to save which headlines are open. I might go with Bootstrap if this causes too many headaches for styling though.  

For this part, we will need to implement 2 components: `StoryList` and `StoryItem`. `StoryItem` is the component for each piece of news, which means `StoryList` will just be a section with a bunch of `StoryItem`s. We will start with `StoryItem`:  

```jsx
import { useState } from "react";

const StoryItem = ({ 
  headline, 
  summary, 
  sources,
  defaultOpen=false
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  const handleToggle = () => {
    setIsOpen(!isOpen);
  }

  return (
    <article>
      {/* ----- Headline ----- */}
      <button
        onClick={handleToggle}
      >
        <span>{headline}</span> 
        <span>
          {isOpen
            ? <i className="bi bi-chevron-up"></i>
            : <i className="bi bi-chevron-down"></i>
          }
        </span>
      </button>

      {/* ----- Content ----- */}
      {isOpen && (
        <div>
          <p>{summary}</p>
          {sources.map(source => (
            <a 
              key={source.publisher}
              href={source.url}
              target="_blank"
            >
              {source.publisher}|
            </a>
          ))}
        </div>
      )}
    </article>
  )
}

export default StoryItem;
```

Next is `StoryList`, which should be significantly more straightforward:  

```jsx
import StoryItem from "./StoryItem";

const StoryList = ({ stories }) => {
  return (
    <section>
      {stories.map((story, index) => (
        <StoryItem 
          key={story.id}
          headline={story.headline}
          summary={story.summary}
          sources={story.sources}
          defaultOpen={index === 0}
        />
      ))}
    </section>
  )
}

export default StoryList;
```

> **Note**: Right now I am using `story.id` as the key, but I am thinking about changing it to something more dynamic like using the story index instead. This is really easy to do so we’ll worry about it later.

<br>

## 5. Footer
---

Last but not least, the footer:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://i.postimg.cc/k4PHRZn6/image.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

There’s not much to talk about for this one, it’s just three lines of words:   

```jsx
const Footer = ({ publishedAt }) => {
  return (
    <footer>
      <h2>That's the brief for today.</h2>
      <p>Published at {publishedAt}</p>
      <p>AI-generated headlines and summaries may contain error.</p>
    </footer>
  )
}

export default Footer;
```

<br>

## 6. App
---

And now, putting them all together:

```jsx
import Header from "./components/Header";
import StoryList from "./components/StoryList";
import Footer from "./components/Footer";
import { mockBrief } from "./data/mockBrief";

const App = () => {
  return (
    <>
      <Header 
        date={mockBrief.date} 
        storyCount={mockBrief.stories.length}
      />

      <StoryList 
        stories={mockBrief.stories}
      />

      <Footer publishedAt={mockBrief.publishedAt}/>
    </>
  )
}

export default App;
```

And now, the extension looks like this:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://i.postimg.cc/wTMwMHXj/image.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Looks rough, I know, but that’s why we’re going to beautify this bad boy next up!

<br>

## Conclusion
---

That should be plenty for today. Next time, we will get started on styling and have it look like an actual product, which would be the part I dreaded the most. See you guys soon!
