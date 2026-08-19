---
layout: post
title: "Building Five-Minute-Brief - Part 4: Empty States & JSON Schema"
date: 2026-08-11
description: Part 4 of building Five-Minute-Brief
toc:
    sidebar: left
tags: five-minute-brief
giscus_comments: true
categories: projects
--- 

Welcome back my lovely people! Today will be quite some work since I have decided to cover two items within today’s blog. Let’s hop straight into it!

<br>

## 1. Added Loading, Empty and Error States  
--- 

### Header

First item of the day is to add user-friendly loading and error states so that the extension doesn’t completely crumble when the story isn’t loaded or something unexpected happens. This should be pretty straightforward.

First changes I made was to the header. Right now, the header looks something like this:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://i.postimg.cc/DfMf0HGn/image.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

However, it would display the date and story count even if those data isn’t passed through. We want the first two lines to display no matter what (as it is static and never changes), and the third line to only appear when this data is available:


```jsx
const Header = ({ date, storyCount }) => (
  <header>
    <p>The essential read</p>
    <h1>Five Minute Brief</h1>
    {date && storyCount &&
      <p>
        {date} 
        <i className="bi bi-dot"></i>
        {storyCount} stories
      </p>
    }
  </header>
)
```

<br>

### Status Message

Next up, we will create a StatusMessage component so that we can display an error message to users when something goes wrong. This is important as we don’t want to display more technical details to user than necessary. We will make further changes to this component when there are more error messages we need to warn the user:

```jsx
const StatusMessage = ({ status }) => {
  if (status === 'success') {
    return null;
  }

  if (status === 'loading') {
    return (
      <section>
        <div
          className="spinner-border"
          role="status"
        />
        <p>Loading today's brief...</p>
      </section>
    )
  }

  if (status === 'empty') {
    return <p>No brief is available yet for today. Check back later!</p>
  }

  if (status === 'network-error') {
    return <p>We couldn’t load the brief. Please try again later.</p>
  }

  if (status === 'invalid-data') {
    return <p>Today’s brief is unavailable.</p>
  }

  return <p>An unexpected error happened. Please try again later.</p>
}

export default StatusMessage;
```

<br>

### App

Finally, we incorporate these changes into the App:

```jsx
const App = () => {
  const [status, setStatus] = useState('success');

  if (status !== 'success') {
    return (
      <main>
        <Header/>
        <StatusMessage status={status}/>
      </main>
    )
}

  return (
    <main>
      <Header 
        date={mockBrief.date} 
        storyCount={mockBrief.stories.length}
      />

      <StoryList 
        stories={mockBrief.stories}
      />

      <Footer publishedAt={mockBrief.publishedAt}/>
    </main>
  )
}
```

<br>

## 2. Zod Schema and Edition Validation
--- 

The heavier codes for today is here. We will define the Zod Schema for the edition, which is basically the format the edition will need to conform too. Also, to follow best practices, we will write tests to make sure that the validation works correctly.  

We will start from the smallest to the biggest schemas, with the first being the news source schema:

```js
const SourceSchema = z.object({
  publisher: z.string().trim().min(1),
  url: z.url().startsWith('https://'),
})
```

This requires the publisher name to have at least one character and that the source url starts with `https://`. Next up is the story schema:

```js
const StorySchema = z.object({
  id: z.string().trim().min(1),
  headline: z.string().trim().min(1),
  summary: z
    .string()
    .trim()
    .min(1)
    .refine((summary) => countWords(summary) <= MAX_SUMMARY_WORDS, {
      message: `Summary must be ${MAX_SUMMARY_WORDS} words or fewer`
    }),
  sources: z.array(SourceSchema).min(1).max(3),
})
```

Here, we required story id and headline to have at least 1 character. I might change my mind in the future about setting a max character/word limit for headlines, but we’ll see how AI-generated headlines look first. We also limited sources to be between 1-3. As you can see we have also made sure that summaries can only have at most `MAX_SUMMARY_WORDS` (which I set to 50) with this helper functions:


```js
const countWords = (text) => {
  const trimmed = text.trim();
  return (trimmed === "" ? 0 : trimmed.split(/\s+/).length);
}

const SourceSchema = z.object({
  publisher: z.string().trim().min(1),
  url: z.url().startsWith('https://'),
})
```

Last but definitely not least is the edition schema:

```js
const EditionSchema = z
  .object({
    schemaVersion: z.literal(1),

    editionDate: z.iso.date(),

    coverageStart: z.iso.datetime(),
    coverageEnd: z.iso.datetime(),
    publishedAt: z.iso.datetime(),

    stories: z.array(StorySchema).min(3).max(7),
  })
  .refine(
    (edition) => {
      const ids = edition.stories.map(story => story.id);
      return (new Set(ids).size === edition.stories.length);
    }, 
    {
      message: 'Story IDs must be unique',
      path: ['stories']
    }
  )
  .refine(
    (edition) => new Date(edition.coverageStart) < new Date(edition.coverageEnd),
    {
      message: 'Coverage end must occur after coverage start',
      path: ['coverageEnd']
    }
  )
```

I won’t go over too much details, but here are some of the validations we are enforcing:

* Dates and times like `editionDate` or `publishedAt` must follow [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html)
* There can only be 3-7 stories
* There are no duplicate story IDs
* Coverage start must be before coverage end

I have also written tests to make sure these validations work correctly by letting the schema parses invalid editions. I won’t show them here since its a lot of codes that no one wants to read (who likes writing tests anyway, certainly not me), but you can checkout the tests in the `[edition.test.js file](https://github.com/ductrl/five-minute-brief/blob/main/tests/edition.test.js)`.

<br>

## Conclusion
---

And that wraps it up for today! I just now realized I have not created an issue for styling the UI, which is a massive overlook on my part. For real this time, we will definitely work on styling the UI next time and have it look like a real product. See you soon!

