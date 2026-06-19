---
layout: post
title: NBA Playoff Qualification Prediction Using Logistic Regression
date: 2025-07-01 15:00:00
description: Predict whether an NBA team will make the playoffs or not using a logistic regression model
toc:
  sidebar: left
tags: machine-learning nba
giscus_comments: true
categories: projects
---

In this project, we will try to predict whether an NBA team will make the playoffs or not based on the team's stats. I intend the use case to be for a fan to input their team's stats in the middle of the season to predict whether they are going to make the playoffs or not. Thus, rather than selecting 16 teams to make the playoffs each season, I believe it makes more sense to treat each team's stats individually and predict solely based on those stats, as fans cannot input all 30 teams' stats. We will use Logistic Regression to predict, so if one wants to see which 16 teams are going to make the NBA playoffs in a certain season, they can select the 8 teams with the highest predicted probability in each conference. For more details and code, check out the [Google Colab notebook](https://colab.research.google.com/drive/1dL6zs01zy1hVEAotqQkb0l-rLnVGzfct?usp=sharing) of this project to learn more.

<br>

## 1. Project Setup

Using a logistic regression model, we will predict if an NBA team can make the playoffs based on its stats.

<br>

## 2. Data Collection

We will use the already available NBA Statistics [NBA Statistics Repository](https://github.com/Brescou/NBA-dataset-stats-player-team) on GitHub. This repository contains many datasets, all of which are in the `csv` format, so they are all structured data that can be processed using Pandas.

<br>

## 3. Data Preparation

<br>

#### i. Exploratory Data Analysis (EDA)

I only want to use regular-season stats to predict the qualification (well, if you have playoff stats, then it would mean you are already in the playoffs). Since we're looking at simple data that fans can input themselves, `team_stats_traditional_rs.csv` should be a good dataset to use.

Here are some observations that can be made:

*   We won't use GP, W, L, and any stats that involve these 3 attributes (such as `W_PCT`) since midseason fans shouldn't need to predict the future.
*   I want to use stats that should be easy to access to a casual fan, and I also said I want to treat each team's stats individually, so we will not use ranking stats (such as `MIN_RANK`)
* `MIN` is pretty useless as it basically only measures how much a team has to go to overtime.
* I personally consider all field goal stats to be too much of a hassle for a fan to look up and put in.
* I think using `REB` is enough, so we will not use `OREB` and `DREB` for memory efficiency.
* `BLKA` (block against) is unnecessary.
* `PF` and `PFD` vary too much depending on the era (e.g., Hack-a-Shaq), so I will not use them.
* Here are the stats that we end up using: `REB`, `AST`, `TOV`, `STL`, `BLK`, `PTS`, `PLUS_MINUS`.

<br>

#### ii. Data Preprocessing

All the data have different scales, and since none of these stats are capped in a certain range (we would if we included, for example, field goal percentage, which ranges from 0 to 1), we will use standardization rather than normalization.

```python
scaler = StandardScaler()
df[relevant_cols] = scaler.fit_transform(df[relevant_cols])
```

Notice that this dataset is still missing a crucial column: the label! We don't have a column that indicates whether this team made the playoffs or not. Thus, we will cross-check with the playoff dataset and see if a team's stats show up in the playoff dataset, and use that to label that team's playoff qualification.

There are many ways you can do this. I will just create a column called `PLAYOFF`, which indicates playoff qualification status with the default value as `1` in the playoff dataset, and then merge the 2 datasets. Then, I will replace all NaN values with `0`.

```python
po_df['PLAYOFF'] = np.ones(len(po_df))
po_df = po_df[['TEAM_ID', 'SEASON', 'PLAYOFF']]
```

Now we merge the datasets to get the label. Then, we will replace NaNs with 0.

```python
df = df.merge(po_df, on=['TEAM_ID', 'SEASON'], how='outer')

relevant_cols.append('PLAYOFF')
df = df[relevant_cols]
df = df.fillna(0)
df.head()
```

<br>

#### iii. Data Splitting

```python
X = df[relevant_cols[:-1]]
y = df['PLAYOFF']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
```

> ##### TIP
>
> It might be better to filter out seasons that are so far away in the past to ensure data relevancy. However, I find that doing so would leave us with too few data points to work with.
{: .block-tip }

> ##### TIP
>
> I believe have the model predict the most recent season (thus split the samples by season e.g. all the team stats in the 3 most recent seasons) as the testing set would be a much better splitting method for evaluation, but I'm going to proceed as usual since we have already spent plenty of time on data preparation.
{: .block-tip }

<br>

## 4. Modeling

Ah, the modeling part! Implementing a Logistic Regression model using `sklearn` is quite straightforward. Since we’re working with a relatively small dataset and only need binary classification, I will be using the `liblinear` solver.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear', random_state=23)
model.fit(X_train, y_train)
```

<br>

## 5. Evaluation

Let’s see how our model does!

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(accuracy * 100, 2)}%')
```

And our model has an accuracy of 88.82%. This is much better than I expected. Now feel free to try out the model with your favorite team!

## 6. Final Words

While you're here, this is part 4 of a machine learning project series where I apply every machine learning algorithm I learned to an NBA-related project. If you want to check out more similar projects, look around my blog and stay tuned for more!


