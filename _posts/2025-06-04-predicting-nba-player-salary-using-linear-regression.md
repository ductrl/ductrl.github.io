---
layout: post
title: Predicting NBA Player Salary Using Linear Regression
date: 2025-06-04 15:00:00
description: Predict NBA salaries from their stats using a linear regression model
toc:
    sidebar: left
tags: machine-learning nba
categories: machine-learning-projects
---

In this project, we will try to predict an NBA player’s salary just from their regular season stats by using a linear regression model. You can have a look at this project’s [Google Colab notebook](https://colab.research.google.com/drive/1OPzUx9T_YxQ4eBxVRwVIsJiUjN7TDRo8?usp=sharing) for more details. 

# 1. Project Setup

Using various individual performance stats, we will predict an NBA player's salary with a linear regression model.

<br>

# 2. Data Preparation

For this project, we will use data set from Kaggle called [NBA Players & Team Data](https://www.kaggle.com/datasets/loganlauton/nba-players-and-team-data?select=NBA+Player+Stats%281950+-+2022%29.csv), which provides multiple datasets, including one for player individual stats and another one for salaries. Since they are separate datasets, we will clean them individually and then merge them.

<br>

### Data cleaning

By taking a look at `df_stats.columns`, we can see that the dataset provides a lot of potentially useful data. However, to keep the input simple enough so that a user can play around with it, the only categories we will use are Age, PPG, APG, RPG, SPG, BPG. Moreover, as player valuation based on stats varies significantly across eras, I will not use data from before 2018, thus ensuring relevancy.

```python
relevant_columns = ['Season', 'Player', 'Age', 'G', 'AST', 'TRB', 'STL', 'BLK', 'PTS']
df_stats = df_stats.filter(items=relevant_columns)
df_stats = df_stats.loc[df_stats['Season'] >= 2018]
```

> ##### TIP
>
> Rather than removing all data before 2018, another alternative is to use the inflation-adjusted salary provided by the dataset. A good reason to do this is that you will end up with much more data than the 1658 I ended up with.
{: .block-tip }

Since the salary dataset only provides data up until 2021, unlike the stats dataset which goes up to 2024, we will first handle mismatching data and empty rows, and then merge them. After that, we can drop all the empty rows (which means dropping data from 2022 to 2024).

The first issue we need to address is that the two datasets have name formatting inconsistencies in special cases: 1) Players with special characters in their names, e.g., D.J. Augustin; 2) Players with name prefixes, e.g., Marvin Bagley III. There are other special cases like JJ Barea, where the salary dataset uses the full unabbreviated name. These cases are too complex to resolve programmatically as they would require manual detection and fixing, so we will just drop them. The naming inconsistencies are addressed with this:

```python
# removing suffixes
suffixes = ['III', 'II', 'IV', 'V']
for suffix in suffixes:
  df_stats['Player'] = df_stats['Player'].str.replace(suffix, '').str.strip()
# removing punctuations
df_stats['Player'] = df_stats['Player'].str.replace('.', '').str.strip()
df_stats['Player'] = df_stats['Player'].str.replace('\'', '').str.strip()
```

Inspecting the salary dataset, we can see that many players do not have a salary value. This is because they are either on a two-way contract or a 10-day contract, or they are a G-League call-up. While we can impute these cases with, for instance, the minimum salary, I will just go with dropping them.

```python
df = df.dropna()
```

Like many monetary datasets, the salaries are also formatted in strings, e.g., *$24,157,304*. This can be addressed pretty easily:

```python
df['salary'] = df['salary'].str.replace('$', '').str.replace(',', '').astype(float)
df.head()
```
<br>


### Feature engineering

The stats dataset does not provide averages in a season, but rather their totals. Thus, we will calculate the average-per-game statistics.

```python
total_stats = ['AST', 'TRB', 'STL', 'BLK', 'PTS']
average_stats = ['APG', 'RPG', 'SPG', 'BPG', 'PPG']
for i in range (5):
  df[average_stats[i]] = df[total_stats[i]] / df['G']
  df[average_stats[i]] = df[average_stats[i]].round(2)
```

<br>

### Data splitting

We will simply go with the classic 80% train and 20% test split using `sklearn`’s `train_test_split`.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
```

<br>

# 3. Modeling

Finally, the good part! As advertised, we will use a linear regression model to predict the salary. 

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

We will evaluate the model using the Mean Absolute Error and R-squared score.

```python
from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_test)
mae = int(mean_absolute_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'On average, the model is off by ${mae}')
print(f'R-squared: {r2}')
```

From the MAE score, we learn that the model is off by about $4.4M on average. Honestly, this is not too bad since I did not expect a linear relationship between player stats and their salaries.

And there we have it! You can try out the model yourself to see how much money it expects your favorite player to make. An example of how to do so:

```python
giannis = pd.DataFrame({
    'Age': [30],
    'APG': [6.5],
    'RPG': [11.9],
    'SPG': [0.9],
    'BPG': [1.2],
    'PPG': [30.4]
})

giannis_pred = model.predict(giannis) / 1000000
print(f'Giannis Antetokounmpo makes ${giannis_pred[0]:.2f}M')
```

After playing around with it, I found that the model works pretty great on role players. Try modifying the input stats or adding new features to see how the predictions change and play around with the model yourself!

<br>

# 4. Final Words

While you're here, this is part 1 of a machine learning project series where I apply every machine learning algorithm I learned to an NBA-related project. If you want to check out more similar projects, look around my blog and stay tune for more!