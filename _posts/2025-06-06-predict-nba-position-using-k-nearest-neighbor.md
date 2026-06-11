---
layout: post
title: Predict NBA Position Using K-nearest Neighbor
date: 2025-06-06 15:00:00
description: Predict NBA position from height and weight using a K-nearest neighbor
toc:
    sidebar: left
tags: machine-learning nba
categories: machine-learning-projects
---

In this project, we will use K-nearest Neighbor (KNN) to predict an NBA player’s position using their height and weight. A key difference from a similar project I did previously using K-means clustering is that, unlike K-means, KNN is a supervised learning algorithm. Thus, this time, our data will have labels rather than having to infer the result, like the last time. For more details and code, check out the [Google Colab notebook](https://colab.research.google.com/drive/13-itv_icjHxi0bd-QhIonL55Rwno2ZpA?usp=sharing) of this project to learn more.

<br>

# 1. Project Setup

Based on NBA players’ height and weight, we will predict their position using a KNN model.

<br>

# 2. Data Preparation

For this project, we will use the [NBA Players data (1950 to 2022)](https://www.kaggle.com/datasets/blitzapurv/nba-players-data-1950-to-2021) from Kaggle, which provides player biological data and their respective position. Note that we cannot use the dataset we used last time, as it did not have labels (position).

One of the first data cleaning issues is that the height data is formatted as strings and in feet and inches, e.g., `6-10` or `7-2`. As such, we will need to convert it into inches. This is done by writing a custom function that converts a height string into inches (float), and then using Pandas’ `apply` method to convert the data.

```python
def convert_height(height):
  feet, inches = height.split('-')
  feet = float(feet)
  inches = float(inches)
  res = feet * 12 + inches
  return res
```

```python
df['Ht'] = df['Ht'].apply(convert_height)
df.head()
```

The next problem is the label. The positions are formatted as strings, labelling `G` as guard, `F` as forward, and `C` as center. Many player also gets a secondary position, e.g., a player with the position `C-F` plays center as their primary position and forward as their secondary position.

We will resolve this by creating a `dict` encoding positions (in my case, I used 0 for `G`, 1 for `F`, and 2 for `C`) and once again use the `apply` method. Before we do that, however, we will just take the first letter of the position, essentially only using their primary position for simplicity.

```python
df['Pos'] = df['Pos'].apply(lambda x: x[0])

pos_dict = {
    'G': 0,
    'F': 1,
    'C': 2
}

df['Pos'] = df['Pos'].apply(lambda x: pos_dict[x])
```

And now, just like last time, we scale the data as they range on different scales. We will use Standardization since KNN uses Euclidean distance. After this, using Pandas’ `describe` method should show that the `Ht` and `Wt` column have a mean of 0 and a standard deviation of 1.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Ht', 'Wt']] = scaler.fit_transform(df[['Ht', 'Wt']])
```

Last but not least, we split the data into the standard 80% train and 20% test data.

```python
from sklearn.model_selection import train_test_split

X = df[['Ht', 'Wt']]
y = df[['Pos']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

<br>

# 3. Modeling 

In this project, I will implement 2 slightly different models, with the main difference being that one gives more weights to closer neighbors, and see if that improves our accuracy.

<br>

### Model 1

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
```

```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'{round(accuracy*100, 2)}')
```

The first model has an accuracy of 75.48%. Now let’s see how the second model does.

<br>

### Model 2

And we implement the same thing.

```python
model_2 = KNeighborsClassifier(n_neighbors=5, weights='distance')
model_2.fit(X_train, y_train)

y_pred = model_2.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'{round(accuracy*100, 2)}')
```

This model has an accuracy of 76.38%, a 0.9% increase. Not too significant in my opinion, but worth a try at least.

Now, whichever model you use, you can try out the model with your favorite player and see what it returns!

```python
player = pd.DataFrame({
    'Ht': [84],
    'Wt': [280]
})

model.predict(player)
```

<br>

# 4. Final Words

While you're here, this is part 3 of a machine learning project series where I apply every machine learning algorithm I learned to an NBA-related project. If you want to check out more similar projects, look around my blog and stay tuned for more!

