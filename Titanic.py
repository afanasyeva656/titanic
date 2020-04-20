#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


titanic_data = pd.read_csv('train.csv')


# In[3]:


titanic_data.isnull().sum()


# In[4]:


X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
Y = titanic_data.Survived


# In[5]:


#В дереве решений все значения, даже качественные, должны быть представленны в виде чисел
X = pd.get_dummies(X)
X.head(3)


# In[6]:


# При реализации дерева решения необходимо позаботиться, чтобы все NaN были заполнены
X = X.fillna({'Age': X.Age.median()})
X.isnull().sum()


# In[7]:


clf = tree.DecisionTreeClassifier(criterion='entropy')


# In[8]:


clf.fit(X, Y)


# In[9]:


plt.figure(figsize=(100, 25))
tree.plot_tree(clf, fontsize=10, feature_names=list(X), filled=True)


# In[10]:


# Разбиение данных на тест и обучение
from sklearn.model_selection import train_test_split


# In[11]:


# Данные, на которых дерево учится, и данные для теста, равные 33%
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.33,
                                                   random_state=42)


# In[12]:


# Правильность предсказания числа ответов
clf.score(X, Y)


# In[13]:


# Валидация значений на новой выборке
clf.fit(X_train, y_train)
clf.score(X_train, y_train)


# In[14]:


# Проверка правильности предсказания
clf.score(X_test, y_test)


# In[15]:


# Ограничиваем глубину дерева для предсказания закономерностей
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)


# In[16]:


clf.fit(X_train, y_train)
clf.score(X_train, y_train)


# In[17]:


# Улучшение правильности предсказания
clf.score(X_test, y_test)


# In[18]:


max_depth_values = range(1, 100)


# In[19]:


scores_data = pd.DataFrame()


# In[20]:


for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    temp_score_data = pd.DataFrame({'max_depth': [max_depth], 
                                    'train_score': [train_score], 
                                    'test_score': [test_score]})
    
    scores_data = scores_data.append(temp_score_data)


# In[21]:


scores_data.head()


# In[22]:


# Изненение формата ДФ
scores_data_long = pd.melt(scores_data, id_vars=['max_depth'],
                          value_vars=['train_score', 'test_score'],
                          var_name='set_type', value_name='score')


# In[23]:


# Различается динамика обоих скоров
# Тест-скор максиальный при 4-5
# Вместе с учеличением тест-скоре трайн-скор начинает постепенно снижаться
# Начинается переобучение

sns.lineplot(x='max_depth', y='score',
             hue='set_type', data=scores_data_long)


# In[24]:


from sklearn.model_selection import cross_val_score


# In[25]:


clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)


# In[26]:


# Точность классификатора при разбиении на 5 кусочков
# Обучение происходило сначала на 1-4 с предсказанием 5
# Затем обучение происходило на 1-3 и 5 с предсказанием 4 и тд
cross_val_score(clf, X_train, y_train, cv=5).mean()


# In[27]:


max_depth_values = range(1, 100)
scores_data = pd.DataFrame()


# In[28]:


for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    temp_score_data = pd.DataFrame({'max_depth': [max_depth], 
                                    'train_score': [train_score], 
                                    'test_score': [test_score], 
                                    'cross_val_score': [mean_cross_val_score]})
    
    scores_data = scores_data.append(temp_score_data)


# In[40]:


scores_data.head(11)


# In[30]:


scores_data_long = pd.melt(scores_data, id_vars=['max_depth'],
                          value_vars=['train_score', 'test_score', 'cross_val_score'],
                          var_name='set_type', value_name='score')


# In[31]:


scores_data_long.query('set_type == "cross_val_score"').head(15)


# In[32]:


sns.lineplot(x='max_depth', y='score',
             hue='set_type', data=scores_data_long)


# In[52]:


best_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)


# In[53]:


cross_val_score(clf, X_test, y_test, cv=5).mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




