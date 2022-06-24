#!/usr/bin/env python
# coding: utf-8

# In[28]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import r2_score, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy import stats
from mlxtend.plotting import scatterplotmatrix
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


# In[29]:


raw_data = pd.read_csv('train.csv', sep=',', index_col=0) # Naming the train data "raw_data"
test_data = pd.read_csv('test.csv', sep=',', index_col=0) # Naming the test data "test_data"


# In[30]:


print(raw_data.isnull().sum()) # Checking how many NaN there are
raw_data.head()


# In[19]:


# Histograms below

raw_data.hist(figsize=(20, 20))
plt.tight_layout()
plt.show()


# In[10]:


# Pairplots below

sns.pairplot(raw_data, hue='Loan_Status')


# ### Data cleaning

# In[31]:


# Using get_dummies to convert categorical values to numerical values

le = LabelEncoder()

raw_data.Education = le.fit_transform(raw_data.Education)
raw_data.Dependents = le.fit_transform(raw_data.Dependents)
raw_data.Property_Area = le.fit_transform(raw_data.Property_Area)

dummies = pd.get_dummies(raw_data[['Gender', 'Married', 'Self_Employed','Loan_Status']], drop_first=True)
dummies2 = pd.get_dummies(test_data[['Gender', 'Married', 'Self_Employed']], drop_first=True)

enc_data = pd.concat([raw_data, dummies], axis=1) # Merging / concatenating two dataframes
enc_data = enc_data._get_numeric_data()
enc_data

# imp = SimpleImputer(missing_values=np.NaN, strategy='median')
# imp_data = imp.fit_transform(raw_data.values)
# imp_df = pd.DataFrame(data=imp_data, columns=raw_data.columns)

# imp_df.head() # Checking if the imputer worked


# In[32]:


imp = SimpleImputer(missing_values=np.NaN, strategy='median')
imp_data = imp.fit(enc_data.values)
imp_data = imp.transform(enc_data.values)
imp_df = pd.DataFrame(imp_data, columns=enc_data.columns)

imp_df.head() # Checking if the imputer worked


# In[33]:


test_data.Education = le.fit_transform(test_data.Education)
test_data.Dependents = le.fit_transform(test_data.Dependents)
test_data.Property_Area = le.fit_transform(test_data.Property_Area)

enc_test = pd.concat([test_data, dummies2], axis=1) # Merging / concatenating two dataframes
enc_test = enc_test._get_numeric_data()
enc_test


# In[34]:


imp2 = SimpleImputer(missing_values=np.NaN, strategy='median')
imp_test = imp2.fit(enc_test.values)
imp_test = imp2.transform(enc_test.values)
imp_test_df = pd.DataFrame(imp_test, columns=enc_test.columns)

imp_test_df.head() # Checking if the imputer worked


# ### Checking if there are any NaN values:

# In[35]:


print(imp_df.isna().sum()) # Checking if the NaN values are gone from the training data
print(f'\n{imp_test_df.isna().sum()}\n') # Checking if the NaN values are gone from the test data
print(f'\nThe shape of the imputed training data is: {imp_df.shape}\n')
print(f'\nThe shape of the imputed testing data is: {imp_test_df.shape}\n')


# In[25]:


imp_df.info() # Checking the data types, should be all float


# In[26]:


imp_test_df.info() # Checking the data types, should be all float


# In[36]:


imp_df.to_csv("imputed_new.csv", index=False)
imp_test_df.to_csv("imputed_test_new.csv", index=False)


# ### Visualisation after cleaning

# In[12]:


# Histograms

imp_df.hist(figsize=(20, 20))
plt.tight_layout()
plt.show()


# In[22]:


# Pairplot

sns.pairplot(imp_df, hue='Loan_Status_Y')


# ### Correlation Matrix

# In[13]:


# Correlation matrix

plt.figure(figsize=(15, 15)) # Making it big for easier inspection
corr_matrix = imp_df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()


# We see that loan status positively correlates with both applicant income and credit history. Loan amount also correlates with both applicant income and credit history.
# Something to note is that married males also positively correlate higher with loan status than unmarried males.
# We have to look at that with skepticism since gender should not be something that makes it easier or harder to get a loan.

# ### Preprocessing

# In[14]:


X = imp_df.drop(['Loan_Status_Y'], axis=1)
y = imp_df['Loan_Status_Y']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### Random Forest Pipeline

# In[61]:


forest = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=21))

param_dist = {
    'randomforestclassifier__n_estimators': [50, 100, 200, 300],
    'randomforestclassifier__max_depth': [2, 4, 6, 8, 10],
    'randomforestclassifier__min_samples_leaf': [1, 5, 10],
    'randomforestclassifier__max_features': ['auto', 'sqrt'],
    'randomforestclassifier__bootstrap': [True, False]
}

forest_gs = GridSearchCV(estimator=forest, param_grid=param_dist, scoring='r2', cv=5, n_jobs=-1)


# In[62]:


forest_gs.fit(X_train, y_train)


# In[63]:


best_params = forest_gs.best_params_

print('The best parameters achieved from the grid search are: ', best_params)


# In[64]:


forest_best =  forest_gs.best_estimator_
forest_best.fit(X_train, y_train)


# In[65]:


y_pred = forest_best.predict(imp_test_df)

print('Forest training data accuracy: {0:.2f}'.format(forest_best.score(X_train, y_train)))

print('Forest test data accuracy: {0:.2f}'.format(forest_best.score(X_test, y_test)))


# ### Kernel Pipeline

# In[18]:


svc_pipe = make_pipeline(StandardScaler(), SVC(random_state=21))

param_dist = [{'svc__C': np.arange(5, 7, 1),
'svc__kernel':['rbf', 'linear'],
'svc__gamma': ['scale', 'auto']},
              {'svc__C': np.arange(5, 7, 1),
               'svc__kernel':['linear']},]

g_search_kernel = GridSearchCV(estimator=svc_pipe,
                                     param_grid=param_dist,
                                     scoring='r2',
                                     cv=10, 
                                     n_jobs=-1)


# In[19]:


g_search_kernel.fit(X_train, y_train) # Finding the best parameters for SVC Kernel


# In[20]:


best_params = g_search_kernel.best_params_
print('The best parameters achieved from the grid search are: ', best_params)


# In[21]:


kernel_best =  g_search_kernel.best_estimator_
kernel_best.fit(X, y) # Fitting the best parameters into the model


# In[23]:


y_pred = kernel_best.predict(imp_test_df)

print('Kernel training data accuracy: {0:.2f}'.format(kernel_best.score(X_train, y_train)))

print('Kernel test data accuracy: {0:.2f}'.format(kernel_best.score(X_test, y_test)))


# ### Gradient Boosting Classifier

# In[16]:


gbc_pipe = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=21))

param_dist = [{'gradientboostingclassifier__learning_rate': [0.01, 0.02, 0.05, 0.1],
'gradientboostingclassifier__n_estimators':[500],
'gradientboostingclassifier__min_samples_leaf': [1, 3, 9],
'gradientboostingclassifier__max_depth':[5, 6, 7],
'gradientboostingclassifier__max_features':[0.3,0.6,1.0]}]

g_search_gbc = GridSearchCV(estimator=gbc_pipe,
                                     param_grid=param_dist,
                                     scoring='r2',
                                     cv=10, 
                                     n_jobs=-1)


# In[17]:


g_search_gbc.fit(X_train, y_train) # Finding the best parameters for GBC


# In[18]:


best_params = g_search_gbc.best_params_
print('The best parameters achieved from the grid search are: ', best_params)


# In[19]:


gbc_best =  g_search_gbc.best_estimator_
gbc_best.fit(X, y) # Fitting the best parameters into the model


# In[20]:


y_pred = gbc_best.predict(imp_test_df)

print('Gradient boosting regressor training data accuracy: {0:.2f}'.format(gbc_best.score(X_train, y_train)))

print('Gradient boosting regressor test data accuracy: {0:.2f}'.format(gbc_best.score(X_test, y_test)))


# In[ ]:



confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
 for j in range(confmat.shape[1]):
    ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()

