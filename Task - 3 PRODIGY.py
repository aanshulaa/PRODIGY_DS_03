#!/usr/bin/env python
# coding: utf-8

# Build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. Use a dataset such as the Bank Marketing dataset from the UCI Machine Learning Repository.

# # Bank Term Deposit Prediction

# Problem Statement:
# Build a Decision Tree classifier to predict if the client will subscribe to a Term Deposit based on their demographic and behavioral data.
# (https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

# Dataset
# This dataset is a Bank Marketing dataset from the UCI Machine Learning Repository

# The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

# Column Description:
# age : Integer Age
# job : Occupation
# marital : Marital Status
# education : Education Level
# default : has credit in default?
# balance : average yearly balance in euros
# housing : has housing loan?
# loan : has personal loan?
# contact : contact communication type
# day_of_week : last contact day of the week
# duration : last contact duration, in seconds (numeric)
# campaign : number of contacts performed during this campaign and for this client
# pdays : number of days that passed by after the client was last contacted from a previous campaign(numeric, -1 means client was not previously contacted)
# previous : number of contacts performed before this campaign and for this client
# poutcome : outcome of the previous marketing campaign
# emp.var.rate: employment variation rate - quarterly indicator (numeric)
# cons.price.idx: consumer price index - monthly indicator (numeric)
# cons.conf.idx: consumer confidence index - monthly indicator (numeric)
# euribor3m: euribor 3 month rate - daily indicator (numeric)
# nr.employed: number of employees - quarterly indicator (numeric)
# y : has the client subscribed a term deposit? (outcome)

# In[2]:


#Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# In[3]:


#Loading data
bank = pd.read_csv("bank.csv",sep=';')


# In[4]:


bank.head()


# In[5]:


bank.shape


# Data Pre-processing

# In[6]:


bank.info()


# In[7]:


#Statistical Analysis
bank.describe()


# In[8]:


#Checking Missing values
bank.isnull().sum()


# In[9]:


#Checking for duplicates
bank.duplicated().sum()


# In[10]:


#Investigating these 0 duplicates
bank[bank.duplicated()]


# Exploratory Data Analysis:

# Age Distribution

# In[12]:


sns.histplot(x="age", data=bank, kde=True, hue= "y")
plt.title("Age Distribution and Deposits\n")
plt.show()


# Maximum clients are between the age 30 - 40 and majority clients subscribing to Term Deposit also fall between these age groups.

# Distribution of Occupation:

# In[13]:


plt.figure(figsize=(15,4))
sns.countplot(x="job", data= bank, hue ="y")
plt.title("Occupation Distribution and Deposits\n")
plt.show()


# Majority of clients have occupation "Admin", "blue-collar" or "Technician" and maximum clients subscribing to Term Deposit also are from these occupation.

# Distribution of Marital Status:

# In[14]:


plt.figure(figsize=(7,3))
sns.countplot(x="marital", data= bank, hue ="y")
plt.title("Marital Status and Deposits\n")
plt.show()


# Clients are mostly married

# Distribution of Education Status:

# In[15]:


plt.figure(figsize=(12,4))
sns.countplot(x="education", data= bank, hue ="y")
plt.title("Education Status and Deposits\n")
plt.show()


# Most of the clients have University degree and have studied high or professional course.

# Credit defaults - any ?

# In[16]:


bank.default.value_counts()


# Majority of clients do not have any Credit defaults. Only 3 clients have a Credit default

# Distribution of Housing Loan:

# In[17]:


plt.figure(figsize=(6,3.5))
sns.countplot(x="housing", data= bank, hue ="y")
plt.title("Housing Loan Distribution and Deposits\n")
plt.show()


# Clients subscribing to Term Deposits mostly have Home loan

# Distribution of Personal Loan:

# In[18]:


plt.figure(figsize=(6,3.5))
sns.countplot(x="loan", data= bank, hue ="y")
plt.title("Personal Loan Distribution and Deposits\n")
plt.show()


# Most of the Clients subscribing to Term Deposits dont have personal loan

# Distribution of Outcome (Term Deposits):

# In[19]:


#Count of Outcome
bank.y.value_counts()
keys = bank.y.value_counts().index
data = bank.y.value_counts().values
plt.figure(figsize=(6,3.5))
explode = [0,0.1]
plt.pie(data,labels=keys,explode=explode, autopct='%.0f%%')
plt.show()


# 12% of clients have subscribed to Term Deposit

# Let us now prepare our data for Model building

# Label Encoding Categorical Features:

# In[20]:


cols = bank.select_dtypes("object").columns
cols


# In[21]:


le = LabelEncoder()

bank[cols] = bank[cols].apply(le.fit_transform)


# In[22]:


bank.head(3)


# Correlation Analysis using Heatmap:

# In[23]:


plt.figure(figsize=(23,10))
sns.heatmap(bank.corr(), cmap='bwr', annot=True)
plt.show()


# Outcome "y" is positively correlated with duration of call and also shows slight correlation with previous attempts of call.
# This implies that clients went ahead with Term deposits subscription after gaining sufficient insights from bank calls and persuation.
# 
# multicolinearty can be seen among some input features. This can be handled by dropping those variables or by performing LDA(Linear discriminant analysis) on the data.

# Standardisation:

# In[24]:


#Splitting input and output
X = bank.drop("y", axis=1)
y = bank.y


# In[25]:


scaler = StandardScaler()

X_scaled = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)


# Model building - Decision Tree Classifier:

# In[26]:


#Train-test split
train_X, test_X, train_y, test_y = train_test_split(X_scaled, y, test_size=0.3)


# In[27]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_X, train_y)


# In[28]:


print('Train Score: {}'.format(decision_tree.score(train_X, train_y)))  
print('Test Score: {}'.format(decision_tree.score(test_X, test_y)))    


# In[29]:


cross_val_score(decision_tree, train_X, train_y, cv=5).mean()


# In[30]:


ypred = decision_tree.predict(test_X)
print(classification_report(test_y,ypred))


# Hyperparameter tunning:

# In[31]:


#Applying Grid search cv to find best estimaters to improve model performance

param_grid = {
    'max_depth': [3, 5, 7,10, None],
    'criterion' : ['gini', 'entropy'],
    'min_samples_leaf': [3, 5, 7, 9,10,20]
    }


# In[32]:


gscv = GridSearchCV(decision_tree, param_grid, cv=5, verbose=1)
gscv.fit(train_X, train_y)


# In[33]:


gscv.best_params_


# In[34]:


gscv.best_estimator_


# In[35]:


cross_val_score(gscv.best_estimator_, train_X, train_y, cv=5).mean() 


# we will use this best estimators to built our model.

# In[36]:


clf = DecisionTreeClassifier(criterion= 'gini', max_depth= 5, min_samples_leaf = 3)
clf.fit(train_X, train_y)


# In[37]:


print('Train Score: {}'.format(clf.score(train_X, train_y)))
print('Test Score: {}'.format(clf.score(test_X, test_y)))


# In[38]:


pred_y = clf.predict(test_X)


# In[39]:


#Confusion Matrix
cm = confusion_matrix(pred_y, test_y)
ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot()
plt.show()


# In[40]:


#Classification Report
print(classification_report(pred_y, test_y))


# In[41]:


#Accuracy Score
accuracy = accuracy_score(test_y,pred_y)
print("Test Accuracy of Decision Tree Classifier : {}".format(accuracy*100))


# In[42]:


#Cross Validation Score
Cross_val = cross_val_score(clf, test_X,test_y, cv=5).mean()     
print("Cross-Validation Accuracy Scores Decision Tree : ",Cross_val*100)


# Visualizing the Tree:

# In[43]:


from sklearn import tree
fig = plt.figure(figsize=(25,20))
t= tree.plot_tree(clf,filled=True,feature_names=X.columns)


# Conclusion and Key Insights:
# In this project, we embarked on a journey to build a predictive model for term deposit subscriptions in a banking context to help a Portuguese banking institution tailor its marketing efforts effectively. Let's recap the key takeaways and insights from this endeavor:

# Data Pre-processing:
# 
# The initial data inspection revealed a well-structured dataset with no missing values. While we encountered some duplicate entries, they were not considered errors but rather different campaigns targeting the same clients.

# Exploratory Data Analysis (EDA):
# 
# The age distribution of clients highlighted that the majority fell in the 30-40 age range, coinciding with the highest subscription rate for term deposits.
# Occupation-wise, clients in administrative, blue-collar and technician roles formed the majority and were more likely to subscribe to term deposits.
# Marital status leaned heavily toward the "married" category.
# Clients with university degrees or high-level education were more prominent and had a higher subscription rate.
# Credit defaults were rare, with the vast majority of clients having no credit defaults.
# Clients with housing loans were more likely to subscribe to term deposits, while those with personal loans were less likely.

# Model Building and Evaluation:
# 
# We implemented a Decision Tree classifier to predict term deposit subscriptions.
# The model exhibited strong performance, achieving an accuracy rate of approximately 91.52% on both training and testing data.
# Cross-validation confirmed the robustness of our model, with an accuracy score of around 91.60%.
# The classification report revealed a balanced model with reasonable precision, recall, and F1-scores for both subscribed and non-subscribed clients.

# Insights:
# The duration of calls was a significant predictor of term deposit subscriptions. Clients tended to subscribe after longer conversations, suggesting the importance of engaging clients effectively during calls.
# Previous attempts at contact also played a role, indicating that repeated contact can be fruitful in convincing clients to subscribe.
# Multicollinearity among some input features was observed, potentially affecting model performance. Consideration should be given to addressing this issue in future iterations.

# Future Directions:
# 
# To further enhance predictive accuracy, exploring other machine learning algorithms or ensemble methods could be beneficial. Feature engineering and selection methods could be employed to mitigate multicollinearity. Incorporating external data sources, such as economic indicators or market trends, may provide additional insights and improve model performance. A real-time deployment of the model into the bank's operational systems can automate decision-making and streamline marketing efforts.
# 
# Thank you !
# 
# Project by: Anshula Killamsetty
