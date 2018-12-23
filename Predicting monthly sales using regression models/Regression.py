
# coding: utf-8

# In[22]:


import pandas as pd
from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

'''Read data'''
orders_df = pd.read_csv('olist_orders_dataset.csv')
cust_df = pd.read_csv('olist_customers_dataset.csv')
order_items_df = pd.read_csv('olist_order_items_dataset.csv')
products_df = pd.read_csv('olist_products_dataset.csv')
prod_trans_df = pd.read_csv('product_category_name_translation.csv')


# In[2]:


orders_df['month'] = pd.DatetimeIndex(orders_df['order_purchase_timestamp']).month
orders_df['year'] = pd.DatetimeIndex(orders_df['order_purchase_timestamp']).year
print(orders_df.head())


# In[3]:


order_month_df = pd.merge(orders_df[['order_id','customer_id','month','year']],cust_df[['customer_id','customer_state']], how = 'inner', on = 'customer_id')
order_month_df.head()


# In[4]:


order_month_df = pd.merge(order_month_df[['order_id','customer_id','month','year','customer_state']],order_items_df[['order_id','product_id','price']], how = 'inner',on = 'order_id')
order_month_df.head()


# In[5]:


order_month_df = pd.merge(order_month_df[['order_id','customer_id','month','year','customer_state','product_id','price']],products_df[['product_id','product_category_name']], how = 'inner', on = 'product_id')
order_month_df.head()


# In[6]:


order_month_df = pd.merge(order_month_df[['order_id','customer_id','month','year','customer_state','product_id','price', 'product_category_name']],prod_trans_df[['product_category_name','product_category_name_english']], how = 'inner', on = 'product_category_name')
order_month_df.head()


# In[7]:




groupby_month_orig = order_month_df.groupby(['year', 'month','customer_state','product_category_name_english'])['price'].sum().reset_index(name = "monthly_sales")
'''for  idx in range(0, len(groupby_month)):
    print(groupby_month.get_group(list(groupby_month.groups)[idx]))'''
groupby_month_orig.head()


# In[8]:


groupby_month_orig.reset_index(inplace = True)
groupby_month_orig=groupby_month_orig.drop(['index'], axis=1)


# In[12]:


groupby_month = pd.DataFrame(groupby_month_orig, columns = ['year','month', 'customer_state','product_category_name_english','monthly_sales' ])

for column in groupby_month_orig.select_dtypes([object]).columns:
    lbl = LabelEncoder() 
    lbl.fit(list(groupby_month_orig[column].values))
    groupby_month[column] = lbl.transform(list(groupby_month_orig[column].values))
groupby_month.head()


# In[13]:




train_x = groupby_month.drop('monthly_sales', axis=1)
train_y = groupby_month.monthly_sales

train_x= train_x.values
train_y = train_y.values


# In[16]:


# Random Forest Regressor

xtrain, xtest, ytrain, ytest = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
k_fold = KFold(n_splits=30) # 80% train split into 10 folds
k_fold.get_n_splits(xtrain)

def train_RF(n_est):
    sum_mae = 0
    sum_rmse = 0
    sum_r2 = 0

    rand_reg = RandomForestRegressor(min_samples_split=3, n_estimators= n_est, min_samples_leaf=2, random_state = 1)

    for train_index, test_index in k_fold.split(xtrain):
        #print('TRAIN:', train_index, 'TEST:', test_index)
        x_train, x_test = xtrain[train_index], xtrain[test_index] # get 9 folds from 80% train and one fold as validation set from the same
        y_train, y_test = ytrain[train_index], ytrain[test_index]

        rand_reg.fit(x_train, y_train)
        preds= rand_reg.predict(x_test)
        #print(mean_absolute_error(y_test, preds))
        sum_mae+=mean_absolute_error(y_test, preds)
        sum_rmse+=sqrt(mean_squared_error(y_test, preds))
        sum_r2+=r2_score(y_test, preds)

    avg_mae = sum_mae/30
    avg_rmse = sum_rmse/30
    avg_r2 = sum_r2/30
    return [avg_mae, avg_rmse, avg_r2]
    #print('MAE : ', avg_mae,'\t MSE:', avg_mse)


# In[17]:


# Decision Tree Regressor 

def train_DTR(depth):
    sum_mae = 0
    sum_mse = 0
    sum_r2 = 0
    sum_rmse = 0

    dt_reg = DecisionTreeRegressor(splitter='best', min_samples_split=6,
                                     min_samples_leaf=2,max_depth=depth, random_state = 1)

    for train_index, test_index in k_fold.split(xtrain):
        #print('TRAIN:', train_index, 'TEST:', test_index)
        x_train, x_test = xtrain[train_index], xtrain[test_index] # get 9 folds from 80% train and one fold as validation set from the same
        y_train, y_test = ytrain[train_index], ytrain[test_index]

        dt_reg.fit(x_train, y_train)
        preds= dt_reg.predict(x_test)
        #print(mean_absolute_error(y_test, preds))
        sum_mae+=mean_absolute_error(y_test, preds)
        sum_r2+=r2_score(y_test, preds)
        sum_rmse+=sqrt(mean_squared_error(y_test,preds))

    avg_mae = sum_mae/30
    avg_rmse = sum_rmse/30
    avg_r2 = sum_r2/30
    joblib.dump(dt_reg, 'dt_reg.pkl')
    return [avg_mae, avg_rmse, avg_r2]
    #print('MAE: ', avg_mae,'\t RMSE:', avg_rmse, '\tR2:',avg_r2)


# In[18]:


# AdaBoost Regressor along with Decision Tree Regressor 

def train_Ada(lr):
    sum_mae = 0
    sum_rmse = 0
    sum_r2 = 0
    dt_reg = joblib.load('dt_reg.pkl')
    ada_dt_reg = AdaBoostRegressor(dt_reg, learning_rate=lr, loss='square', n_estimators=50)

    for train_index, test_index in k_fold.split(xtrain):
        #print('TRAIN:', train_index, 'TEST:', test_index)
        x_train, x_test = xtrain[train_index], xtrain[test_index] # get 9 folds from 80% train and one fold as validation set from the same
        y_train, y_test = ytrain[train_index], ytrain[test_index]

        ada_dt_reg.fit(x_train, y_train)
        preds= ada_dt_reg.predict(x_test)
        #print(mean_absolute_error(y_test, preds))
        sum_mae+=mean_absolute_error(y_test, preds)
        sum_rmse+=sqrt(mean_squared_error(y_test, preds))
        sum_r2+=r2_score(y_test, preds)

    avg_mae = sum_mae/30
    avg_rmse = sum_rmse/30
    avg_r2 = sum_r2/30
    
    return [avg_mae, avg_rmse, avg_r2]
    #print('MAE : ', avg_mae,'\t MSE:', avg_mse, '\t R2:', avg_r2)
    


# In[23]:


# Hyper-parameter tuning using cross-validation

mae_DT = {}
rmse_DT = {}
r2_DT = {}
mae_Ada = {}
rmse_Ada = {}
r2_Ada = {}
mae_RF = {}
rmse_RF = {}
r2_RF = {}
  
for n_est in [25,50,75,100]:
    mae_RF[n_est], rmse_RF[n_est],r2_RF[n_est] = train_RF(n_est)

for depth in [5,10,15,20]:
    mae_DT[depth], rmse_DT[depth],r2_DT[depth] = train_DTR(depth)
    
for lr in [0.001,0.01,0.1,1]:
    mae_Ada[lr], rmse_Ada[lr],r2_Ada[lr] = train_Ada(lr)
  

        


# In[24]:


# Train models using the best parameters

#Random Forest Regressor
rand_reg = RandomForestRegressor(min_samples_split=3, n_estimators= 100, min_samples_leaf=2, random_state = 1)
rand_reg.fit(xtrain, ytrain)

#Decision Tree Regressor
dt_reg = DecisionTreeRegressor(splitter='best', min_samples_split=6,
                                     min_samples_leaf=2,max_depth=20, random_state = 1)
dt_reg.fit(xtrain, ytrain)
joblib.dump(dt_reg, 'dt_reg.pkl')

#AdaBoost Regressor
dt_reg = joblib.load('dt_reg.pkl')
ada_dt_reg = AdaBoostRegressor(dt_reg, learning_rate=0.01, loss='square', n_estimators=50)
ada_dt_reg.fit(xtrain, ytrain)

#save the models
joblib.dump(rand_reg, 'rand_reg.pkl')
joblib.dump(ada_dt_reg, 'ada_dt_reg.pkl')


# In[25]:


#Predicting on test data
mae = {}
rmse = {}
r2 = {}

#AdaBoost Regressor
predict_ada = ada_dt_reg.predict(xtest)
mae['ada'] = mean_absolute_error(ytest, predict_ada)
rmse['ada'] = sqrt(mean_squared_error(ytest, predict_ada))
r2['ada'] = r2_score(ytest, predict_ada)

#Decision Tree Regressor
predict_dtr = dt_reg.predict(xtest)
mae['dtr'] = mean_absolute_error(ytest, predict_dtr)
rmse['dtr'] = sqrt(mean_squared_error(ytest, predict_dtr))
r2['dtr'] = r2_score(ytest, predict_dtr)

#Random Forest Regressor
predict_rf = rand_reg.predict(xtest)
mae['rf'] = mean_absolute_error(ytest, predict_rf)
rmse['rf'] = sqrt(mean_squared_error(ytest, predict_rf))
r2['rf'] = r2_score(ytest, predict_rf)


# In[26]:


import matplotlib.pyplot as plt

plt.suptitle('Evaluation Measure for Decision Tree Classifer')
plt.subplot(3,1,1)
plt.plot(mae_DT.keys(), mae_DT.values(), marker = 'o')
plt.xlabel('Max Depth')
plt.ylabel('MAE')
        
plt.subplot(3, 1, 2)
plt.plot(rmse_DT.keys(), rmse_DT.values(), marker = 'o')
plt.xlabel('Max Depth')
plt.ylabel('RMSE')
        
plt.subplot(3, 1, 3)
plt.plot(r2_DT.keys(), r2_DT.values(), marker = 'o')
plt.xlabel('Max Depth')
plt.ylabel('R2')
plt.show()

plt.suptitle('Evaluation Measure for Ada Boosting Classifer')
plt.subplot(3,1,1)
plt.plot(mae_Ada.keys(), mae_Ada.values(), marker = 'o')
plt.xlabel('Learning Rate')
plt.ylabel('MAE')
        
plt.subplot(3, 1, 2)
plt.plot(rmse_Ada.keys(), rmse_Ada.values(), marker = 'o')
plt.xlabel('Learning Rate')
plt.ylabel('RMSE')
        
plt.subplot(3, 1, 3)
plt.plot(r2_Ada.keys(), r2_Ada.values(), marker = 'o')
plt.xlabel('Learning Rate')
plt.ylabel('R2')
plt.show()

plt.suptitle('Evaluation Measure for Random Forest Classifer')
plt.subplot(3,1,1)
plt.plot(mae_RF.keys(), mae_RF.values(), marker = 'o')
plt.xlabel('Number of estimators')
plt.ylabel('MAE')
        
plt.subplot(3, 1, 2)
plt.plot(rmse_RF.keys(), rmse_RF.values(), marker = 'o')
plt.xlabel('Number of estimators')
plt.ylabel('RMSE')
        
plt.subplot(3, 1, 3)
plt.plot(r2_RF.keys(), r2_RF.values(), marker = 'o')
plt.xlabel('Number of estimators')
plt.ylabel('R2')
plt.show()


# In[47]:


#Data visualization

#Top 10 states with highest sales

states = groupby_month_orig["customer_state"].nunique()
sales = groupby_month_orig.groupby('customer_state')['monthly_sales'].nunique().sort_values(ascending=False)
sales_head= sales.head(10)
plt.figure(figsize=(16,8))
sales_head.plot(kind="bar",rot=0)
plt.title('Top 10 states with highest sales')
plt.xlabel('Customer State')
plt.ylabel('Total sales')


# In[30]:


#Top 5 product categories

months = order_month_df["product_category_name_english"].nunique()
sales = order_month_df.groupby('product_category_name_english')['price'].nunique().sort_values(ascending=False)
sales_head= sales.head(5)


sales_head.plot.pie(subplots = True,figsize=(10, 10), legend = True, startangle=0, textprops={'weight':'bold', 'fontsize':16}, 
                      autopct = '%.2f')
plt.legend(loc = 'upper right')
plt.title("Top 5 Product Categories", fontweight='bold', size=16)
plt.ylabel("")


# In[ ]:


#np.max(predict_ada)


# In[ ]:


#np.min(predict_ada)


# In[ ]:


#np.max(ytest)


# In[ ]:


#np.min(ytest)


# In[46]:


from prettytable import PrettyTable

print('\n\t\tResults of the train data for AdaBoost Regressor with Decision Tree')
t_ada = PrettyTable()         
t_ada.field_names = ['Learning rate','Mean Absolute Error', 'Root Mean Squared Error', 'R2 Score']
t_ada.add_row(['0.001', mae_Ada[0.001], rmse_Ada[0.001], r2_Ada[0.001]])
t_ada.add_row(['0.01', mae_Ada[0.01], rmse_Ada[0.01], r2_Ada[0.01]])
t_ada.add_row(['0.1',mae_Ada[0.1], rmse_Ada[0.1], r2_Ada[0.1]])
t_ada.add_row(['1', mae_Ada[1], rmse_Ada[1], r2_Ada[1]])
print(t_ada)

print('\n\t\t\tResults of the train data for Decision Tree Regressor')
t_dtr = PrettyTable()         
t_dtr.field_names = ['Maximum depth','Mean Absolute Error', 'Root Mean Squared Error', 'R2 Score']
t_dtr.add_row(['5', mae_DT[5], rmse_DT[5], r2_DT[5]])
t_dtr.add_row(['10', mae_DT[10], rmse_DT[10], r2_DT[10]])
t_dtr.add_row(['15',mae_DT[15], rmse_DT[15], r2_DT[15]])
t_dtr.add_row(['20', mae_DT[20], rmse_DT[20], r2_DT[20]])
print(t_dtr)    

print('\n\t\t\tResults of the train data for Random Forest Regressor')
t_rf = PrettyTable()         
t_rf.field_names = ['No. of estimators','Mean Absolute Error', 'Root Mean Squared Error', 'R2 Score']
t_rf.add_row(['25', mae_RF[25], rmse_RF[25], r2_RF[25]])
t_rf.add_row(['50', mae_RF[50], rmse_RF[50], r2_RF[50]])
t_rf.add_row(['75',mae_RF[75], rmse_RF[75], r2_RF[75]])
t_rf.add_row(['100', mae_RF[100], rmse_RF[100], r2_RF[100]])
print(t_rf)    

print('\n\t\t\t\t\tResults of the test data')
x  = PrettyTable()         
x.field_names = ['Model','Mean Absolute Error', 'Root Mean Squared Error', 'R2 Score']
x.add_row(['AdaBoost Regressor with Decision Tree', mae['ada'], rmse['ada'], r2['ada']])
x.add_row(['Decision Tree Regressor', mae['dtr'], rmse['dtr'], r2['dtr']])
x.add_row(['Random Forest Regressor', mae['rf'], rmse['rf'], r2['rf']])
print(x)              

