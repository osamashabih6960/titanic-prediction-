#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
d = {'a':1,'b':2,"c":5,"f":4}
pd.Series(d)


# In[2]:


d = {'a':1,'b':2,"c":5,"f":4}
ser = pd.Series(data = d,index = ['a','b','c','d'])
ser


# In[3]:


#indexing..
cities = ['kolkata','mumbai','tornato','lisbon']
populations = [14.56,2.61, 2.93,0.51]
city_series = pd.Series(populations,index = cities)
city_series.index


# In[4]:


import pandas as pd
import numpy as np
dates = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8,4), index = dates, columns = ['A','B','C','D'])
df


# In[5]:


s = df['A']
s[dates[5]]


# In[6]:


df


# In[7]:


df[['B','A']] = df[['A','B']]
df


# In[8]:


df[['A','B']]


# In[9]:


df[['A','B']]


# In[10]:


#swap,,,col in using row values..
df.loc[:]




# In[ ]:





# In[13]:


#rename..
import pandas as pd
d = {'col1': [1,2,3,4,5], 'col2':[5,4,3,2,1],'col3':[9,8,7,6,5]}
df = pd.DataFrame(data=d)
print("Original Dataframes")
print(df)
df.columns = ['Column1', 'Column2', 'Column3']
df = df.rename(columns={'col1': 'Column1','col2':'Column2', 'col3':'Column3'})
print("New dataframe after renaming columns:")
print(df)


# In[15]:


#select row..
import pandas as pd
import numpy as np
d = {'col1': [1,2,3,4,5], 'col2':[5,4,3,2,1],'col3':[9,8,7,6,5]}
df = pd.DataFrame(data=d)
print("Original Dataframes")
print(df)
print(df.loc[df['col1']==4])
print(df)



# In[16]:


#inter col..
import pandas as pd
d = {'col1': [1,2,3,4,5], 'col2':[5,4,3,2,1],'col3':[9,8,7,6,5]}
df = pd.DataFrame(data=d)
print("Original Dataframes")
print(df)
df = df[['col3','col2','col1']]
print(df)



# In[17]:


#add data
import pandas as pd
d = {'col1': [1,2,3,4,5], 'col2':[5,4,3,2,1],'col3':[9,8,7,6,5]}
df = pd.DataFrame(data=d)
print("Original Dataframes")
print(df)
df2 = {'col1':10,'col2':11,'col3':12}
df = df.append(df2, ignore_index=True)
print(df)



# In[18]:


#using tab operator,,.
import pandas as pd
d = {'col1': [1,2,3,4,5], 'col2':[5,4,3,2,1],'col3':[9,8,7,6,5]}
df = pd.DataFrame(data=d)
print("Original Dataframes")
print(df)
df.to_csv('new_file.csv', sep= '\t', index = False)
new_df = pd.read_csv('new_file.csv')
print(new_df)




# In[1]:


print('osama')


# In[1]:


print("osama")


# In[8]:


import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")


# In[9]:


df


# In[10]:


df.columns


# In[11]:


df.head(4)


# In[12]:


df.tail(2)


# In[13]:


df.dtypes


# In[14]:


df.info()


# In[15]:


df


# In[16]:


df.describe() #gives numerical column.


# In[20]:


df['Survived']


# In[21]:


df.dtypes == 'object'


# In[23]:


df.dtypes[df.dtypes == 'object'] #col name 


# In[24]:


df.dtypes[df.dtypes == 'object'].index 


# In[25]:


df.dtypes[df.dtypes != 'object'].index 


# In[26]:


df[df.dtypes[df.dtypes != 'object'].index].describe()


# In[27]:


df.describe()


# In[29]:


df.describe(include = 'object')


# In[30]:


df.describe(include = 'all')


# In[31]:


df.describe()#five points summary..


# In[33]:


df.astype


# In[34]:


df.astype('object').describe() #gives obj statotical infor categorical.


# In[35]:


df


# In[36]:


df[0:100]


# In[37]:


df[0:100:5]


# In[38]:


df


# In[39]:


df['new_col'] = "pwskills"


# In[40]:


df


# In[41]:


df['family'] = df['SibSp'] + df['Parch']


# In[42]:


df


# In[45]:


pd.Categorical(df['Pclass']) #differ categories and class..


# In[46]:


pd.Categorical(df['Cabin'])


# In[48]:


df['Cabin'].unique()


# In[49]:


df['Cabin'].value_counts()


# In[50]:


df


# In[51]:


3 #Q.how many passengers less 5 yrs old.
df['Age'] <5


# In[52]:


df[df['Age'] < 5]


# In[53]:


len(df[df['Age'] < 5])


# In[54]:


df[df['Age'] < 5].Name


# In[56]:


list(df[df['Age'] < 5].Name)


# In[57]:


#how may person less 18 yrs old..
list(df[df['Age'] > 18].Name)


# In[58]:


len(df[df['Age'] > 18].Name)


# In[59]:


df['Fare']


# In[60]:


df['Fare'].mean()


# In[61]:


df['Fare'] < 32.20


# In[62]:


df[df['Fare'] < 32.20]


# In[63]:


list(df['Fare'] < 32.20)


# In[64]:


len(df['Fare'] < 32.20)


# In[65]:


#how many passengers paid for zero,,
df['Fare'] == 0


# In[67]:


df[df['Fare'] == 0]


# In[68]:


len(df[df['Fare'] == 0])


# In[69]:


list(df['Fare'] == 0)


# In[70]:


df[df['Fare'] == 0].Name


# In[71]:


df[df['Sex']== "male"]


# In[72]:


len(df[df['Sex']== "male"])


# In[73]:


len(df[df['Sex']== "female"])


# In[76]:


df['Sex'].value_counts(normalize = True)


# In[78]:


#how many passengers are one classs.
df['Pclass'] == 1


# In[79]:


df[df['Pclass'] == 1]


# In[80]:


df[df['Survived'] == 1]


# In[81]:


len(df[df['Survived'] == 1])


# In[82]:


df['Survived'].value_counts(normalize = True)


# In[84]:


#how many female more than avg fare..
df['Sex'] == 'Female'


# In[85]:


df['Fare'].mean()


# In[89]:


df[(df['Sex'] == 'female')  &  (df['Fare']> df['Fare'].mean())]


# In[90]:


len(df[(df['Sex'] == 'female')  &  (df['Fare']> df['Fare'].mean())])


# In[91]:


len(df[(df['Sex'] == 'male')  &  (df['Fare']> df['Fare'].mean())])


# In[92]:


import numpy as np
np.mean(df.Fare)


# In[93]:


df.Fare.mean()


# In[96]:


max(df.Fare)


# In[97]:


df['Fare'] == max(df.Fare)


# In[98]:


df[df['Fare'] == max(df.Fare)]


# In[99]:


df.Age


# In[101]:


(df.Age > 18 ) & (df['Survived'] == 1)


# In[102]:


df[(df.Age > 18 ) & (df['Survived'] == 1)]


# # slicing - access

# In[103]:


#implicit index- internal index...
df[0:100]


# In[104]:


#explicit index or name index..
df.iloc # iimplicit index 0 to 2 index


# In[105]:


df.iloc[0:2]


# In[106]:


df


# In[107]:


#named.
df.loc[0:2]


# In[ ]:


#df.iloc[0:2,['Name','Parch']] # error.


# In[108]:


df.loc[0:2,['Name','Parch']]


# In[109]:


df.iloc[0:2, 3:6] #3 sae 6 tkk chaiye. 


# In[110]:


#second to 4 name chaiye..
df['Name'][2:5]


# In[111]:


pd.Series(list(df['Name'][2:5]), index = ['a','b','c'])


# In[118]:


s = pd.Series(list(df['Name'][2:5]))
s


# In[119]:


"pw" + "skills"


# In[120]:


s+s1


# In[121]:


#index wise concat..
s1 = pd.Series(list(df['Name'][5:8]), index = ['a','b','c'])
s1


# In[122]:


s


# In[123]:


s + s1


# In[124]:


df.drop('PassengerId',axis = 1)


# In[125]:


df


# In[129]:


df.drop('PassengerId', axis = 1, inplace = True)


# In[130]:


df


# In[131]:


#drop rows wise..
df.drop(1, inplace = True)


# In[132]:


df


# In[133]:


df.reset_index(drop = True)


# In[134]:


df.set_index('Name')


# In[135]:


#original one.
df.set_index('Name', inplace = True)


# In[136]:


df


# In[138]:


df.loc[Johnston, Miss. Catherine Helen "Carrie"]


# In[139]:


df.reset_index(inplace = True)


# In[140]:


df


# In[142]:


d = {'a':[2,3,4,5],
    'b':[4,5,6,7],
    'c':[2,3,4,5]}


# In[143]:


d


# In[144]:


pd.DataFrame(d)


# In[145]:


df1 = pd.read_csv("customers-100.csv")


# In[146]:


df1


# In[147]:


df1.shape


# In[148]:


df1.describe()


# In[149]:


df1.info()


# In[150]:


df1


# In[151]:


df1.isnull()


# In[152]:


df1.isnull().sum()


# In[153]:


df


# In[154]:


df1


# In[155]:


df1.dropna()    #remove null values..


# In[156]:


df1


# In[157]:


#modify row wise..
df1.dropna()


# In[158]:


#drop col wise..
df1.dropna(axis = 1)


# In[160]:


#one col also null values..
df1[['Customer Id']].dropna(axis = 1)


# In[161]:


#imputation of missing values..
#numeric data..mean and median..
#categorical data..mode..
# impute missing values..> with const and zeros..




# In[162]:


df1


# In[163]:


df1.fillna("somevalue")


# In[164]:


df1.fillna(0)


# In[169]:


df1.parent_id.fillna()


# In[171]:


data1 = {
    'A':[1,2,None,4,5,None,7,8,9,10],
    'B':[None,11,12,13,None,15,16,None,18,19]
}
df2 = pd.DataFrame(data1)
df2


# In[172]:


#access col... some null value.
df2.A


# In[174]:


df2.A.fillna(df2['A'].mean())


# In[175]:


df2.A.fillna(df2['A'].median())


# In[176]:


df2


# In[177]:


df2.fillna(0)


# In[178]:


df2.fillna('something')


# In[179]:


df2.fillna(method = 'ffill') #forward fill-> start seeing rom bottom


# In[180]:


df2.fillna(method = 'bfill') # top to .. fil next


# In[181]:


df2


# In[182]:


df2.duplicated()


# In[183]:


df2.duplicated().sum()


# In[184]:


df.mean()


# In[185]:


df.median()


# In[186]:


df.std()


# In[187]:


df.cov()


# In[188]:


df.Age.describe()


# In[198]:


df


# In[199]:


#why what is every fare was survived.
df[df.Survived == 1]['Fare'].mean()


# In[200]:


#avg fare  didn't servive..
df[df.Survived == 0]['Fare'].mean()


# In[201]:


df[df.Survived == 0]['Age'].mean()


# In[202]:


#group by ...
df.groupby('Survived').mean()


# In[203]:


df.groupby('Survived').mean(numeric_only = True)


# In[204]:


df.groupby('Survived').median(numeric_only = True)


# In[205]:


df.groupby('Survived').sum(numeric_only = True)


# In[206]:


df.groupby('Survived').describe()


# In[207]:


import numpy as np
df.groupby(['Survived'])['Fare'].agg([min,'max','mean','median','count',np.std,'var'])


# In[208]:


df[['Survived','Fare']]


# In[213]:


[image.png]()


# In[214]:


df.groupby


# In[216]:


import pandas as pd
df.groupby(['Sex','Pclass'])['Survived'].sum()


# In[217]:


df.groupby(['Sex','Pclass'])['Survived'].sum().to_frame()


# In[218]:


df.groupby(['Sex','Pclass'])['Survived'].sum().unstack()


# In[219]:


df


# In[221]:


a = df.groupby('Pclass').sum(numeric_only = True)


# In[222]:


a


# In[223]:


a.transpose


# In[224]:


df.head()


# In[225]:


df.head().T


# In[226]:


import pandas as pd
import numpy as np


# In[227]:


df


# In[230]:


#concat..
df1 = df[["Name","Sex","Age"]]


# In[231]:


df1


# In[232]:


df1 = df[["Name","Sex","Age"]][0:5]


# In[233]:


df1


# In[234]:


df2 = df[["Name","Sex","Age"]][5:10]


# In[235]:


df2


# In[236]:


"pw"+"skills"


# In[237]:


pd.concat([df1,df2], axis = 0)


# In[238]:


#col wise concat
pd.concat([df1,df2], axis = 1)


# In[248]:


df2.reset_index(drop = True)


# In[250]:


#merge and join..


# In[269]:


df5 = pd.DataFrame({'a':[1,2,3,45,6],
                    'b':[45,5,7,8,12],
                   'c':[87,8,56,34,67]
}
)


# In[270]:


df5


# In[271]:


df6 = pd.DataFrame({'a':[1,2,3,5,6],
                    'b':[5,5,7,8,2],
                   'c':[7,8,6,4,7]
}
)


# In[272]:


df6


# In[273]:


#merge..
pd.merge(df5,df6, how = 'inner')


# In[274]:


pd.merge(df5,df6, how = 'left')


# In[275]:


pd.merge(df5,df6, how = 'right')


# In[276]:


pd.merge(df5,df6, how = 'outer') #both data frame.


# In[277]:


pd.merge(df5,df6, how = 'cross')


# In[278]:


pd.merge(df5,df6,how = "left", left_on = "a", right_on = "c")


# In[279]:


pd.merge(df5,df6,how = "left", left_on = "a", right_on = "b")


# # join..

# In[280]:


#on basic of index


# In[285]:


v  = pd.DataFrame({'a':[1,2,3,45,6],
                    'b':[45,5,7,8,12],
                   'c':[87,8,56,34,67]},
                   index = ['x','y','z','t','s']
)


# In[286]:


v


# In[289]:


v1  = pd.DataFrame({'p':[1,2,3,45,6],
                    'q':[45,5,7,8,12],
                   'r':[87,8,56,34,67]},
                   index = ['x','y','k','i','m']
)


# In[290]:


v1


# In[292]:


v.join(v1,how = "inner") #common


# In[293]:


v.join(v1,how = "outer")


# In[294]:


v.join(v1,how = "left")


# In[295]:


v.join(v1,how = "cross")


# In[296]:


v.join(v1,how = "right")


# In[297]:


df


# In[298]:


df.Fare #dollar


# In[299]:


df['Fare'].apply


# In[304]:


df['Fare_inr'] = df['Fare'].apply(lambda x:x*90) #ekk print prr apply krni hai data print kae.


# In[305]:


df['Fare_inr']


# In[307]:


len(df['Fare_inr'])


# In[308]:


df


# In[309]:


df['Name_len'] = df['Name'].apply(len)


# In[310]:


df['Name_len']


# In[313]:


def convert(x):
    return x*90


# In[314]:


df['Fare'].apply(convert)


# In[ ]:


def create_flag():
    if x < 10:
        return "cheap"
    elif x >= 10 and x<20:
        return "medium"
    else:
        return "hig"
    


# In[316]:


df['flag_fare'] = df['Fare'].apply(create_flag)


# In[317]:


df


# In[318]:


v


# In[324]:


v.set_index('a', inplace = True)


# In[325]:


v


# In[326]:


v.reset_index(inplace = True)


# In[327]:


v


# In[328]:


v.reindex(['a','g','u','p','r'])


# In[329]:


v.reindex([0,1,2,3,4])


# In[330]:


v


# In[331]:


for i in v.iterrows(): #iteration rows wise..
    print(i,"..............")


# In[332]:


for i in v1.items():#col wise
    print(i)


# In[335]:


def fun_sum(x):
    return x.sum()



# In[336]:


v.apply(func_sum, axis = 0)


# In[2]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




