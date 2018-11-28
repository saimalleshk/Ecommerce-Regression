
## Linear Regression- ECOMMERCE

### Project Description:

You got some contract with an Ecommerce company based in New York City that sells clothing online but also have in-store style and clothing advice sessions.

Customers come in to the store, have sessions/meetings with a personal stylist,that they can go home and order either on a mobile app or website for the clothes they want.

The company is trying to decide whether to focus their efforts on their mobile app experience or their website.
They've hired you on contract to help them figure out!

### Imports


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
%matplotlib inline
```

## Get the Data

We'll work with the Ecommerce Customers csv file from the company. It has Customer info, such as Email,Address and their color Avatar. Then it also has numerical value columns.

- Avg.Session Length: Average session of in-store advice sessions.
- Time on App: Average time spent on App in minutes.
- Time on Website: Average time spent on Website in minutes.
- Length of Membership:How many years the customer has been a member.

**Read in the Ecommerce Customers csv file as a DataFrame called customers.**


```python
customers = pd.read_csv('Ecommerce Customers')
```


```python
customers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Email</th>
      <th>Address</th>
      <th>Avatar</th>
      <th>Avg. Session Length</th>
      <th>Time on App</th>
      <th>Time on Website</th>
      <th>Length of Membership</th>
      <th>Yearly Amount Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mstephenson@fernandez.com</td>
      <td>835 Frank Tunnel\nWrightmouth, MI 82180-9605</td>
      <td>Violet</td>
      <td>34.497268</td>
      <td>12.655651</td>
      <td>39.577668</td>
      <td>4.082621</td>
      <td>587.951054</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hduke@hotmail.com</td>
      <td>4547 Archer Common\nDiazchester, CA 06566-8576</td>
      <td>DarkGreen</td>
      <td>31.926272</td>
      <td>11.109461</td>
      <td>37.268959</td>
      <td>2.664034</td>
      <td>392.204933</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pallen@yahoo.com</td>
      <td>24645 Valerie Unions Suite 582\nCobbborough, D...</td>
      <td>Bisque</td>
      <td>33.000915</td>
      <td>11.330278</td>
      <td>37.110597</td>
      <td>4.104543</td>
      <td>487.547505</td>
    </tr>
    <tr>
      <th>3</th>
      <td>riverarebecca@gmail.com</td>
      <td>1414 David Throughway\nPort Jason, OH 22070-1220</td>
      <td>SaddleBrown</td>
      <td>34.305557</td>
      <td>13.717514</td>
      <td>36.721283</td>
      <td>3.120179</td>
      <td>581.852344</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mstephens@davidson-herman.com</td>
      <td>14023 Rodriguez Passage\nPort Jacobville, PR 3...</td>
      <td>MediumAquaMarine</td>
      <td>33.330673</td>
      <td>12.795189</td>
      <td>37.536653</td>
      <td>4.446308</td>
      <td>599.406092</td>
    </tr>
  </tbody>
</table>
</div>




```python
customers.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Avg. Session Length</th>
      <th>Time on App</th>
      <th>Time on Website</th>
      <th>Length of Membership</th>
      <th>Yearly Amount Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>33.053194</td>
      <td>12.052488</td>
      <td>37.060445</td>
      <td>3.533462</td>
      <td>499.314038</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.992563</td>
      <td>0.994216</td>
      <td>1.010489</td>
      <td>0.999278</td>
      <td>79.314782</td>
    </tr>
    <tr>
      <th>min</th>
      <td>29.532429</td>
      <td>8.508152</td>
      <td>33.913847</td>
      <td>0.269901</td>
      <td>256.670582</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.341822</td>
      <td>11.388153</td>
      <td>36.349257</td>
      <td>2.930450</td>
      <td>445.038277</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>33.082008</td>
      <td>11.983231</td>
      <td>37.069367</td>
      <td>3.533975</td>
      <td>498.887875</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>33.711985</td>
      <td>12.753850</td>
      <td>37.716432</td>
      <td>4.126502</td>
      <td>549.313828</td>
    </tr>
    <tr>
      <th>max</th>
      <td>36.139662</td>
      <td>15.126994</td>
      <td>40.005182</td>
      <td>6.922689</td>
      <td>765.518462</td>
    </tr>
  </tbody>
</table>
</div>




```python
customers.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 8 columns):
    Email                   500 non-null object
    Address                 500 non-null object
    Avatar                  500 non-null object
    Avg. Session Length     500 non-null float64
    Time on App             500 non-null float64
    Time on Website         500 non-null float64
    Length of Membership    500 non-null float64
    Yearly Amount Spent     500 non-null float64
    dtypes: float64(5), object(3)
    memory usage: 31.3+ KB
    

## Exploratory Data analysis

#### Let's explore the data!

For the rest of the exercise we will only be using the numerical data of the csv file.

**Use Seaborn to create a joinplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?**


```python
sns.jointplot(data = customers, x ='Time on Website', y='Yearly Amount Spent')
```

    C:\Users\mallesh\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    




    <seaborn.axisgrid.JointGrid at 0x25f1c596d68>




![png](output_12_2.png)


   **Do the same but with the Time on App column instead.**


```python
sns.jointplot(data=customers,x='Time on App', y='Length of Membership', kind='hex')
```

    C:\Users\mallesh\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    




    <seaborn.axisgrid.JointGrid at 0x25f1f1349e8>




![png](output_14_2.png)



```python
sns.pairplot(customers)
```




    <seaborn.axisgrid.PairGrid at 0x25f1f40d518>




![png](output_15_1.png)


### Based off this plot what looks to be the most correlated feature with yearly amount spent?

### Answer: Length of Membership

---

### Create a linear model plot(using seaborn's Implot) of Yearly Amount Spent vs Length of Membership


```python
sns.lmplot(x ='Length of Membership', y ='Yearly Amount Spent', data=customers)
```

    C:\Users\mallesh\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    




    <seaborn.axisgrid.FacetGrid at 0x25f1fcd95f8>




![png](output_19_2.png)


 ---

## Training and Testing Data

Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.

**Set a Variable X equal to the numerical features of the customers and a variable Y equal to the "Yearly Amount Spent" column.**


```python
customers.columns
```




    Index(['Email', 'Address', 'Avatar', 'Avg. Session Length', 'Time on App',
           'Time on Website', 'Length of Membership', 'Yearly Amount Spent'],
          dtype='object')




```python
Y = customers['Yearly Amount Spent']
```


```python
X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
```

**Use Cross_validation,train_test_split from sklearn to split the data into training and testing sets. Set test size=0.3 and random state=101**


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
```

---

## Training the Model

Now its time to train our model on our trianing data!

**Import LinearRegression from sklearn.linear_model**


```python
from sklearn.linear_model import LinearRegression
```


```python
regressor = LinearRegression()
```

**Train/fit regressor on the training data.**


```python
regressor.fit(X_train, Y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False)



**Print out the coefficients of the model**


```python
regressor.coef_
```




    array([25.98154972, 38.59015875,  0.19040528, 61.27909654])



## Predicting Test Data

Now that we have fit our model, lets evaluate its performance by predicting off the test values!

**Use regressor.predict() to predict off the X_test of the data.**


```python
predictions = regressor.predict(X_test)
```

**Create a Scatterplot of the real test values versus the predicted values**


```python
plt.scatter(Y_test,predictions)
plt.xlabel('Y_test(True Values)')
plt.ylabel('Predicted Values')
```




    Text(0,0.5,'Predicted Values')




![png](output_39_1.png)


## Evaluating the model

Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score(R^2).

**Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.Refer to Wikipedia for the formulas.**


```python
from sklearn import metrics
```


```python
print('MAE', metrics.mean_absolute_error(Y_test, predictions))
print('MSE', metrics.mean_squared_error(Y_test, predictions))
print('RMSE', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
```

    MAE 7.228148653430853
    MSE 79.81305165097487
    RMSE 8.933815066978656
    


```python
metrics.explained_variance_score(Y_test, predictions)
```




    0.9890771231889606



## Residuals

You should have gotten a very good model with a good fit. 
Lets quickly explore the residuals to make sure everything was okay with our data.

**Plot a histogram of the residualas and make sure it looks normally distributed. Use either seaborn displot, or just plt.hist()**


```python
sns.distplot((Y_test-predictions), bins=50)
```

    C:\Users\mallesh\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    




    <matplotlib.axes._subplots.AxesSubplot at 0x25f21d71e80>




![png](output_45_2.png)


## Conclusion

We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important. Let's see if we can interpret the coefficients at all to get an idea.

**Recreate the dataframe below.**


```python
cdf = pd.DataFrame(regressor.coef_, X.columns, columns=['Coeff'])
cdf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coeff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Avg. Session Length</th>
      <td>25.981550</td>
    </tr>
    <tr>
      <th>Time on App</th>
      <td>38.590159</td>
    </tr>
    <tr>
      <th>Time on Website</th>
      <td>0.190405</td>
    </tr>
    <tr>
      <th>Length of Membership</th>
      <td>61.279097</td>
    </tr>
  </tbody>
</table>
</div>




```python
cdf.to_csv('Results.csv')
```

#### How can you interpret these coefficients?

Basically we can interpret one at a time.

For instance, Avg.Session Length for every increase in 1 value there will be increase in 26$ approximately.

Similary for increase in Time on App there will be increase of 38$ spent per year.

Time on Website will be 0.19$ increase  spent per year.

Length of Membership for every increase there will 61.27$ increase per year.

#### Do you think the company should focus more on their mobile app or on their website?

The coefficients showing that We could develop the website to catch up the mobile apps.Since the website needs much efforts than the mobile apps.

Or, we should focus more on mobile app since its already working better. 
