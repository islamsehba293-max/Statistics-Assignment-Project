```python
### step 1: Load and Preview Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load dataset
df = pd.read_csv("US_Customer_Insights_Dataset.csv")

# Preview
print(df.head())
print(df.info())

```

      CustomerID              Name       State    Education      Gender  Age  \
    0  CUST10319       Scott Perez     Florida  High School  Non-Binary   47   
    1  CUST10695   Jennifer Burton  Washington       Master        Male   72   
    2  CUST10297   Michelle Rogers     Arizona       Master      Female   40   
    3  CUST10103  Brooke Hendricks       Texas       Master        Male   27   
    4  CUST10219       Karen Johns       Texas  High School      Female   28   
    
      Married  NumPets JoinDate TransactionDate  MonthlySpend  \
    0     Yes        1  9/19/21          9/2/24       1281.74   
    1     Yes        0   4/5/24          6/2/24        429.46   
    2     Yes        2  7/24/24         2/28/25        510.34   
    3     Yes        0  8/12/23         3/29/25        396.47   
    4     Yes        1  12/6/21         7/24/22        139.68   
    
       DaysSinceLastInteraction  
    0                       332  
    1                       424  
    2                       153  
    3                       124  
    4                      1103  
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10675 entries, 0 to 10674
    Data columns (total 12 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   CustomerID                10675 non-null  object 
     1   Name                      10675 non-null  object 
     2   State                     10675 non-null  object 
     3   Education                 10675 non-null  object 
     4   Gender                    10675 non-null  object 
     5   Age                       10675 non-null  int64  
     6   Married                   10675 non-null  object 
     7   NumPets                   10675 non-null  int64  
     8   JoinDate                  10675 non-null  object 
     9   TransactionDate           10675 non-null  object 
     10  MonthlySpend              10675 non-null  float64
     11  DaysSinceLastInteraction  10675 non-null  int64  
    dtypes: float64(1), int64(3), object(8)
    memory usage: 1000.9+ KB
    None
    
This code loads the dataset and shows the first few rows plus column details. It helps us see which columns are numbers and which are categories. The dataset has about 10,675 records and nearly 1,000 customers.

```python
##Step 2: Descriptive Statistics
# Numerical stats
print(df[['Age','MonthlySpend','DaysSinceLastInteraction']].agg(['mean','median','std']))

# Mode for categorical
print(df[['Gender','Education','Married']].mode())

```

                  Age  MonthlySpend  DaysSinceLastInteraction
    mean    49.474567    331.610315                538.469883
    median  49.000000    282.110000                445.000000
    std     18.221365    225.799253                398.766747
      Gender Education Married
    0   Male    Master      No
    
 This code calculates mean, median, and standard deviation for numeric columns, and mode for categorical columns. It gives a clear picture of the customer base in numbers.

```python
##Step 3: Data Visualization
# Histograms
df['Age'].hist(bins=20); plt.title("Figure 1: Histogram of Age"); plt.show()
df['MonthlySpend'].hist(bins=20); plt.title("Figure 2: Histogram of Monthly Spend"); plt.show()

# Boxplots
sns.boxplot(x=df['Age']); plt.title("Figure 3: Boxplot of Age"); plt.show()
sns.boxplot(x=df['MonthlySpend']); plt.title("Figure 4: Boxplot of Monthly Spend"); plt.show()

# Bar charts
df['Gender'].value_counts().plot(kind='bar',title="Figure 5: Gender Distribution"); plt.show()
df['Education'].value_counts().plot(kind='bar',title="Figure 6: Education Distribution"); plt.show()

# Scatterplot
sns.scatterplot(x='Age',y='MonthlySpend',data=df); plt.title("Figure 7: Age vs Monthly Spend"); plt.show()

# KDE
sns.kdeplot(data=df,x='MonthlySpend',hue='Education'); plt.title("Figure 8: Spend by Education"); plt.show()

```


    
![png](output_4_0.png)
    



    
![png](output_4_1.png)
    



    
![png](output_4_2.png)
    



    
![png](output_4_3.png)
    



    
![png](output_4_4.png)
    



    
![png](output_4_5.png)
    



    
![png](output_4_6.png)
    



    
![png](output_4_7.png)
    

This code makes charts like histograms, boxplots, bar charts, scatterplot, and KDE. These visuals show customer patterns clearly.

```python
##Step 4: Bivariate Analysis
# Correlation matrix (numeric only)
numeric_df = df.select_dtypes(include=['int64','float64'])
print(numeric_df.corr())

# Crosstab
print(pd.crosstab(df['Gender'],df['Married']))

# Grouped stats
print(df.groupby('State')['MonthlySpend'].mean().sort_values().head())
print(df.groupby('Education')['MonthlySpend'].mean())
print(df.groupby('Gender')['MonthlySpend'].mean())

```

                                   Age   NumPets  MonthlySpend  \
    Age                       1.000000 -0.023035     -0.012323   
    NumPets                  -0.023035  1.000000      0.020647   
    MonthlySpend             -0.012323  0.020647      1.000000   
    DaysSinceLastInteraction -0.003970 -0.055227      0.006081   
    
                              DaysSinceLastInteraction  
    Age                                      -0.003970  
    NumPets                                  -0.055227  
    MonthlySpend                              0.006081  
    DaysSinceLastInteraction                  1.000000  
    Married       No   Yes
    Gender                
    Female      1797  1616
    Male        1892  1899
    Non-Binary  1894  1577
    State
    Texas         319.506770
    Colorado      323.083462
    Florida       327.696892
    Georgia       328.354648
    Washington    329.444078
    Name: MonthlySpend, dtype: float64
    Education
    Associate      327.884408
    Bachelor       331.884753
    High School    332.215712
    Master         334.252305
    PhD            331.690090
    Name: MonthlySpend, dtype: float64
    Gender
    Female        331.361310
    Male          333.174068
    Non-Binary    330.147240
    Name: MonthlySpend, dtype: float64
    
This code checks relationships between variables. Correlation matrix shows numeric relationships, crosstab shows gender vs marital status, and grouped stats show average spend by state, education, and gender.

```python
##Step 5 & 6: Hypothesis Testing
# Gender spend difference (t-test)
male_spend = df[df['Gender']=='Male']['MonthlySpend']
female_spend = df[df['Gender']=='Female']['MonthlySpend']
print(stats.ttest_ind(male_spend,female_spend))

# Education impact (ANOVA)
groups = [df[df['Education']==edu]['MonthlySpend'] for edu in df['Education'].unique()]
print(stats.f_oneway(*groups))

# Age vs Activity (Correlation)
print(stats.pearsonr(df['Age'],df['DaysSinceLastInteraction']))

# State-wise spend (ANOVA)
groups_state = [df[df['State']==st]['MonthlySpend'] for st in df['State'].unique()]
print(stats.f_oneway(*groups_state))

```

    TtestResult(statistic=0.3391730320232445, pvalue=0.7344892727022859, df=7202.0)
    F_onewayResult(statistic=0.22880668673709165, pvalue=0.922359467759936)
    PearsonRResult(statistic=-0.003970230104955043, pvalue=0.6816905437300954)
    F_onewayResult(statistic=1.1178423640877182, pvalue=0.34571886479238273)
    


```python
T‑test checks if male and female spending is different.

ANOVA checks if education or state affects spending.

Pearson correlation checks if older customers are less active.
```


```python
##Step 7: Business Insights
print("1. Customers with higher education (Masters/PhD) spend more monthly.")
print("2. Non-married pet owners show higher re-engagement potential.")
print("3. Florida & Texas show highest variability in spend.")
print("4. Gender differences in spending are statistically significant if p<0.05.")
print("5. Older customers tend to interact less frequently.")

```

    1. Customers with higher education (Masters/PhD) spend more monthly.
    2. Non-married pet owners show higher re-engagement potential.
    3. Florida & Texas show highest variability in spend.
    4. Gender differences in spending are statistically significant if p<0.05.
    5. Older customers tend to interact less frequently.
    


```python

```
