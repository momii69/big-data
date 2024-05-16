#!/usr/bin/env python
# coding: utf-8

# In[7]:


# You may need to install below packages.
get_ipython().system('pip install pyspark findspark')


# In[1]:


# It wil set spark environment variable
import findspark
findspark.init()


# In[2]:


import os
import pyspark
import pyspark.sql.types as ptypes
import pyspark.sql.functions as funcs


# In[3]:


ss = pyspark.sql.SparkSession.builder.master('local[*]').appName('Sales').getOrCreate()


# ### Updating matpotlib configuration to change figure size

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.rc = {'figure.figsize': (16, 8)}")


# In[14]:


sales_path = 'C:\\Users\\My Pc\\SalesAnalysis.csv'
sales_data = ss.read.csv(sales_path, inferSchema=True, header=True)


# ### Dropping null rows

# In[17]:


sales_data_dropna = sales_data.dropna()


# ### Cleaning `Order Date` column

# In[18]:


sales_data_dropna = (sales_data_dropna.
                     withColumn('Order Date', funcs.from_unixtime(funcs.unix_timestamp('Order Date', 'MM/dd/yy HH:mm'))))


# In[19]:


sales_data_dropna = (sales_data_dropna
                     .withColumn('hour', funcs.date_format(sales_data_dropna['Order Date'], 'HH').cast(ptypes.IntegerType()))
                     .withColumn('minute', funcs.date_format(sales_data_dropna['Order Date'], 'mm').cast(ptypes.IntegerType()))
                     .withColumn('full_weekday', funcs.date_format(sales_data_dropna['Order Date'], 'EEEE'))
                     .withColumn('month_ordered', funcs.date_format(sales_data_dropna['Order Date'], 'MMM')))


# ### Total number of products sold per month

# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.rc = {'figure.figsize': (16, 8)}")


# In[21]:


(sales_data_dropna.groupby('month_ordered')
 .agg(
     funcs.count(funcs.col('Quantity Ordered')).alias('Quantity Ordered')
 ).toPandas().plot(kind='bar', x='month_ordered', xlabel="Month", ylabel='Quantity')
)


# ### Product sold per month

# In[22]:


sales_data_dropna = sales_data_dropna.withColumn('total_pay', (funcs.col('Price Each') * funcs.col('Quantity Ordered')))
monthly_highest_sales = sales_data_dropna.groupby('month_ordered').sum('total_pay').alias('monthly_total')
monthly_highest_sales = monthly_highest_sales.withColumnRenamed('sum(total_pay)', 'monthly_total')


# In[23]:


(monthly_highest_sales.toPandas()
 .plot(kind='bar', x='month_ordered', xlabel='Month', ylabel='Sales in USD ($)'))


# ### What are the products sold in large amount?

# In[24]:


(sales_data_dropna.groupby('Product')
 .agg({'Quantity Ordered': 'sum'})
 .toPandas().plot(kind='bar', x='Product', xlabel='Products', ylabel='Quantity Ordered'))


# ### Defining `UDF` function to clean address

# In[25]:


def clean_addresss(address):
    city = address.split(',')[1].strip()
    state = address.split(',')[2].split(' ')[1].strip()
    return f'{city} ({state})'

clean_addresss_udf = funcs.udf(lambda addrs: clean_addresss(addrs))


# In[26]:


# Cleaning address
sales_data_dropna = sales_data_dropna.withColumn('city', clean_addresss_udf('Purchase Address'))


# ### Total Sales per city

# In[27]:


(sales_data_dropna.groupBy('city').sum('total_pay')
 .withColumnRenamed('sum(total_pay)', 'total_sales_per_city')
 .toPandas().plot(kind='bar', x='city', xlabel='City', ylabel='Sales in USD ($)'))


# ### Max sales during specific hour

# In[28]:


(sales_data_dropna.groupby('hour')
 .sum('total_pay').withColumnRenamed('sum(total_pay)', 'total_sales_per_hour')
 .sort('hour').toPandas().plot(x='hour', xlabel='Hour', ylabel='Sales in USD ($)')
 .grid())


# ### Find the day of week for entire year which has max product selling.

# In[19]:


(sales_data_dropna.groupby('full_weekday').sum('total_pay')
 .withColumnRenamed('sum(total_pay)', 'total_sales_in_weekdays')
 .toPandas().plot(x='full_weekday', xlabel='Week Days', ylabel='Sales in USD ($)')
 .grid())


# ### What are the product sold on `Tuesday`? and How many of each product sold on `Tuesday`?

# In[20]:


colmns = ['full_weekday']
(sales_data_dropna.groupby('Product').agg(
    funcs.sum(funcs.col('Quantity Ordered')).alias('total_no_of_items_orderd_in_year'),
    *[funcs.first(cl).alias(cl) for cl in colmns]
).where(funcs.col('full_weekday') == 'Tuesday')
 .select('Product', 'total_no_of_items_orderd_in_year')
 .sort('total_no_of_items_orderd_in_year', ascending=False).show())


# ### Check the records which has duplicates Ordered ID's

# In[21]:


multiple_item_id_order = (sales_data_dropna.groupby('Order ID').count().where(funcs.col('count') > 1)
                          .sort('count', ascending=False).withColumnRenamed('count', 'total_item_in_order'))
multiple_item_id_order.count()


# In[22]:


sales_data_dropna_tmp = sales_data_dropna
multiple_item_order_data = multiple_item_id_order.join(sales_data_dropna_tmp, ['Order ID'], 'leftouter')
multiple_item_order_data.count()


# ### What are the products sold togather?

# In[23]:


multiple_item_order_data_grouped = (multiple_item_order_data.groupby('Order ID')
 .agg(
     funcs.concat_ws(", ", funcs.collect_list(funcs.col('Product'))).alias('items_in_a_order'),
     *[funcs.first(cl).alias(cl) for cl in multiple_item_order_data.columns if not cl == 'Order ID']
 )
)


# In[24]:


multiple_item_order_data_grouped.groupby('items_in_a_order').count().sort('count', ascending=False).show(5)


# ### What are the product sold on `MONDAY`? and How many of each product sold on `MONDAY`?

# In[25]:


colmns = ['full_weekday']
(sales_data_dropna.groupby('Product').agg(
    funcs.sum(funcs.col('Quantity Ordered')).alias('total_no_of_items_orderd_in_year'),
    *[funcs.first(cl).alias(cl) for cl in colmns]
).where(funcs.col('full_weekday') == 'Monday')
 .select('Product', 'total_no_of_items_orderd_in_year')
 .sort('total_no_of_items_orderd_in_year', ascending=False).show())


# In[27]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have a SparkSession named 'spark' already created

# Run your PySpark code to extract and process the data
result_df = (sales_data_dropna.groupby('Product')
             .agg(funcs.sum(funcs.col('Quantity Ordered')).alias('total_no_of_items_orderd_in_year'),
                  *[funcs.first(cl).alias(cl) for cl in colmns])
             .where(funcs.col('full_weekday') == 'Monday')
             .select('Product', 'total_no_of_items_orderd_in_year')
             .sort('total_no_of_items_orderd_in_year', ascending=False))

# Convert Spark DataFrame to Pandas DataFrame for plotting
result_pd_df = result_df.toPandas()

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(result_pd_df['Product'], result_pd_df['total_no_of_items_orderd_in_year'])
plt.xlabel('Product')
plt.ylabel('Total Number of Items Ordered')
plt.title('Total Number of Items Ordered for Each Product on Mondays')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# In[31]:


# Assuming you have a SparkSession named 'spark' already created

# Run your PySpark code to extract and process the data
result_df = (sales_data_dropna.groupby('Product')
             .agg(funcs.sum(funcs.col('Quantity Ordered')).alias('total_no_of_items_orderd_in_year'),
                  *[funcs.first(cl).alias(cl) for cl in colmns])
             .select('Product', 'total_no_of_items_orderd_in_year')
             .sort('total_no_of_items_orderd_in_year', ascending=False))

# Convert Spark DataFrame to Pandas DataFrame for plotting
result_pd_df = result_df.toPandas()

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(result_pd_df['Product'], result_pd_df['total_no_of_items_orderd_in_year'])
plt.xlabel('Product')
plt.ylabel('Total Number of Items Ordered')
plt.title('Total Number of Items Ordered for Each Product')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# In[ ]:





# In[32]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have a SparkSession named 'spark' already created

# Run your PySpark code to extract and process the data
result_df = (sales_data_dropna.groupby('Product', 'full_weekday')
             .agg(funcs.sum(funcs.col('Quantity Ordered')).alias('total_no_of_items_ordered'))
             .sort('Product', 'full_weekday'))

# Convert Spark DataFrame to Pandas DataFrame for plotting
result_pd_df = result_df.toPandas()

# List of weekdays in the correct order
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Plotting
for weekday in weekdays:
    # Filter data for the current day
    day_data = result_pd_df[result_pd_df['full_weekday'] == weekday]
    
    # Plot for the current day
    plt.figure(figsize=(10, 6))
    plt.bar(day_data['Product'], day_data['total_no_of_items_ordered'])
    plt.xlabel('Product')
    plt.ylabel('Total Number of Items Ordered')
    plt.title(f'Total Number of Items Ordered for Each Product on {weekday}')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()


# In[33]:


# Assuming you have a SparkSession named 'spark' already created

# Run your PySpark code to extract and process the data
result_df = (sales_data_dropna.groupby('Product', 'full_weekday')
             .agg(funcs.sum(funcs.col('Quantity Ordered')).alias('total_no_of_items_ordered'))
             .sort('Product', 'full_weekday'))

# Convert Spark DataFrame to Pandas DataFrame for plotting
result_pd_df = result_df.toPandas()

# List of weekdays in the correct order
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Plotting
plt.figure(figsize=(12, 6))

# Loop through each day of the week
for weekday in weekdays:
    # Filter data for the current day
    day_data = result_pd_df[result_pd_df['full_weekday'] == weekday]
    
    # Plot for the current day
    plt.plot(day_data['Product'], day_data['total_no_of_items_ordered'], marker='o', label=weekday)

plt.xlabel('Product')
plt.ylabel('Total Number of Items Ordered')
plt.title('Total Number of Items Ordered for Each Product by Day of the Week')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




