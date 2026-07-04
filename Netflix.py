#!/usr/bin/env python
# coding: utf-8

# # ============================================
# # Netflix Movies & TV Shows Analysis
# # Mini Project - 4 (EDA-2)
# # ============================================

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("netflix_titles.csv")
df.head()


# # Task 1 : Dataset Understanding
# ### Shape of Dataset

# In[3]:


df.shape


# ### Number of Rows and Columns

# In[4]:


print("Rows =", df.shape[0])
print("Columns =", df.shape[1])


# ### Column Names

# In[6]:


df.columns


# ### Dataset Information

# In[7]:


df.info()


# ### Data Types

# In[9]:


df.dtypes


# ### Statistical Summary

# In[10]:


df.describe(include="all")


# ### Missing Values

# In[11]:


df.isnull().sum()


# ### Duplicate Values

# In[13]:


df.duplicated().sum()


# ### First 10 Rows

# In[14]:


df.head(10)


# ### Findings
# 
# - The dataset contains **8,807 rows** and **12 columns**, providing information about Netflix Movies and TV Shows.
# 
# - The dataset consists of both **categorical** and **numerical** variables.
# 
# - The **release_year** column is numerical, while most of the remaining columns are categorical.
# 
# - Missing values are present in the **director, cast, country, date_added, rating,** and **duration** columns.
# 
# - No duplicate records were found in the dataset.
# 
# - The dataset is suitable for Exploratory Data Analysis after performing data cleaning.

# # Task 2 : Data Cleaning
# ### Remove Duplicate Values

# In[16]:


df = df.drop_duplicates()


# ### Check Missing Values Again

# In[17]:


df.isnull().sum()


# ### Fill Missing Values of Director

# In[18]:


df["director"] = df["director"].fillna("Unknown")


# ### Fill Missing Values of Cast

# In[19]:


df["cast"] = df["cast"].fillna("Unknown")


# ### Fill Missing Values of Country

# In[20]:


df["country"] = df["country"].fillna(df["country"].mode()[0])


# ### Fill Missing Values of Rating

# In[21]:


df["rating"] = df["rating"].fillna(df["rating"].mode()[0])


# ### Remove Remaining Missing Values

# In[22]:


df = df.dropna()


# ### Check Missing Values

# In[23]:


df.isnull().sum()


# ### Check Shape After Cleaning

# In[24]:


df.shape


# ### Display Clean Dataset

# In[26]:


df.head()


# ### Findings
# 
# - No duplicate records were found in the dataset.
# 
# - Missing values are still present after removing duplicates.
# 
# - The **director** column has the highest number of missing values (**2634**).
# 
# - The **cast** column contains **825** missing values, while the **country** column contains **831** missing values.
# 
# - Very few missing values are present in the **date_added (10)**, **rating (4)**, and **duration (3)** columns.
# 
# - The missing values need to be handled before performing further analysis and visualization.

# # Task 3: Content Type Analysis
# ### Question 1: How many Movies?

# In[27]:


movies = df[df["type"]=="Movie"].shape[0]
print("Number of Movies :", movies)


# ### Question 2: How many TV Shows?

# In[28]:


tvshows = df[df["type"]=="TV Show"].shape[0]
print("Number of TV Shows :", tvshows)


# ### Question 3: Percentage Distribution

# In[29]:


percentage = round(df["type"].value_counts(normalize=True)*100,2)
percentage


# ### Visualization 1: Count Plot

# In[38]:


plt.figure(figsize=(8,5))

sns.countplot(
    x="type",
    data=df,
    hue="type",
    palette="Set2",
    legend=False
)
plt.title("Distribution of Movies and TV Shows")
plt.xlabel("Content Type")
plt.ylabel("Count")
plt.show()


# ### Visualization 2: Pie Chart

# In[43]:


plt.figure(figsize=(7,7))

plt.pie(df["type"].value_counts(),
        labels=df["type"].value_counts().index,
        colors=["royalblue","coral"],
        startangle=90)

plt.title("Percentage Distribution of Movies and TV Shows")
plt.show()


# ### Findings
# 
# - The dataset contains **6,131 Movies** and **2,676 TV Shows**.
# 
# - Movies make up approximately **69.6%** of the total content available on Netflix.
# 
# - TV Shows account for approximately **30.4%** of the total content.
# 
# - The count plot shows that the number of Movies is much higher than the number of TV Shows.
# 
# - The pie chart clearly indicates that Movies occupy the largest share of the Netflix content library.
# 
# - This analysis shows that Netflix offers more Movies than TV Shows, indicating a greater focus on movie content.

# # Task 4 : Country Analysis
# ### Question 1: Top 10 Content-Producing Countries

# In[45]:


top10 = df["country"].value_counts().head(10)
top10


# ### Question 2: Visualize Country Distribution

# In[47]:


plt.figure(figsize=(10,6))

top10.plot(kind="barh",
           color=["mediumpurple","mediumseagreen","coral","gold","skyblue",
                  "tomato","orchid","teal","orange","slateblue"])

plt.title("Top 10 Content-Producing Countries")
plt.xlabel("Number of Titles")
plt.ylabel("Country")

plt.show()


# ### Findings
# 
# - The **United States** is the largest content-producing country on Netflix.
# 
# - **India** is the second-highest contributor to Netflix content.
# 
# - Countries such as the **United Kingdom, Japan, South Korea, Canada, France, Spain, Mexico,** and **Turkey** are also among the top content producers.
# 
# - The bar chart shows that the United States has produced significantly more content than the other countries.
# 
# - This indicates that Netflix's content library is dominated by productions from the United States, followed by contributions from several other countries.

# # Task 5 : Yearly Trend Analysis
# ### Question 1: Number of Titles Released Each Year

# In[48]:


year = df["release_year"].value_counts().sort_index()
year


# ### Question 2: Has Content Production Increased Over Time?

# In[49]:


year.tail(10)


# ### Visualization 1: Line Chart

# In[50]:


plt.figure(figsize=(12,6))
plt.plot(year.index,
         year.values,
         color="mediumslateblue",
         marker="o",
         linewidth=2.5,
         markersize=6)
plt.title("Trend of Netflix Titles Released Over the Years", fontsize=16, fontweight="bold")
plt.xlabel("Release Year", fontsize=12)
plt.ylabel("Number of Titles", fontsize=12)

plt.grid(linestyle="--", alpha=0.5)

plt.show()


# ### Visualization 2: Bar Chart

# In[52]:


plt.figure(figsize=(13,6))

year.plot(kind="bar",
          color="crimson",
          edgecolor="black",
          width=0.8)

plt.title("Year-wise Distribution of Netflix Titles", fontsize=16, fontweight="bold")
plt.xlabel("Release Year", fontsize=12)
plt.ylabel("Number of Titles", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()


# ### Findings
# 
# - The number of Netflix titles released varies across different years.
# 
# - Content production remained relatively low in the earlier years.
# 
# - A significant increase in the number of titles can be observed after the year 2000.
# 
# - The highest number of titles was released during the late 2010s.
# 
# - The line chart and bar chart show an overall increasing trend in content production over time.
# 
# - This indicates that Netflix has expanded its content library significantly over the years.

# # Task 6 : Rating Analysis
# ### Question 1: Most Common Maturity Rating

# In[53]:


df["rating"].value_counts().head(1)


# ### Question 2: Distribution of Ratings

# In[54]:


rating = df["rating"].value_counts()

rating


# ### Question 3: Visualize Using Count Plot

# In[56]:


plt.figure(figsize=(12,6))
sns.countplot(x="rating",
              data=df,
              order=df["rating"].value_counts().index,
              color="mediumorchid")
plt.title("Distribution of Netflix Content Ratings", fontsize=15, fontweight="bold")
plt.xlabel("Rating")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.show()


# ### Findings
# 
# - **TV-MA** is the most common maturity rating in the Netflix dataset.
# 
# - It is followed by **TV-14** and **TV-PG**, indicating that a large portion of Netflix content is intended for teenagers and mature audiences.
# 
# - Ratings such as **G**, **NC-17**, and **UR** have the fewest titles.
# 
# - The count plot shows that mature-content ratings dominate the Netflix library.
# 
# - This suggests that Netflix primarily focuses on content suitable for older viewers rather than children.

# # Task 7 : Genre Analysis
# ### Question 1: Top 10 Genres

# In[57]:


genre = df["listed_in"].str.split(", ", expand=True).stack().value_counts().head(10)

genre


# ### Question 2: Most Popular Genre Category

# In[58]:


genre.head(1)


# ### Question 3: Visualize Findings

# ### Visualization 1

# In[61]:


plt.figure(figsize=(12,6))

plt.plot(genre.index,
         genre.values,
         marker="o",
         color="coral",
         linewidth=2)

plt.title("Top 10 Netflix Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Titles")

plt.xticks(rotation=45)

plt.grid(linestyle="--", alpha=0.5)

plt.show()


# ### Visualization 2

# In[62]:


plt.figure(figsize=(12,6))

genre.sort_values().plot(kind="area",
                         color="mediumseagreen",
                         alpha=0.7)

plt.title("Top 10 Netflix Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)

plt.show()


# ### Findings
# 
# - **International Movies** is the most popular genre on Netflix.
# 
# - It is followed by **Dramas**, **Comedies**, and **International TV Shows**.
# 
# - The remaining genres have comparatively fewer titles.
# 
# - The visualizations show that Netflix offers a wide variety of content, with international and drama-related genres dominating the platform.

# # Task 8 : Duration Analysis
# ### Question 1: Average Movie Duration

# In[63]:


movie = df[df["type"]=="Movie"].copy()

movie["duration"] = movie["duration"].str.replace(" min","").astype(int)

movie["duration"].mean()


# ### Question 2: Longest Movie

# In[64]:


movie[movie["duration"]==movie["duration"].max()][["title","duration"]]


# ### Question 3: Shortest Movie

# In[65]:


movie[movie["duration"]==movie["duration"].min()][["title","duration"]]


# ### Question 4: Distribution of Movie Duration

# In[66]:


movie["duration"].describe()


# ### Visualization 1: Histogram

# In[73]:


plt.figure(figsize=(10,6))

plt.hist(movie["duration"],
         bins=20,
         color="tan",
         edgecolor="black")

plt.title("Distribution of Movie Duration", fontsize=15, fontweight="bold")
plt.xlabel("Duration (Minutes)")
plt.ylabel("Number of Movies")

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.show()


# ### Visualization 2: Box Plot

# In[77]:


plt.figure(figsize=(9,5))

sns.boxplot(x=movie["duration"],
            color="coral",
            showmeans=True)

plt.title("Distribution of Movie Duration", fontsize=15, fontweight="bold")
plt.xlabel("Duration (Minutes)")

plt.grid(axis="x", linestyle="--", alpha=0.5)

plt.show()


# ### Findings
# 
# - The average movie duration is approximately **100 minutes**.
# 
# - The dataset contains movies with both very short and very long durations.
# 
# - The histogram shows that most Netflix movies have a duration between **80 and 120 minutes**.
# 
# - The box plot indicates the presence of a few outliers representing exceptionally long movies.
# 
# - Overall, Netflix primarily offers movies with a standard feature-film duration.

# # Task 9 : Outlier Analysis
# ### Question 1: Detect Outliers Using IQR

# In[78]:


Q1 = movie["duration"].quantile(0.25)
Q3 = movie["duration"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
iqr_outliers = movie[(movie["duration"] < lower) | (movie["duration"] > upper)]
iqr_outliers


# ### Number of Outliers (IQR)

# In[79]:


print("Number of Outliers:", iqr_outliers.shape[0])


# ### Question 2: Detect Outliers Using Z-Score

# In[80]:


mean = movie["duration"].mean()
std = movie["duration"].std()

movie["Z_score"] = (movie["duration"] - mean) / std
z_outliers = movie[(movie["Z_score"] > 3) | (movie["Z_score"] < -3)]
z_outliers


# ### Number of Outliers (Z-Score)

# In[81]:


print("Number of Outliers:", z_outliers.shape[0])


# ### Question 3: Compare Results

# In[82]:


print("IQR Outliers :", iqr_outliers.shape[0])
print("Z-Score Outliers :", z_outliers.shape[0])


# ### Findings
# 
# - The IQR method detected movie durations that fall outside the normal range based on the interquartile range.
# 
# - The Z-score method detected movies whose durations are more than 3 standard deviations away from the average.
# 
# - The IQR method identified more outliers than the Z-score method.
# 
# - This is because the IQR method is more sensitive to unusual values, while the Z-score method mainly detects extreme outliers.
# 
# - Both methods confirm that a few movies have unusually long durations compared to the majority of Netflix movies.

# # Task 10 : Feature Engineering
# ### Question 1: Create Release Decade

# In[86]:


df["Release Decade"] = ""

df.loc[(df["release_year"] >= 1980) & (df["release_year"] <= 1989), "Release Decade"] = "1980s"

df.loc[(df["release_year"] >= 1990) & (df["release_year"] <= 1999), "Release Decade"] = "1990s"

df.loc[(df["release_year"] >= 2000) & (df["release_year"] <= 2009), "Release Decade"] = "2000s"

df.loc[(df["release_year"] >= 2010) & (df["release_year"] <= 2019), "Release Decade"] = "2010s"

df.loc[df["release_year"] >= 2020, "Release Decade"] = "2020s"

df["Release Decade"].value_counts().sort_index()


# ### Question 2: Create Content Age

# In[87]:


current_year = 2021

df["Content Age"] = current_year - df["release_year"]

df[["release_year","Content Age"]].head()


# ### Question 3: Create Duration Category

# In[88]:


movie = df[df["type"]=="Movie"].copy()

movie["duration"] = movie["duration"].str.replace(" min","").astype(int)

movie["Duration Category"] = pd.cut(movie["duration"],
                                    bins=[0,60,120,500],
                                    labels=["Short","Medium","Long"])

movie["Duration Category"].value_counts()


# ### Visualization 1: Release Decade

# In[99]:


plt.figure(figsize=(8,5))

decade = df["Release Decade"].value_counts().sort_index()

plt.plot(decade.index,
         decade.values,
         marker="o",
         color="mediumorchid",
         linewidth=3)

plt.title("Content Released by Decade")
plt.xlabel("Release Decade")
plt.ylabel("Number of Titles")

plt.grid(linestyle="--", alpha=0.5)

plt.show()


# ### Visualization 2: Duration Category

# In[94]:


plt.figure(figsize=(8,5))

sns.countplot(x="Duration Category",
              data=movie,
              hue="Duration Category",
              palette="Set2",
              legend=False)

plt.title("Movie Duration Categories")
plt.xlabel("Duration Category")
plt.ylabel("Number of Movies")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.show()


# ### Visualization 3: Content Age Distribution

# In[98]:


plt.figure(figsize=(10,5))

plt.hist(df["Content Age"],
         bins=15,
         color="teal",
         edgecolor="black")

plt.title("Distribution of Content Age")
plt.xlabel("Content Age (Years)")
plt.ylabel("Number of Titles")

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.show()


# ### Findings
# 
# - Most Netflix titles were released during the **2010s**, followed by the **2020s**.
# 
# - Very few titles belong to the **1980s** and **1990s**, indicating that Netflix mainly focuses on newer content.
# 
# - The **Content Age** feature shows that most titles are less than 15 years old.
# 
# - Most movies fall into the **Medium** duration category (61–120 minutes).
# 
# - Short and Long duration movies make up a smaller portion of the Netflix movie library.
# 
# - Feature engineering makes the dataset easier to analyze by grouping release years, calculating content age, and categorizing movie durations.

# # Task 11 : Multivariate Analysis
# ### Question 1: Country vs Content Type

# In[100]:


country_type = pd.crosstab(df["country"], df["type"])

country_type.head()


# ### Top 10 Countries

# In[101]:


top_country = df["country"].value_counts().head(10).index

country_type = pd.crosstab(df[df["country"].isin(top_country)]["country"],
                           df[df["country"].isin(top_country)]["type"])

country_type


# ### Visualization (Grouped Bar Chart)

# In[102]:


plt.figure(figsize=(12,6))

country_type.plot(kind="bar",
                  figsize=(12,6),
                  color=["mediumpurple","mediumseagreen"],
                  edgecolor="black")

plt.title("Top 10 Countries vs Content Type",fontsize=15,fontweight="bold")
plt.xlabel("Country")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.grid(axis="y",linestyle="--",alpha=0.5)

plt.show()


# ### Question 2: Year vs Content Type
# #### GroupBy

# In[103]:


year_type = df.groupby(["release_year","type"]).size().unstack()

year_type.tail()


# ### Visualization (Stacked Area Chart)

# In[104]:


plt.figure(figsize=(12,6))

year_type.plot(kind="area",
               stacked=True,
               figsize=(12,6),
               color=["mediumpurple","mediumseagreen"])

plt.title("Year vs Content Type",fontsize=15,fontweight="bold")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.show()


# ### Question 3: Rating vs Content Type
# #### Crosstab

# In[105]:


rating_type = pd.crosstab(df["rating"],df["type"])

rating_type


# ### Visualization (Stacked Bar Chart)

# In[108]:


plt.figure(figsize=(12,6))

rating_type.plot(kind="bar",
                 stacked=True,
                 figsize=(12,6),
                 color=["coral","royalblue"],
                 edgecolor="black")

plt.title("Rating vs Content Type",fontsize=15,fontweight="bold")
plt.xlabel("Rating")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.grid(axis="y",linestyle="--",alpha=0.5)

plt.show()


# ### Heatmap

# In[109]:


plt.figure(figsize=(8,6))

sns.heatmap(country_type,
            annot=True,
            cmap="YlGnBu",
            fmt="d")

plt.title("Heatmap of Country vs Content Type")

plt.show()


# ### Findings
# 
# - The **United States** has the highest number of both Movies and TV Shows.
# 
# - Countries such as **India** and the **United Kingdom** also contribute significantly, but Movies dominate their content.
# 
# - Over the years, Netflix has steadily increased both Movie and TV Show releases, with rapid growth after 2015.
# 
# - Movies consistently outnumber TV Shows across most release years.
# 
# - Ratings such as **TV-MA** and **TV-14** contain the highest number of titles.
# 
# - The heatmap clearly highlights countries with the highest contribution and the dominance of Movies over TV Shows.

# # Task 12
# ## - Business Insights

# ### 1. Movies dominate Netflix's catalog, though TV Shows are also significant and often strategically important for retention.
# ### 2. A small number of countries contribute most of the platform's content, with the U.S. typically leading by a large margin.
# ### 3. Content production has accelerated strongly over time, especially from the 2010s onward.
# ### 4. A handful of genres account for a large portion of the catalog, showing strong concentration in certain viewer-friendly categories.
# ### 5. Most movies cluster in a medium duration range, suggesting Netflix favors broadly accessible watch lengths.
# 
# 

# ## - Strategic Recommendations
# ### 1. Increase regional original content in emerging markets to reduce reliance on a few dominant countries and improve global subscriber growth.
# ### 2. Expand investment in top-performing genres while testing adjacent niche genres for differentiated audience capture.
# ### 3. Prioritize serialized TV content alongside movies, since TV series can improve long-term engagement and repeat platform visits.

# In[ ]:




