# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

OUTPUT_DIR = "output"
HTML_FILE = "covid_report.html"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_URL = ("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv")

df = pd.read_csv(DATA_URL)

df = df[
    [
        "iso_code",
        "location",
        "date",
        "new_cases",
        "population"
    ]
]

# Remove aggregated regions (World, Europe, income groups...)
df = df[df["iso_code"].str.len() == 3]

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Sort data
df = df.sort_values(["location", "date"])

# Remove missing values
df = df.dropna(subset=["new_cases", "population"])

# Last 7 days for each country
last_7_days = df.groupby("location").tail(7)

# Average number of new cases per day
weekly_avg = (
    last_7_days
    .groupby("location")
    .agg(
        avg_new_cases=("new_cases", "mean"),
        population=("population", "first"),
        iso_code=("iso_code", "first")
    )
)

# Cases per 100,000 inhabitants
weekly_avg["cases_per_100k"] = (
    weekly_avg["avg_new_cases"] / weekly_avg["population"] * 100_000
)

# Top 20 countries by incidence
top20 = (
    weekly_avg
    .sort_values("cases_per_100k", ascending=False)
    .head(20)
)

# Bar charts
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Population chart
axes[0].bar(top20.index, top20["population"])
axes[0].set_title("Population of Countries (Top 20)")
axes[0].set_ylabel("Population")
axes[0].tick_params(axis="x", rotation=90)

# Cases per 100k chart
axes[1].bar(top20.index, top20["cases_per_100k"])
axes[1].set_title(
    "Average Number of New COVID-19 Cases\n"
    "Over 7 Days per 100,000 Inhabitants"
)
axes[1].set_ylabel("Cases per 100,000")
axes[1].tick_params(axis="x", rotation=90)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bar_charts.png", dpi=150)
plt.close()

# Scatter plot
X = top20["population"].values.reshape(-1, 1)
y = top20["cases_per_100k"].values

plt.figure(figsize=(10, 6))
plt.scatter(X, y)

for i, code in enumerate(top20["iso_code"]):
    plt.text(X[i], y[i], code)

plt.xlabel("Population")
plt.ylabel("Cases per 100,000 Inhabitants")
plt.title("Relationship Between Population Size and COVID-19 Incidence (Top 20)")

plt.savefig(f"{OUTPUT_DIR}/scatter.png", dpi=150)
plt.close()

# Linear regression
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X, y)
plt.plot(X, y_pred)

plt.xlabel("Population")
plt.ylabel("Cases per 100,000 Inhabitants")
plt.title("Linear Regression Fit")

plt.savefig(f"{OUTPUT_DIR}/linear_fit.png", dpi=150)
plt.close()

# Residual analysis
residuals = y - y_pred

mu, sigma = norm.fit(residuals)

plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=10, density=True, alpha=0.6)

x = np.linspace(residuals.min(), residuals.max(), 100)
plt.plot(x, norm.pdf(x, mu, sigma))

plt.xlabel("Residual")
plt.ylabel("Probability Density")
plt.title("Histogram of Residuals with Normal Distribution")

plt.savefig(f"{OUTPUT_DIR}/residuals.png", dpi=150)
plt.close()


# create html
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>COVID-19 Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: auto;
        }}
        img {{
            width: 100%;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>

<h1>COVID-19 Data Analysis</h1>

<h2>1. Population and Weekly Incidence (Top 20 Countries)</h2>
<p>
The left chart shows the total population of the selected countries, while the right chart
shows the average number of new COVID-19 cases over the last seven days
normalized per 100,000 inhabitants. These charts demonstrate that a high incidence
rate is not directly related to the absolute size of the population.
</p>
<img src="{OUTPUT_DIR}/bar_charts.png">
Comparing these two charts confirms that a large population does not automatically imply a higher infection rate. Therefore, it is crucial to monitor indicators normalized to population size.
<h2>2. Relationship Between Population Size and COVID-19 Incidence</h2>
<p>
The scatter plot illustrates the relationship between a country's population size and
its average number of new COVID-19 cases per 100,000 inhabitants. The visualization
suggests that there is no strong linear dependence between these two variables.
</p>
<img src="{OUTPUT_DIR}/scatter.png">

<h2>3. Linear Regression Fit</h2>
<p>
A linear regression line was added to the scatter plot to provide a simple approximation
of the relationship between population size and COVID-19 incidence. The fitted line
indicates only a weak linear trend.
</p>
<img src="{OUTPUT_DIR}/linear_fit.png">

<h2>4. Residual Analysis</h2>
<p>
The residual histogram shows the distribution of differences between the observed values
and the values predicted by the linear regression model. A fitted normal distribution
is overlaid to assess whether the residuals follow an approximately normal distribution,
which is an important assumption of linear regression.
</p>
<img src="{OUTPUT_DIR}/residuals.png">

</body>
</html>
"""
with open(HTML_FILE, "w", encoding="utf-8") as f:
    f.write(html_content)
