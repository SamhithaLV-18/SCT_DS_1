# ===========================================
# 📈 chart_generation_task_final.py
# Full pipeline: Load dataset, analyze, and generate charts including Age & Gender
# ===========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# -------------------------------------------
# 🔹 Step 1: Load or generate dataset
# -------------------------------------------
csv_file = "sample_dataset.csv"

if not os.path.exists(csv_file):
    # Generate sample dataset with multiple years
    countries = ["India", "China", "United States", "Brazil", "Germany",
                 "Nigeria", "Japan", "Australia", "Canada", "France"]
    continents = ["Asia", "Asia", "North America", "South America", "Europe",
                  "Africa", "Asia", "Oceania", "North America", "Europe"]
    years = [2020, 2021, 2022, 2023, 2024]

    data = {"Country": [], "Continent": [], "Population (Millions)": [], 
            "GDP (Billion USD)": [], "Life Expectancy": [], "Year": [],
            "Age": [], "Gender": []}

    pop_values = [1380, 1410, 331, 212, 83, 206, 126, 25, 38, 65]
    gdp_values = [3000, 17000, 21000, 2100, 4000, 450, 4100, 1600, 2200, 2800]
    life_values = [69, 77, 78, 74, 80, 61, 84, 82, 82, 81]

    genders = ["Male", "Female"]
    for i, country in enumerate(countries):
        for j, year in enumerate(years):
            data["Country"].append(country)
            data["Continent"].append(continents[i])
            data["Population (Millions)"].append(pop_values[i] + j*5)
            data["GDP (Billion USD)"].append(gdp_values[i] + j*200)
            data["Life Expectancy"].append(life_values[i] + j*0.3)
            data["Year"].append(year)
            data["Age"].append(np.random.randint(0, 101))
            data["Gender"].append(np.random.choice(genders))

    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"✅ Sample dataset '{csv_file}' created with Age & Gender.")
else:
    df = pd.read_csv(csv_file)
    print(f"✅ Dataset '{csv_file}' loaded successfully.")

    # Add Age & Gender columns if they don't exist
    if "Age" not in df.columns:
        df["Age"] = np.random.randint(0, 101, size=len(df))
        print("ℹ️ 'Age' column added to existing dataset.")
    if "Gender" not in df.columns:
        df["Gender"] = np.random.choice(["Male", "Female"], size=len(df))
        print("ℹ️ 'Gender' column added to existing dataset.")

# -------------------------------------------
# 🔹 Step 2: Inspect the dataset
# -------------------------------------------
print("\n--- Dataset Head ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Dataset Description ---")
print(df.describe())

# -------------------------------------------
# 🔹 Step 3: Generate charts
# -------------------------------------------
sns.set(style="whitegrid")
latest_year = df["Year"].max()
df_latest = df[df["Year"] == latest_year]

# 3.1: Bar chart - Population by Country
plt.figure(figsize=(10,6))
sns.barplot(x="Country", y="Population (Millions)", data=df_latest, palette="viridis")
plt.title(f"Population by Country (Millions) - {latest_year}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("population_bar_chart.png")
plt.show()

# 3.2: Bar chart - GDP by Country
plt.figure(figsize=(10,6))
sns.barplot(x="Country", y="GDP (Billion USD)", data=df_latest, palette="magma")
plt.title(f"GDP by Country (Billion USD) - {latest_year}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("gdp_bar_chart.png")
plt.show()

# 3.3: Scatter plot - Life Expectancy vs GDP
plt.figure(figsize=(8,6))
sns.scatterplot(x="GDP (Billion USD)", y="Life Expectancy", hue="Continent", data=df_latest, s=100, palette="tab10")
plt.title(f"Life Expectancy vs GDP - {latest_year}")
plt.tight_layout()
plt.savefig("life_expectancy_vs_gdp.png")
plt.show()

# 3.4: Line plot - Population trend
plt.figure(figsize=(12,6))
sns.lineplot(x="Year", y="Population (Millions)", hue="Country", data=df, marker="o")
plt.title("Population Trend by Country (2020-2024)")
plt.tight_layout()
plt.savefig("population_trend_line_chart.png")
plt.show()

# 3.5: Histogram - Age distribution
plt.figure(figsize=(10,6))
sns.histplot(df_latest["Age"], bins=20, kde=True, color="skyblue")
plt.title(f"Age Distribution - {latest_year}")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("age_distribution_histogram.png")
plt.show()

# 3.6: Bar chart - Gender distribution
plt.figure(figsize=(6,6))
sns.countplot(x="Gender", data=df_latest, palette="Set2")
plt.title(f"Gender Distribution - {latest_year}")
plt.tight_layout()
plt.savefig("gender_distribution_bar_chart.png")
plt.show()

# -------------------------------------------
# 🔹 Step 4: Completion message
# -------------------------------------------
print("\n🎉 All charts including Age & Gender generated successfully!")
