# Dynamic Pricing Impact on Conversion and Margin in E-commerce 📈🛒

A comprehensive data analytics and pricing strategy project that investigates how dynamic pricing affects conversion rates, demand elasticity, and profit margins in e-commerce environments.

---

## 📋 Project Overview

This project explores the relationship between pricing strategies and customer purchasing behavior using e-commerce transaction and behavioral data. The analysis focuses on identifying price elasticity patterns, evaluating whether dynamic pricing improves business performance, and estimating optimal pricing points for revenue maximization.

### Features

* 📊 Data collection and loading from CSV
* 🧹 Data cleaning and preprocessing
* 📈 Exploratory Data Analysis (EDA)
* 📉 Statistical and regression analysis
* 🔍 Price elasticity estimation
* 💰 Revenue and margin analysis
* 📊 Data visualization using Matplotlib and Seaborn

---

## 🎯 Business Understanding

Traditional fixed pricing strategies may cause businesses to lose potential revenue opportunities. Products may be overpriced, reducing conversion rates, or underpriced, reducing profit margins.

This project aims to investigate whether dynamic pricing strategies can:

* improve conversion rates,
* increase profit margins,
* and identify optimal price points based on customer demand behavior.

---

## ❓ Research Questions (RBL)

1. How does price elasticity vary across different products?
2. Does dynamic pricing improve conversion rate and profit margin?
3. What is the optimal price point for maximizing revenue?

---

## 🔬 Analytic Approach

The project follows a regression-based analytical framework:

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Price Variation Analysis
* Regression Modeling
* Elasticity Estimation
* Revenue and Margin Evaluation

### Potential Techniques

* Linear Regression
* Log-Log Regression
* Correlation Analysis
* Revenue Optimization Analysis

### Elasticity Model

```math
\log(Q)=\beta_0+\beta_1\log(P)+\epsilon
```

Where:

* (Q) represents quantity sold
* (P) represents price
* (\beta_1) approximates price elasticity

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10 or higher
* Jupyter Notebook or VS Code
* pip or uv package manager

---

## 📦 Dependencies

* `pandas` - Data manipulation and analysis
* `numpy` - Numerical computing
* `matplotlib` - Data visualization
* `seaborn` - Statistical visualization
* `scikit-learn` - Regression and machine learning models
* `jupyter` - Notebook environment

---

## 📁 Project Structure

```text
dynamic-pricing-ecommerce/
├── data/
│   └── ecommerce_dataset.csv
├── notebooks/
│   └── main.ipynb
├── results/
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

Dataset Source:
[Kaggle E-commerce Dataset](https://www.kaggle.com/datasets/dastgeerjutt/e-commerce-customer-behavior?utm_source=chatgpt.com)

The dataset contains e-commerce customer behavior and transaction-related features such as:

| Column          | Description                   |
| --------------- | ----------------------------- |
| product_id      | Product identifier            |
| category        | Product category              |
| price           | Product selling price         |
| discount        | Discount amount or percentage |
| quantity_sold   | Quantity sold                 |
| customer_rating | Customer satisfaction rating  |
| payment_method  | Customer payment type         |
| timestamp       | Transaction date/time         |
| conversion_rate | Purchase conversion indicator |
| revenue         | Revenue generated             |
| margin          | Profit margin                 |

---

## 📈 Analysis Sections

The notebook is organized into the following sections:

1. Data Collection
2. Data Understanding
3. Data Cleaning & Preprocessing
4. Exploratory Data Analysis (EDA)
5. Price Elasticity Analysis
6. Regression Modeling
7. Revenue and Margin Analysis
8. Visualization and Insights
9. Final Conclusions

---

## 📊 Sample Visualizations

The project generates visualizations including:

* Price distribution plots
* Correlation heatmaps
* Price vs Quantity Sold scatter plots
* Revenue comparison charts
* Product category pricing analysis
* Margin analysis visualizations

---

## 🎯 Expected Outcomes

This project is expected to:

* identify products with high or low price sensitivity,
* estimate elasticity across categories,
* evaluate whether dynamic pricing improves conversion and margin,
* and provide insights for revenue optimization strategies.

---

## ⚠️ Limitations

The dataset may contain educational or simulated data and may not fully represent real-world e-commerce systems. Therefore, findings should be interpreted as analytical insights rather than direct business recommendations.

---

## ▶️ How to Run

### Option 1: Using Jupyter Notebook

1. Open `main.ipynb`
2. Select the Python kernel
3. Run all notebook cells sequentially

### Option 2: Using Command Line

```bash
pip install -r requirements.txt
jupyter notebook
```

---

## 👥 Authors

* Le Pham Hue Chi

---

## 📄 License

This project is intended for educational and research purposes.

---

## 🙋 Support

If you encounter any issues:

1. Check dataset paths and dependencies
2. Ensure required libraries are installed
3. Review notebook comments and documentation

---

