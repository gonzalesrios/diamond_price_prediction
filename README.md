# diamond_price_prediction

# Predicting Diamond Prices Using Machine Learning

## What Is This Project?
In this project, I used Python to predict how much a diamond costs based on its size, cut, color, and clarity. 

I built this project to practice the basics of machine learning: cleaning messy data, making charts to understand trends, and testing different computer models to see which one makes the best guesses.

---

## The Steps I Took

### 1. Cleaning the Data
Before training any models, I checked the dataset for mistakes:
* Found rows where diamond measurements (length, width, or depth) were listed as **0 mm** (which is impossible for a real diamond) and removed or fixed them.
* Converted text descriptions (like Cut: "Ideal" or Color: "E") into numbers using **One-Hot Encoding** so the models could read them.

### 2. Exploring the Data (EDA)
I created charts to see what actually drives a diamond's price:
* **Carat weight** has the biggest impact on price: heavier diamonds cost significantly more.
* The cut, color, and clarity also change the price, especially for larger stones.

### 3. Testing the Models
I tested a few common beginner models using 5-fold cross-validation:
* **Linear Regression (Baseline):** A basic model that assumes a straight-line relationship between diamond features and price.
* **Ridge & Lasso Regression:** Tweaked versions of linear regression designed to prevent the model from overreacting to noise in the data.
* **Decision Tree Regressor:** A model that splits data into branching "yes/no" questions based on features.

---

## What I Found
* The **Linear Regression** model gave a solid starting point, but it struggled because diamond prices do not rise in a simple straight line (for example, hitting a full 1.0 or 2.0 carat mark causes a big jump in price).
* The **Decision Tree Regressor** performed much better. It captured the sharp price jumps and had a much lower error rate (lower RMSE and MAE).

---

## How to Run This Code

1. Download or clone this repository.
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
