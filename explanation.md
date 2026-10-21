### Inspecting and Cleaning the Raw Data
I looked at the data and made sure it was not broken or had information that didn't make sense realistically. 

#### A diamond is a 3-d solid physical object. The data has: $x Length in mm$, $y Width in mm$, $z Depth in mm$. 
- A real diamond cannot  have length(x), width(y),or depth(z) that is 0 mm or negative. 
- If $x=0$, $y=0$, or $z=0$, the measurement was either skipped, lost, or recorded as zero instead of missing 

The rows that had $z=0$, meaning no depth were not removed entirely; A formula for diamond cutters was used. The formula
evaluates a diamonds depth as a ratio relative to its average diameter.
#### depth percentage = (z /(x+y/2) ) x 100
Now solving for $z$:
depth /100 = z (x + y/2)
depth /100 x (x + y)/2)  = z 
depth x (x + y) /(200) = z
z  = depth x (x + y) / (200)

In Python: 
$x --> df['x']$ 
$y --> df['y']$
$z --> df['z']$

# From assignment1.ipynb:
df = df[~((df['x'] == 0) & (df['y'] == 0))].copy()
missing_z = (df['z'] == 0) & (df['x'] > 0) & (df['y'] > 0)
df.loc[missing_z, 'z'] = df['depth'] * (df['x'] + df['y']) / 200
depth --> df['depth'] 

#However now having to look at it I could of :
# Step 1: Drop rows where length (x) or width (y) is 0
# Why: If both x and y are 0, we have no foundation to calculate z!
df = df[df['x'] > 0]
df = df[df['y'] > 0]

# Step 2: Find rows where depth (z) is 0
missing_depth = (df['z'] == 0)

# Step 3: Fill missing z using our rearranged math formula:
# z = depth * (x + y) / 200
df.loc[missing_depth, 'z'] = df['depth'] * (df['x'] + df['y']) / 200

Connecting this back to the math: df['depth] = depth percentage, df['x'] + df['y'] = length plus width added together


### Splitting Data 
In Python:
from sklearn.model_selection import train_test_split
X = df.drop(columns='price')
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=432
)

-X_train & y_train: The study guide (80% of data). The model learns patterns and weights from this set. 
-X_test & y_test is the final exam (20% of the data).  Kept completely separate to check if the model actually learned or just memorized answers.


### Feature Scaling with StandardScaler
The Math Formula ($Z$-Score)Features have completely different sizes: carat goes from $0.2$ to $5.0$, while table and depth are around $60$, and price is in the thousands.   

To make sure big numbers do not overpower small numbers, we scale them using the $z$-score formula:   
$$z\text{-score} = \frac{x_i - \mu}{\sigma}
$$$x_i$: The original measurement of that diamond (e.g., $0.75$ carats)   
$\mu$ (mu / mean): The average of all diamonds in the training set   
$\sigma$ (sigma / standard deviation): How spread out the values are   

What the Result Means:
$z = 0$: The diamond is exactly average.   
$z = +1.5$: The diamond is $1.5$ standard deviations larger than average.
$z = -1.0$: The diamond is $1.0$ standard deviation smaller than average.


### Preparing Features for Encoding and Scaling
Machine learning models only read and understand numbers and so details that describe a diamond like; cut, color, and clarity would not be understod. 
Ordinal Encoding:
Because a "Premium" cut is strictly better than a "Fair" cut, so numerical values of hierarchy were assigned.
Application to DatasetCut: 
$\text{Fair} (0) < \text{Good} (1) < \text{Very Good} (2) < \text{Premium} (3) < \text{Ideal} (4)$   
Color: $\text{J} (0) < \text{I} (1) < \text{H} (2) < \text{G} (3) < \text{F} (4) < \text{E} (5) < \text{D} (6)$   
Clarity: $\text{I1} (0) < \text{SI2} (1) < \text{SI1} (2) < \text{VS2} (3) < \text{VS1} (4) < \text{VVS2} (5) < \text{VVS1} (6) < \text{IF} (7)$

In Python:
cut_rank = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_rank = ["J", "I", "H", "G", "F", "E", "D"]
clarity_rank = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

ordinal_encoder = OrdinalEncoder(
    categories=[cut_rank, color_rank, clarity_rank]
)
df[["cut", "color", "clarity"]] = ordinal_encoder.fit_transform(
    df[["cut", "color", "clarity"]]
)

One-Hot Encoding:
Assign a 0,1  in a mathematical way to say yes or no. It will give each category its own seperate column containing a  1 (yes) or 0 (no).

For example taking the feature $cut$, we would want $Ideal$ cut:
Row    cut_Fair  cut_Good   cut_VeryGood  cut_Premium  cut_Ideal
1       0         0              0       0             1

In Python:
from sklearn.preprocessing import OneHotEncoder

# drop=None ensures every category (including 'Fair') gets its own explicit column
ohe = OneHotEncoder(sparse_output=False, drop=None)

# Fit on training data only
X_train_ohe = ohe.fit_transform(X_train[["cut", "color", "clarity"]])
X_test_ohe = ohe.transform(X_test[["cut", "color", "clarity"]])

