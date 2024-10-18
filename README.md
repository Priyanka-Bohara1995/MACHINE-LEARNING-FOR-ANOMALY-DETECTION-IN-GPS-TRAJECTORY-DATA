import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Load the dataset
file_path = "C:\\Users\\Priyanka\\Downloads\\gps_track.csv" # Ensure this path is correct for your environment
gps_track = pd.read_csv(file_path)
# Display the first few rows of the dataset
print(gps_track.head())
 id id_android speed time distance rating rating_bus rating_weather \
0 1 0 19.21 0.14 2.652 3 0 0
1 2 0 30.85 0.17 5.290 3 0 0
2 3 1 13.56 0.07 0.918 3 0 0
3 4 1 19.77 0.39 7.700 3 0 0
4 8 0 25.81 0.15 3.995 2 0 0
 car_or_bus
0 1
1 1
2 2
3 2
4 1
# Summary of the dataset
data_summary = gps_track.describe()
print(data_summary)
 id id_android speed time distance \
count 163.000000 163.000000 163.000000 163.000000 163.000000
mean 15607.650307 7.386503 16.704847 0.264049 5.302411
std 18644.257138 7.348742 16.016527 0.292783 7.639011
min 1.000000 0.000000 0.010000 0.000000 0.001000
25% 48.500000 1.000000 1.590000 0.035000 0.034500
50% 158.000000 4.000000 16.690000 0.210000 3.995000
75% 37991.000000 10.000000 23.915000 0.390000 7.333000
max 38092.000000 27.000000 96.210000 1.940000 55.770000
 rating rating_bus rating_weather car_or_bus
count 163.000000 163.000000 163.000000 163.000000
mean 2.515337 0.386503 0.515337 1.466258
std 0.679105 0.687859 0.841485 0.500397
min 1.000000 0.000000 0.000000 1.000000
25% 2.000000 0.000000 0.000000 1.000000
50% 3.000000 0.000000 0.000000 1.000000
75% 3.000000 1.000000 1.000000 2.000000
max 3.000000 3.000000 2.000000 2.000000
# Display the structure of the Dataset
print(gps_track.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 163 entries, 0 to 162
Data columns (total 9 columns):
# Column Non-Null Count Dtype
--- ------ -------------- -----
0 id 163 non-null int64
1 id_android 163 non-null int64
2 speed 163 non-null float64
3 time 163 non-null float64
4 distance 163 non-null float64
5 rating 163 non-null int64
6 rating_bus 163 non-null int64
7 rating_weather 163 non-null int64
8 car_or_bus 163 non-null int64
dtypes: float64(3), int64(6)
memory usage: 11.6 KB
None
# Remove ID and id_android columns
gps_track = gps_track.drop(columns=['id', 'id_android'])
print(gps_track)
 speed time distance rating rating_bus rating_weather car_or_bus
0 19.21 0.14 2.652 3 0 0 1
1 30.85 0.17 5.290 3 0 0 1
2 13.56 0.07 0.918 3 0 0 2
3 19.77 0.39 7.700 3 0 0 2
4 25.81 0.15 3.995 2 0 0 1
.. ... ... ... ... ... ... ...
158 30.05 0.22 6.574 2 0 0 1
159 30.17 0.26 7.706 3 0 0 1
160 1.15 0.01 0.015 1 3 2 2
161 0.84 0.01 0.006 3 1 2 2
162 1.37 0.02 0.023 3 1 2 2
[163 rows x 7 columns]
# Check for missing values
missing_values = gps_track.isnull().sum()
print("Missing values in each column:\n", missing_values)
Missing values in each column:
speed 0
time 0
distance 0
rating 0
rating_bus 0
rating_weather 0
car_or_bus 0
dtype: int64
# Visualize the missing heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(gps_track.isna(), cbar=False, cmap='gray')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.title('Missing Data Heatmap')
plt.show()
# Drop rows with missing values
gps_track_clean = gps_track.dropna()
gps_track_clean.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 163 entries, 0 to 162
Data columns (total 9 columns):
# Column Non-Null Count Dtype
--- ------ -------------- -----
0 id 163 non-null int64
1 id_android 163 non-null int64
2 speed 163 non-null float64
3 time 163 non-null float64
4 distance 163 non-null float64
5 rating 163 non-null int64
6 rating_bus 163 non-null int64
7 rating_weather 163 non-null int64
8 car_or_bus 163 non-null int64
dtypes: float64(3), int64(6)
memory usage: 11.6 KB
# Boxplot for all variables
plt.figure(figsize=(12, 8))
sns.boxplot(data=gps_track)
plt.title('Boxplot for all variables')
plt.xticks(rotation=45)
plt.show()
# Define a function to remove outliers based on IQR
def remove_outliers(df, column):
 Q1 = df[column].quantile(0.25)
 Q3 = df[column].quantile(0.75)
 IQR = Q3 - Q1
 lower_bound = Q1 - 1.5 * IQR
 upper_bound = Q3 + 1.5 * IQR
 df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
 return df
# List of columns to remove outliers from
columns_to_clean = ['speed', 'time', 'distance', 'rating_bus']
#Apply the function to each column
for column in columns_to_clean:
 gps_track = remove_outliers(gps_track, column)
# Boxplot for Speed
plt.figure(figsize=(6, 4))
sns.boxplot(y=gps_track['speed'])
plt.title('Speed')
plt.xlabel('Speed')
plt.ylabel('Value')
plt.show()
# Boxplot for Time
plt.figure(figsize=(6, 4))
sns.boxplot(y=gps_track['time'])
plt.title('Time')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
# Histogram for all variables
numerical_columns = ['speed', 'time', 'distance', 'rating', 'rating_bus', 'rating_weather', 'car_or_bus']
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
fig.suptitle('Histograms of All Variables')
axes = axes.flatten()
for i, col in enumerate(numerical_columns):
 axes[i].hist(gps_track[col].dropna(), bins=20)
 axes[i].set_title(col)
# Hide any remaining empty subplots
for j in range(i + 1, len(axes)):
 fig.delaxes(axes[j])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# Histogram
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(gps_track['speed'], kde=True)
plt.title('Histogram of Speed')
plt.subplot(1, 2, 2)
sns.histplot(gps_track['time'], kde=True)
plt.title('Histogram of Time')
plt.show()
# Min-Max Scaling
scaler = MinMaxScaler()
gps_track_mm = pd.DataFrame(scaler.fit_transform(gps_track), columns=gps_track.columns)
# Plotting the Boxplot
plt.figure(figsize=(10, 6))
gps_track_mm.boxplot()
plt.title("Min-Max Scaling")
plt.show()
# Define the z-score scaling functions
def z_score_scaling(column):
 return (column - column.mean()) / column.std()
def z_score_scaling_2sd(column):
 return (column - column.mean()) / (2 * column.std())
# Apply z-score scaling to each column
gps_track_z1 = gps_track.apply(z_score_scaling)
gps_track_z2 = gps_track.apply(z_score_scaling_2sd)
# Create the boxplots
plt.figure(figsize=(12, 6))
# Boxplot for Z-score with 1 standard deviation
plt.subplot(1, 2, 1)
gps_track_z1.boxplot()
plt.title('Z-score, 1 sd')
plt.ylabel('Z-Score Values')
plt.xlabel('Columns')
# Boxplot for Z-score with 2 standard deviations
plt.subplot(1, 2, 2)
gps_track_z2.boxplot()
plt.title('Z-score, 2 sd')
plt.ylabel('Z-Score Values')
plt.xlabel('Columns')
plt.tight_layout()
plt.show()
import pandas as pd
from scipy.stats import spearmanr
# Calculate Spearman correlation
spearman_corr, spearman_p_value = spearmanr(gps_track['speed'], gps_track['distance'])
print(f"Spearman correlation: {spearman_corr}, p-value: {spearman_p_value}")
spearman_corr, spearman_p_value = spearmanr(gps_track['speed'], gps_track['rating'])
print(f"Spearman correlation: {spearman_corr}, p-value: {spearman_p_value}")
spearman_corr, spearman_p_value = spearmanr(gps_track['speed'], gps_track['rating_bus'])
print(f"Spearman correlation: {spearman_corr}, p-value: {spearman_p_value}")
spearman_corr, spearman_p_value = spearmanr(gps_track['speed'], gps_track['rating_weather'])
print(f"Spearman correlation: {spearman_corr}, p-value: {spearman_p_value}")
spearman_corr, spearman_p_value = spearmanr(gps_track['speed'], gps_track['car_or_bus'])
print(f"Spearman correlation: {spearman_corr}, p-value: {spearman_p_value}")
Spearman correlation: 0.7668548591459986, p-value: 6.526246554373504e-30
Spearman correlation: 0.28937891659427184, p-value: 0.0003607733237625879
Spearman correlation: -0.1594498295184976, p-value: 0.052898725895505316
Spearman correlation: -0.1697602653522656, p-value: 0.039140353208210896
Spearman correlation: -0.36158266681979745, p-value: 6.323429391491284e-06
# Calculate Pearson correlation
pearson_corr = gps_track['speed'].corr(gps_track['distance'], method='pearson')
print(f"Pearson correlation: {pearson_corr}")
pearson_corr = gps_track['speed'].corr(gps_track['rating'], method='pearson')
print(f"Pearson correlation: {pearson_corr}")
pearson_corr = gps_track['speed'].corr(gps_track['rating_bus'], method='pearson')
print(f"Pearson correlation: {pearson_corr}")
pearson_corr = gps_track['speed'].corr(gps_track['rating_weather'], method='pearson')
print(f"Pearson correlation: {pearson_corr}")
pearson_corr = gps_track['speed'].corr(gps_track['car_or_bus'], method='pearson')
print(f"Pearson correlation: {pearson_corr}")
Pearson correlation: 0.7469802663565005
Pearson correlation: 0.24910045519249213
Pearson correlation: -0.14308250088641128
Pearson correlation: -0.18289788148774183
Pearson correlation: -0.37841259722056725
# Calculate the correlation matrix for the reduced dataframe
correlation_matrix_reduced = gps_track.corr()
# Display the reduced correlation matrix
print(correlation_matrix_reduced)
 speed time distance rating rating_bus \
speed 1.000000 0.365841 0.746980 0.249100 -0.143083
time 0.365841 1.000000 0.790376 -0.063641 -0.032405
distance 0.746980 0.790376 1.000000 0.077826 -0.079070
rating 0.249100 -0.063641 0.077826 1.000000 -0.005037
rating_bus -0.143083 -0.032405 -0.079070 -0.005037 1.000000
rating_weather -0.182898 -0.030613 -0.102467 0.139658 0.908459
car_or_bus -0.378413 -0.188675 -0.279799 -0.120404 0.662337
 rating_weather car_or_bus
speed -0.182898 -0.378413
time -0.030613 -0.188675
distance -0.102467 -0.279799
rating 0.139658 -0.120404
rating_bus 0.908459 0.662337
rating_weather 1.000000 0.693324
car_or_bus 0.693324 1.000000
# Calculate the correlation matrix
correlation_matrix = gps_track.corr()
# Plot the correlagram
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Corrplot of GPS Track Data')
plt.show()
# Remove the highly correlated variables
columns_to_drop = ['rating_bus', 'rating_weather']
gps_track_cor = gps_track.drop(columns=columns_to_drop)
# Print the resulting DataFrame
print(gps_track_cor)
 speed time distance rating car_or_bus
0 19.21 0.14 2.652 3 1
1 30.85 0.17 5.290 3 1
2 13.56 0.07 0.918 3 2
3 19.77 0.39 7.700 3 2
4 25.81 0.15 3.995 2 1
.. ... ... ... ... ...
157 28.34 0.11 3.130 3 1
158 30.05 0.22 6.574 2 1
159 30.17 0.26 7.706 3 1
161 0.84 0.01 0.006 3 2
162 1.37 0.02 0.023 3 2
[148 rows x 5 columns]
# Check correlation matrix again
correlation_matrix_updated = gps_track.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix_updated, annot=True, cmap='coolwarm')
plt.title('Updated Correlation Matrix')
plt.show()
#Split the dataset
import pandas as pd
from sklearn.model_selection import train_test_split
X = gps_track.drop(columns=['speed', 'time'])
y_speed = gps_track['speed']
y_time = gps_track['time']
X_train, X_test, y_train_speed, y_test_speed = train_test_split(X, y_speed, test_size=0.2, random_state=42)
_, _, y_train_time, y_test_time = train_test_split(X, y_time, test_size=0.2, random_state=42)
# Calculate proportions of the target variables in training and test sets
train_speed_proportions = y_train_speed.value_counts(normalize=True)
test_speed_proportions = y_test_speed.value_counts(normalize=True)
train_time_proportions = y_train_time.value_counts(normalize=True)
test_time_proportions = y_test_time.value_counts(normalize=True)
# Print proportions
print("Training Data Speed Proportions:")
print(train_speed_proportions)
print("\nTest Data Speed Proportions:")
print(test_speed_proportions)
print("\nTraining Data Time Proportions:")
print(train_time_proportions)
print("\nTest Data Time Proportions:")
print(test_time_proportions)
Training Data Speed Proportions:
speed
21.67 0.016949
4.87 0.016949
13.56 0.016949
9.45 0.016949
26.90 0.016949
 ...
0.58 0.008475
16.78 0.008475
1.55 0.008475
28.34 0.008475
4.68 0.008475
Name: proportion, Length: 105, dtype: float64
Test Data Speed Proportions:
speed
25.45 0.033333
0.31 0.033333
0.13 0.033333
1.31 0.033333
23.05 0.033333
14.53 0.033333
33.20 0.033333
6.76 0.033333
24.91 0.033333
0.56 0.033333
21.51 0.033333
27.53 0.033333
9.08 0.033333
17.93 0.033333
19.42 0.033333
21.30 0.033333
0.54 0.033333
24.43 0.033333
30.05 0.033333
13.53 0.033333
17.10 0.033333
8.90 0.033333
13.47 0.033333
28.10 0.033333
0.25 0.033333
16.36 0.033333
0.37 0.033333
20.80 0.033333
38.03 0.033333
14.44 0.033333
Name: proportion, dtype: float64
Training Data Time Proportions:
time
0.01 0.101695
0.00 0.101695
0.23 0.042373
0.02 0.042373
0.17 0.033898
0.13 0.033898
0.20 0.033898
0.46 0.033898
0.45 0.025424
0.31 0.025424
0.03 0.025424
0.33 0.025424
0.53 0.025424
0.39 0.016949
0.58 0.016949
0.26 0.016949
0.10 0.016949
0.25 0.016949
0.21 0.016949
0.24 0.016949
0.08 0.016949
0.54 0.016949
0.19 0.016949
0.28 0.016949
0.09 0.016949
0.77 0.016949
0.11 0.016949
0.36 0.016949
0.14 0.016949
0.35 0.016949
0.29 0.016949
0.44 0.016949
0.50 0.008475
0.18 0.008475
0.56 0.008475
0.15 0.008475
0.07 0.008475
0.16 0.008475
0.22 0.008475
0.27 0.008475
0.37 0.008475
0.04 0.008475
0.49 0.008475
0.48 0.008475
0.32 0.008475
0.51 0.008475
0.06 0.008475
Name: proportion, dtype: float64
Test Data Time Proportions:
time
0.02 0.166667
0.08 0.066667
0.29 0.066667
0.26 0.066667
0.19 0.033333
0.44 0.033333
0.32 0.033333
0.10 0.033333
0.49 0.033333
0.05 0.033333
0.66 0.033333
0.51 0.033333
0.24 0.033333
0.22 0.033333
0.42 0.033333
0.46 0.033333
0.47 0.033333
0.13 0.033333
0.00 0.033333
0.28 0.033333
0.43 0.033333
0.17 0.033333
0.27 0.033333
Name: proportion, dtype: float64
#Regression tree regression mode
# Set random seed for reproducibility
np.random.seed(12345)
# Create and fit the regression tree model for speed
speed_tree = DecisionTreeRegressor(random_state=12345)
speed_tree.fit(X_train, y_train_speed)
# Get basic information about the tree
print(speed_tree)
# Fit the decision tree regressor for time
e_rpart_time = DecisionTreeRegressor(random_state=12345)
e_rpart_time.fit(X_train, y_train_time)
# Get basic information about the trees
tree_rules_speed = export_text(e_rpart_speed, feature_names=list(X_train.columns))
tree_rules_time = export_text(e_rpart_time, feature_names=list(X_train.columns))
# Generate summary of the decision tree model for speed
n_nodes_speed = e_rpart_speed.tree_.node_count
max_depth_speed = e_rpart_speed.tree_.max_depth
feature_importances_speed = e_rpart_speed.feature_importances_
summary_speed = {
 "Number of nodes": n_nodes_speed,
 "Maximum depth": max_depth_speed,
 "Feature importances": dict(zip(X_train.columns, feature_importances_speed))
}
# Generate summary of the decision tree model for time
n_nodes_time = e_rpart_time.tree_.node_count
max_depth_time = e_rpart_time.tree_.max_depth
feature_importances_time = e_rpart_time.feature_importances_
summary_time = {
 "Number of nodes": n_nodes_time,
 "Maximum depth": max_depth_time,
 "Feature importances": dict(zip(X_train.columns, feature_importances_time))
}
# Display the summaries
print("Decision Tree Summary for Speed")
print(pd.DataFrame([summary_speed]))
print("\nDecision Tree Summary for Time")
print(pd.DataFrame([summary_time]))
(tree_rules_speed, tree_rules_time)
DecisionTreeRegressor(random_state=12345)
Decision Tree Summary for Speed
 Number of nodes Maximum depth \
0 209 12
 Feature importances
0 {'distance': 0.9374525506904673, 'rating': 0.0...
Decision Tree Summary for Time
 Number of nodes Maximum depth \
0 183 11
 Feature importances
0 {'distance': 0.8999249756260264, 'rating': 0.0...
('|--- distance <= 0.57\n| |--- distance <= 0.01\n| | |--- distance <= 0.00\n| | | |--- distance <= 0.00\n| | | | |--- car_or_bus <= 1.50\n| | | | | |-
-- rating <= 2.50\n| | | | | | |--- value: [0.07]\n| | | | | |--- rating > 2.50\n| | | | | | |--- value: [0.06]\n| | | | |--- car_or_bus >
1.50\n| | | | | |--- value: [0.10]\n| | | |--- distance > 0.00\n| | | | |--- rating <= 1.50\n| | | | | |--- distance <= 0.00\n| | | | |
| |--- value: [0.04]\n| | | | | |--- distance > 0.00\n| | | | | | |--- value: [0.01]\n| | | | |--- rating > 1.50\n| | | | | |--- rating <
= 2.50\n| | | | | | |--- car_or_bus <= 1.50\n| | | | | | | |--- value: [0.92]\n| | | | | | |--- car_or_bus > 1.50\n| | | | | | |
|--- distance <= 0.00\n| | | | | | | | |--- value: [0.58]\n| | | | | | | |--- distance > 0.00\n| | | | | | | | |--- value: [0.64]\n|
| | | | |--- rating > 2.50\n| | | | | | |--- value: [1.03]\n| | |--- distance > 0.00\n| | | |--- distance <= 0.01\n| | | | |--- rating_weathe
r <= 1.00\n| | | | | |--- value: [1.41]\n| | | | |--- rating_weather > 1.00\n| | | | | |--- value: [1.83]\n| | | |--- distance > 0.01\n| | |
| |--- rating_weather <= 1.00\n| | | | | |--- rating <= 2.50\n| | | | | | |--- car_or_bus <= 1.50\n| | | | | | | |--- value: [1.49]\n| | |
| | | |--- car_or_bus > 1.50\n| | | | | | | |--- value: [0.59]\n| | | | | |--- rating > 2.50\n| | | | | | |--- value: [0.07]\n| | |
| |--- rating_weather > 1.00\n| | | | | |--- distance <= 0.01\n| | | | | | |--- value: [0.84]\n| | | | | |--- distance > 0.01\n| | | | |
| |--- value: [1.09]\n| |--- distance > 0.01\n| | |--- car_or_bus <= 1.50\n| | | |--- distance <= 0.01\n| | | | |--- value: [1.35]\n| | | |--- distance
> 0.01\n| | | | |--- distance <= 0.30\n| | | | | |--- distance <= 0.03\n| | | | | | |--- distance <= 0.01\n| | | | | | | |--- value: [0.
73]\n| | | | | | |--- distance > 0.01\n| | | | | | | |--- value: [0.78]\n| | | | | |--- distance > 0.03\n| | | | | | |--- distance
<= 0.10\n| | | | | | | |--- value: [0.46]\n| | | | | | |--- distance > 0.10\n| | | | | | | |--- value: [0.31]\n| | | | |--- distance
> 0.30\n| | | | | |--- value: [1.28]\n| | |--- car_or_bus > 1.50\n| | | |--- distance <= 0.06\n| | | | |--- distance <= 0.04\n| | | | | |---
distance <= 0.04\n| | | | | | |--- distance <= 0.02\n| | | | | | | |--- rating <= 1.50\n| | | | | | | | |--- value: [0.20]\n| | | |
| | | |--- rating > 1.50\n| | | | | | | | |--- distance <= 0.01\n| | | | | | | | | |--- rating <= 2.50\n| | | | | | | | |
| |--- distance <= 0.01\n| | | | | | | | | | | |--- value: [2.74]\n| | | | | | | | | | |--- distance > 0.01\n| | | | | | |
| | | | |--- value: [1.55]\n| | | | | | | | | |--- rating > 2.50\n| | | | | | | | | | |--- rating_bus <= 1.50\n| | | | | |
| | | | | |--- truncated branch of depth 2\n| | | | | | | | | | |--- rating_bus > 1.50\n| | | | | | | | | | | |--- value: [1.02]
\n| | | | | | | | |--- distance > 0.01\n| | | | | | | | | |--- value: [3.46]\n| | | | | | |--- distance > 0.02\n| | | | | |
| |--- distance <= 0.02\n| | | | | | | | |--- value: [4.68]\n| | | | | | | |--- distance > 0.02\n| | | | | | | | |--- distance <= 0.
03\n| | | | | | | | | |--- rating_bus <= 0.50\n| | | | | | | | | | |--- value: [0.89]\n| | | | | | | | | |--- rating_bus >
0.50\n| | | | | | | | | | |--- value: [1.37]\n| | | | | | | | |--- distance > 0.03\n| | | | | | | | | |--- rating <= 1.50\n|
| | | | | | | | | |--- value: [2.88]\n| | | | | | | | | |--- rating > 1.50\n| | | | | | | | | | |--- value: [3.24]\n| |
| | | |--- distance > 0.04\n| | | | | | |--- value: [0.18]\n| | | | |--- distance > 0.04\n| | | | | |--- value: [4.87]\n| | | |--- distan
ce > 0.06\n| | | | |--- distance <= 0.11\n| | | | | |--- value: [0.17]\n| | | | |--- distance > 0.11\n| | | | | |--- distance <= 0.15\n| |
| | | | |--- value: [1.45]\n| | | | | |--- distance > 0.15\n| | | | | | |--- value: [1.26]\n|--- distance > 0.57\n| |--- distance <= 4.45\n| |
|--- distance <= 4.07\n| | | |--- distance <= 3.38\n| | | | |--- rating <= 2.50\n| | | | | |--- car_or_bus <= 1.50\n| | | | | | |--- value: [9.4
5]\n| | | | | |--- car_or_bus > 1.50\n| | | | | | |--- value: [16.78]\n| | | | |--- rating > 2.50\n| | | | | |--- distance <= 1.49\n| |
| | | | |--- distance <= 1.03\n| | | | | | | |--- distance <= 0.80\n| | | | | | | | |--- value: [17.41]\n| | | | | | | |--- dista
nce > 0.80\n| | | | | | | | |--- value: [13.56]\n| | | | | | |--- distance > 1.03\n| | | | | | | |--- value: [6.77]\n| | | | |
|--- distance > 1.49\n| | | | | | |--- distance <= 3.21\n| | | | | | | |--- distance <= 2.99\n| | | | | | | | |--- distance <= 2.11\n| |
| | | | | | | |--- value: [22.32]\n| | | | | | | | |--- distance > 2.11\n| | | | | | | | | |--- distance <= 2.75\n| | | | |
| | | | | |--- distance <= 2.53\n| | | | | | | | | | | |--- truncated branch of depth 2\n| | | | | | | | | | |--- distance > 2.5
3\n| | | | | | | | | | | |--- value: [19.21]\n| | | | | | | | | |--- distance > 2.75\n| | | | | | | | | | |--- value: [1
7.18]\n| | | | | | | |--- distance > 2.99\n| | | | | | | | |--- distance <= 3.13\n| | | | | | | | | |--- value: [28.34]\n| | |
| | | | | |--- distance > 3.13\n| | | | | | | | | |--- value: [21.81]\n| | | | | | |--- distance > 3.21\n| | | | | | | |--- d
istance <= 3.28\n| | | | | | | | |--- value: [10.29]\n| | | | | | | |--- distance > 3.28\n| | | | | | | | |--- value: [15.04]\n| |
| |--- distance > 3.38\n| | | | |--- distance <= 3.61\n| | | | | |--- value: [33.20]\n| | | | |--- distance > 3.61\n| | | | | |--- distance <
= 3.87\n| | | | | | |--- value: [23.50]\n| | | | | |--- distance > 3.87\n| | | | | | |--- value: [25.81]\n| | |--- distance > 4.07\n| | |
|--- car_or_bus <= 1.50\n| | | | |--- value: [8.69]\n| | | |--- car_or_bus > 1.50\n| | | | |--- value: [7.14]\n| |--- distance > 4.45\n| | |--- car_or_
bus <= 1.50\n| | | |--- rating <= 2.50\n| | | | |--- distance <= 12.58\n| | | | | |--- distance <= 5.94\n| | | | | | |--- value: [23.83]\n| |
| | | |--- distance > 5.94\n| | | | | | |--- distance <= 7.55\n| | | | | | | |--- distance <= 7.26\n| | | | | | | | |--- value: [13.
56]\n| | | | | | | |--- distance > 7.26\n| | | | | | | | |--- value: [16.26]\n| | | | | | |--- distance > 7.55\n| | | | | | |
|--- distance <= 9.21\n| | | | | | | | |--- distance <= 8.46\n| | | | | | | | | |--- value: [22.38]\n| | | | | | | | |--- distance
> 8.46\n| | | | | | | | | |--- value: [25.65]\n| | | | | | | |--- distance > 9.21\n| | | | | | | | |--- rating <= 1.50\n| | |
| | | | | | |--- value: [17.65]\n| | | | | | | | |--- rating > 1.50\n| | | | | | | | | |--- value: [14.63]\n| | | | |--- dista
nce > 12.58\n| | | | | |--- value: [25.96]\n| | | |--- rating > 2.50\n| | | | |--- distance <= 8.96\n| | | | | |--- distance <= 6.59\n| | |
| | | |--- distance <= 5.39\n| | | | | | | |--- distance <= 5.11\n| | | | | | | | |--- distance <= 4.74\n| | | | | | | | | |---
value: [25.56]\n| | | | | | | | |--- distance > 4.74\n| | | | | | | | | |--- value: [25.45]\n| | | | | | | |--- distance > 5.11\n|
| | | | | | | |--- distance <= 5.34\n| | | | | | | | | |--- value: [30.85]\n| | | | | | | | |--- distance > 5.34\n| | | | |
| | | | |--- value: [32.38]\n| | | | | | |--- distance > 5.39\n| | | | | | | |--- distance <= 6.17\n| | | | | | | | |--- distance
<= 5.77\n| | | | | | | | | |--- value: [21.67]\n| | | | | | | | |--- distance > 5.77\n| | | | | | | | | |--- value: [21.22]\n|
| | | | | | |--- distance > 6.17\n| | | | | | | | |--- distance <= 6.47\n| | | | | | | | | |--- distance <= 6.29\n| | | | |
| | | | | |--- value: [26.90]\n| | | | | | | | | |--- distance > 6.29\n| | | | | | | | | | |--- value: [28.14]\n| | | | |
| | | |--- distance > 6.47\n| | | | | | | | | |--- value: [19.93]\n| | | | | |--- distance > 6.59\n| | | | | | |--- distance <= 8.36
\n| | | | | | | |--- distance <= 8.17\n| | | | | | | | |--- distance <= 7.40\n| | | | | | | | | |--- distance <= 6.98\n| | | |
| | | | | | |--- value: [32.52]\n| | | | | | | | | |--- distance > 6.98\n| | | | | | | | | | |--- value: [36.65]\n| | | |
| | | | |--- distance > 7.40\n| | | | | | | | | |--- distance <= 7.87\n| | | | | | | | | | |--- distance <= 7.59\n| | | | |
| | | | | | |--- value: [31.09]\n| | | | | | | | | | |--- distance > 7.59\n| | | | | | | | | | | |--- value: [30.17]\n| |
| | | | | | | |--- distance > 7.87\n| | | | | | | | | | |--- value: [31.37]\n| | | | | | | |--- distance > 8.17\n| | | | |
| | | |--- value: [22.47]\n| | | | | | |--- distance > 8.36\n| | | | | | | |--- value: [36.85]\n| | | | |--- distance > 8.96\n| | | |
| |--- distance <= 9.81\n| | | | | | |--- value: [16.89]\n| | | | | |--- distance > 9.81\n| | | | | | |--- distance <= 10.10\n| | | | |
| | |--- value: [21.24]\n| | | | | | |--- distance > 10.10\n| | | | | | | |--- distance <= 10.13\n| | | | | | | | |--- value: [19.83]
\n| | | | | | | |--- distance > 10.13\n| | | | | | | | |--- value: [20.58]\n| | |--- car_or_bus > 1.50\n| | | |--- distance <= 6.70\n| |
| | |--- distance <= 6.09\n| | | | | |--- rating_bus <= 1.50\n| | | | | | |--- distance <= 5.51\n| | | | | | | |--- value: [15.62]\n| | |
| | | |--- distance > 5.51\n| | | | | | | |--- value: [17.54]\n| | | | | |--- rating_bus > 1.50\n| | | | | | |--- distance <= 5.14\n| |
| | | | | |--- value: [24.50]\n| | | | | | |--- distance > 5.14\n| | | | | | | |--- value: [23.20]\n| | | | |--- distance > 6.09\n| |
| | | |--- rating_bus <= 1.50\n| | | | | | |--- value: [29.54]\n| | | | | |--- rating_bus > 1.50\n| | | | | | |--- distance <= 6.46\n| |
| | | | | |--- value: [23.78]\n| | | | | | |--- distance > 6.46\n| | | | | | | |--- value: [21.20]\n| | | |--- distance > 6.70\n| | |
| |--- distance <= 9.84\n| | | | | |--- distance <= 8.28\n| | | | | | |--- rating_weather <= 0.50\n| | | | | | | |--- rating <= 2.50\n| | |
| | | | | |--- value: [12.61]\n| | | | | | | |--- rating > 2.50\n| | | | | | | | |--- distance <= 7.91\n| | | | | | | | |
|--- value: [19.77]\n| | | | | | | | |--- distance > 7.91\n| | | | | | | | | |--- value: [15.40]\n| | | | | | |--- rating_weather >
0.50\n| | | | | | | |--- distance <= 6.91\n| | | | | | | | |--- rating <= 2.00\n| | | | | | | | | |--- value: [19.38]\n| | | |
| | | | |--- rating > 2.00\n| | | | | | | | | |--- value: [14.90]\n| | | | | | | |--- distance > 6.91\n| | | | | | | | |---
distance <= 7.18\n| | | | | | | | | |--- value: [24.00]\n| | | | | | | | |--- distance > 7.18\n| | | | | | | | | |--- rating_bus
<= 1.50\n| | | | | | | | | | |--- distance <= 8.07\n| | | | | | | | | | | |--- value: [20.37]\n| | | | | | | | | | |---
distance > 8.07\n| | | | | | | | | | | |--- truncated branch of depth 2\n| | | | | | | | | |--- rating_bus > 1.50\n| | | | | |
| | | | |--- value: [16.69]\n| | | | | |--- distance > 8.28\n| | | | | | |--- value: [10.94]\n| | | | |--- distance > 9.84\n| | | | |
|--- value: [25.55]\n',
'|--- distance <= 4.07\n| |--- distance <= 0.05\n| | |--- rating <= 1.50\n| | | |--- distance <= 0.01\n| | | | |--- rating_bus <= 0.50\n| | | | | |---
value: [0.06]\n| | | | |--- rating_bus > 0.50\n| | | | | |--- value: [0.31]\n| | | |--- distance > 0.01\n| | | | |--- rating_weather <= 1.00\n| |
| | | |--- value: [0.09]\n| | | | |--- rating_weather > 1.00\n| | | | | |--- value: [0.01]\n| | |--- rating > 1.50\n| | | |--- distance <= 0.04\n
| | | | |--- car_or_bus <= 1.50\n| | | | | |--- rating <= 2.50\n| | | | | | |--- distance <= 0.00\n| | | | | | | |--- value: [0.02]\n|
| | | | | |--- distance > 0.00\n| | | | | | | |--- distance <= 0.01\n| | | | | | | | |--- value: [0.00]\n| | | | | | | |--- di
stance > 0.01\n| | | | | | | | |--- distance <= 0.01\n| | | | | | | | | |--- value: [0.01]\n| | | | | | | | |--- distance > 0.01
\n| | | | | | | | | |--- value: [0.02]\n| | | | | |--- rating > 2.50\n| | | | | | |--- distance <= 0.01\n| | | | | | | |--- di
stance <= 0.00\n| | | | | | | | |--- value: [0.02]\n| | | | | | | |--- distance > 0.00\n| | | | | | | | |--- value: [0.00]\n| | |
| | | |--- distance > 0.01\n| | | | | | | |--- distance <= 0.01\n| | | | | | | | |--- value: [0.10]\n| | | | | | | |--- distance >
0.01\n| | | | | | | | |--- value: [0.01]\n| | | | |--- car_or_bus > 1.50\n| | | | | |--- distance <= 0.02\n| | | | | | |--- distance <
= 0.01\n| | | | | | | |--- distance <= 0.00\n| | | | | | | | |--- value: [0.01]\n| | | | | | | |--- distance > 0.00\n| | | | |
| | | |--- value: [0.00]\n| | | | | | |--- distance > 0.01\n| | | | | | | |--- distance <= 0.01\n| | | | | | | | |--- rating_bus <=
0.50\n| | | | | | | | | |--- value: [0.02]\n| | | | | | | | |--- rating_bus > 0.50\n| | | | | | | | | |--- value: [0.01]\n| |
| | | | | |--- distance > 0.01\n| | | | | | | | |--- distance <= 0.01\n| | | | | | | | | |--- rating_weather <= 0.50\n| | | | |
| | | | | |--- distance <= 0.01\n| | | | | | | | | | | |--- value: [0.00]\n| | | | | | | | | | |--- distance > 0.01\n| | |
| | | | | | | | |--- value: [0.01]\n| | | | | | | | | |--- rating_weather > 0.50\n| | | | | | | | | | |--- distance <= 0.01\n|
| | | | | | | | | | |--- value: [0.01]\n| | | | | | | | | | |--- distance > 0.01\n| | | | | | | | | | | |--- value:
[0.01]\n| | | | | | | | |--- distance > 0.01\n| | | | | | | | | |--- value: [0.00]\n| | | | | |--- distance > 0.02\n| | | | |
| |--- distance <= 0.03\n| | | | | | | |--- distance <= 0.03\n| | | | | | | | |--- value: [0.02]\n| | | | | | | |--- distance > 0.03\n
| | | | | | | | |--- value: [0.03]\n| | | | | | |--- distance > 0.03\n| | | | | | | |--- value: [0.01]\n| | | |--- distance > 0.04
\n| | | | |--- rating_weather <= 1.50\n| | | | | |--- value: [0.01]\n| | | | |--- rating_weather > 1.50\n| | | | | |--- value: [0.21]\n| |---
distance > 0.05\n| | |--- distance <= 0.57\n| | | |--- rating_bus <= 1.50\n| | | | |--- distance <= 0.07\n| | | | | |--- value: [0.11]\n| | | |
|--- distance > 0.07\n| | | | | |--- distance <= 0.15\n| | | | | | |--- value: [0.46]\n| | | | | |--- distance > 0.15\n| | | | | | |---
car_or_bus <= 1.50\n| | | | | | | |--- value: [0.36]\n| | | | | | |--- car_or_bus > 1.50\n| | | | | | | |--- value: [0.13]\n| | | |---
rating_bus > 1.50\n| | | | |--- value: [0.09]\n| | |--- distance > 0.57\n| | | |--- distance <= 2.51\n| | | | |--- distance <= 1.03\n| | | | |
|--- distance <= 0.80\n| | | | | | |--- value: [0.04]\n| | | | | |--- distance > 0.80\n| | | | | | |--- value: [0.07]\n| | | | |--- distan
ce > 1.03\n| | | | | |--- distance <= 1.49\n| | | | | | |--- value: [0.17]\n| | | | | |--- distance > 1.49\n| | | | | | |--- distance <
= 2.11\n| | | | | | | |--- value: [0.08]\n| | | | | | |--- distance > 2.11\n| | | | | | | |--- value: [0.13]\n| | | |--- distance >
2.51\n| | | | |--- distance <= 2.63\n| | | | | |--- value: [0.28]\n| | | | |--- distance > 2.63\n| | | | | |--- distance <= 3.19\n| | | |
| | |--- distance <= 2.99\n| | | | | | | |--- distance <= 2.75\n| | | | | | | | |--- value: [0.14]\n| | | | | | | |--- distance > 2.
75\n| | | | | | | | |--- value: [0.17]\n| | | | | | |--- distance > 2.99\n| | | | | | | |--- distance <= 3.13\n| | | | | | |
| |--- value: [0.11]\n| | | | | | | |--- distance > 3.13\n| | | | | | | | |--- value: [0.14]\n| | | | | |--- distance > 3.19\n| | |
| | | |--- distance <= 3.38\n| | | | | | | |--- car_or_bus <= 1.50\n| | | | | | | | |--- distance <= 3.28\n| | | | | | | | | |-
-- value: [0.32]\n| | | | | | | | |--- distance > 3.28\n| | | | | | | | | |--- value: [0.22]\n| | | | | | | |--- car_or_bus > 1.50
\n| | | | | | | | |--- value: [0.19]\n| | | | | | |--- distance > 3.38\n| | | | | | | |--- distance <= 3.61\n| | | | | | | |
|--- value: [0.10]\n| | | | | | | |--- distance > 3.61\n| | | | | | | | |--- rating <= 2.50\n| | | | | | | | | |--- value: [0.15]\n|
| | | | | | | |--- rating > 2.50\n| | | | | | | | | |--- value: [0.16]\n|--- distance > 4.07\n| |--- distance <= 8.07\n| | |--- distance <=
4.45\n| | | |--- rating_bus <= 0.50\n| | | | |--- value: [0.50]\n| | | |--- rating_bus > 0.50\n| | | | |--- value: [0.58]\n| | |--- distance > 4.45
\n| | | |--- distance <= 6.72\n| | | | |--- distance <= 5.39\n| | | | | |--- rating_bus <= 0.50\n| | | | | | |--- distance <= 5.11\n| | | |
| | | |--- rating <= 2.50\n| | | | | | | | |--- value: [0.20]\n| | | | | | | |--- rating > 2.50\n| | | | | | | | |--- distance <
= 4.74\n| | | | | | | | | |--- value: [0.18]\n| | | | | | | | |--- distance > 4.74\n| | | | | | | | | |--- value: [0.19]\n| |
| | | | |--- distance > 5.11\n| | | | | | | |--- value: [0.17]\n| | | | | |--- rating_bus > 0.50\n| | | | | | |--- rating <= 2.00\n|
| | | | | | |--- value: [0.20]\n| | | | | | |--- rating > 2.00\n| | | | | | | |--- value: [0.33]\n| | | | |--- distance > 5.39\n| |
| | | |--- distance <= 6.61\n| | | | | | |--- distance <= 6.47\n| | | | | | | |--- rating <= 2.50\n| | | | | | | | |--- rating <= 1.5
0\n| | | | | | | | | |--- value: [0.27]\n| | | | | | | | |--- rating > 1.50\n| | | | | | | | | |--- value: [0.33]\n| | | |
| | | |--- rating > 2.50\n| | | | | | | | |--- distance <= 6.17\n| | | | | | | | | |--- distance <= 5.79\n| | | | | | | | |
| |--- rating_bus <= 1.00\n| | | | | | | | | | | |--- value: [0.25]\n| | | | | | | | | | |--- rating_bus > 1.00\n| | | | | |
| | | | | |--- value: [0.23]\n| | | | | | | | | |--- distance > 5.79\n| | | | | | | | | | |--- value: [0.29]\n| | | | | |
| | |--- distance > 6.17\n| | | | | | | | | |--- car_or_bus <= 1.50\n| | | | | | | | | | |--- value: [0.23]\n| | | | | | | |
| |--- car_or_bus > 1.50\n| | | | | | | | | | |--- value: [0.21]\n| | | | | | |--- distance > 6.47\n| | | | | | | |--- rating_bus <
= 1.00\n| | | | | | | | |--- value: [0.33]\n| | | | | | | |--- rating_bus > 1.00\n| | | | | | | | |--- value: [0.31]\n| | | |
| |--- distance > 6.61\n| | | | | | |--- value: [0.20]\n| | | |--- distance > 6.72\n| | | | |--- distance <= 7.32\n| | | | | |--- distance <=
7.06\n| | | | | | |--- distance <= 6.91\n| | | | | | | |--- distance <= 6.85\n| | | | | | | | |--- value: [0.35]\n| | | | | | |
|--- distance > 6.85\n| | | | | | | | |--- value: [0.46]\n| | | | | | |--- distance > 6.91\n| | | | | | | |--- value: [0.29]\n| | |
| | |--- distance > 7.06\n| | | | | | |--- distance <= 7.25\n| | | | | | | |--- value: [0.53]\n| | | | | | |--- distance > 7.25\n| |
| | | | | |--- value: [0.58]\n| | | | |--- distance > 7.32\n| | | | | |--- car_or_bus <= 1.50\n| | | | | | |--- rating <= 2.50\n| | |
| | | | |--- distance <= 7.55\n| | | | | | | | |--- value: [0.45]\n| | | | | | | |--- distance > 7.55\n| | | | | | | | |--- va
lue: [0.35]\n| | | | | | |--- rating > 2.50\n| | | | | | | |--- distance <= 7.40\n| | | | | | | | |--- value: [0.20]\n| | | | |
| | |--- distance > 7.40\n| | | | | | | | |--- distance <= 7.59\n| | | | | | | | | |--- value: [0.24]\n| | | | | | | | |--- di
stance > 7.59\n| | | | | | | | | |--- value: [0.26]\n| | | | | |--- car_or_bus > 1.50\n| | | | | | |--- distance <= 7.57\n| | | | |
| | |--- value: [0.45]\n| | | | | | |--- distance > 7.57\n| | | | | | | |--- value: [0.39]\n| |--- distance > 8.07\n| | |--- distance <= 9.21\n
| | | |--- car_or_bus <= 1.50\n| | | | |--- distance <= 8.36\n| | | | | |--- value: [0.37]\n| | | | |--- distance > 8.36\n| | | | | |--- r
ating <= 2.50\n| | | | | | |--- value: [0.36]\n| | | | | |--- rating > 2.50\n| | | | | | |--- value: [0.23]\n| | | |--- car_or_bus > 1.50\n
| | | | |--- distance <= 8.28\n| | | | | |--- rating_weather <= 0.50\n| | | | | | |--- value: [0.53]\n| | | | | |--- rating_weather > 0.50\n
| | | | | | |--- rating <= 2.00\n| | | | | | | |--- value: [0.44]\n| | | | | | |--- rating > 2.00\n| | | | | | | |--- value: [0.
45]\n| | | | |--- distance > 8.28\n| | | | | |--- value: [0.77]\n| | |--- distance > 9.21\n| | | |--- rating <= 2.50\n| | | | |--- distance <=
12.58\n| | | | | |--- rating <= 1.50\n| | | | | | |--- value: [0.53]\n| | | | | |--- rating > 1.50\n| | | | | | |--- value: [0.77]\n|
| | | |--- distance > 12.58\n| | | | | |--- value: [0.54]\n| | | |--- rating > 2.50\n| | | | |--- distance <= 9.81\n| | | | | |--- value:
[0.56]\n| | | | |--- distance > 9.81\n| | | | | |--- rating_weather <= 1.00\n| | | | | | |--- distance <= 10.10\n| | | | | | | |--- valu
e: [0.48]\n| | | | | | |--- distance > 10.10\n| | | | | | | |--- distance <= 10.13\n| | | | | | | | |--- value: [0.51]\n| | | | |
| | |--- distance > 10.13\n| | | | | | | | |--- value: [0.49]\n| | | | | |--- rating_weather > 1.00\n| | | | | | |--- value: [0.44]\n')
# Plot the tree
plt.figure(figsize=(20,10))
plot_tree(speed_tree, feature_names=X.columns, filled=True, rounded=True, precision=3)
plt.show()
# Create and fit the regression tree model for time
time_tree = DecisionTreeRegressor(random_state=12345)
time_tree.fit(X_train, y_train_time)
# Plot the tree
plt.figure(figsize=(20,10))
plot_tree(time_tree, feature_names=X.columns, filled=True, rounded=True, precision=3)
plt.show()
# Generate predictions for the testing dataset
y_pred_speed = speed_tree.predict(X_test)
y_pred_time = time_tree.predict(X_test)
# Calculate and print the correlation between actual and predicted (Spearman)
from scipy.stats import spearmanr
corr_speed, _ = spearmanr(y_test_speed, y_pred_speed)
print(f'Spearman correlation for speed: {corr_speed}')
corr_time, _ = spearmanr(y_test_time, y_pred_time)
print(f'Spearman correlation for time: {corr_time}')
Spearman correlation for speed: 0.6672602486138792
Spearman correlation for time: 0.668044568522594
# Calculate the mean absolute error (MAE)
def MAE(actual, predicted):
 return mean_absolute_error(actual, predicted)
mae_speed = MAE(y_test_speed, y_pred_speed)
print(f'MAE for speed: {mae_speed}')
mae_time = MAE(y_test_time, y_pred_time)
print(f'MAE for time: {mae_time}')
# Calculate MAE for mean of actual values
mae_speed_mean = MAE(y_test_speed, np.full_like(y_test_speed, np.mean(y_train_speed)))
print(f'MAE for mean speed: {mae_speed_mean}')
mae_time_mean = MAE(y_test_time, np.full_like(y_test_time, np.mean(y_train_time)))
print(f'MAE for mean time: {mae_time_mean}')
# Calculate the root mean square error (RMSE)
def RMSE(actual, predicted):
 return np.sqrt(mean_squared_error(actual, predicted))
rmse_speed = RMSE(y_test_speed, y_pred_speed)
print(f'RMSE for speed: {rmse_speed}')
rmse_time = RMSE(y_test_time, y_pred_time)
print(f'RMSE for time: {rmse_time}')
# Calculate RMSE for mean of actual values
rmse_speed_mean = RMSE(y_test_speed, np.full_like(y_test_speed, np.mean(y_train_speed)))
print(f'RMSE for mean speed: {rmse_speed_mean}')
rmse_time_mean = RMSE(y_test_time, np.full_like(y_test_time, np.mean(y_train_time)))
print(f'RMSE for mean time: {rmse_time_mean}')
# Define MSE function
def MSE(actual, predicted):
 return mean_squared_error(actual, predicted)
# Calculate MSE for speed
mse_speed = MSE(y_test_speed, y_pred_speed)
print(f'MSE for speed: {mse_speed}')
# Calculate MSE for time
mse_time = MSE(y_test_time, y_pred_time)
print(f'MSE for time: {mse_time}')
# Calculate MSE for mean of actual values
mse_speed_mean = MSE(y_test_speed, np.full_like(y_test_speed, np.mean(y_train_speed)))
print(f'MSE for mean speed: {mse_speed_mean}')
mse_time_mean = MSE(y_test_time, np.full_like(y_test_time, np.mean(y_train_time)))
print(f'MSE for mean time: {mse_time_mean}')
MAE for speed: 6.266333333333334
MAE for time: 0.10266666666666668
MAE for mean speed: 9.309192090395479
MAE for mean time: 0.151090395480226
RMSE for speed: 8.696808418418026
RMSE for time: 0.14592235378218557
RMSE for mean speed: 10.999448077219478
RMSE for mean time: 0.18036452975249126
MSE for speed: 75.63447666666666
MSE for time: 0.02129333333333333
MSE for mean speed: 120.98785800344727
MSE for mean time: 0.03253136359283731
#SVR
# Initialize the SVR model with RBF kernel
svm_model = SVR(kernel='rbf', C=10, gamma=0.1)
# Initialize and train the SVR model for speed with RBF kernel and sigma equivalent to gamma=0.1
svr_speed = SVR(kernel='rbf', C=10, gamma=0.1) # gamma is equivalent to sigma in R
svr_speed.fit(X_train, y_train_speed)
# Initialize and train the SVR model for time with RBF kernel
svr_time = SVR(kernel='rbf', C=10, gamma=0.1)
svr_time.fit(X_train, y_train_time)
# Summary information (in scikit-learn, you can inspect the model directly)
print("SVR Model for Speed:", svr_speed)
print("SVR Model for Time:", svr_time)
# Predict on the test set
y_pred_speed = svr_speed.predict(X_test)
y_pred_time = svr_time.predict(X_test)
# Calculate the errors and metrics for speed
mse_speed = mean_squared_error(y_test_speed, y_pred_speed)
rmse_speed = np.sqrt(mse_speed)
mae_speed = mean_absolute_error(y_test_speed, y_pred_speed)
# Calculate the errors and metrics for time
mse_time = mean_squared_error(y_test_time, y_pred_time)
rmse_time = np.sqrt(mse_time)
mae_time = mean_absolute_error(y_test_time, y_pred_time)
# Create a table to compare predictions with actual values for speed
comparison_df_speed = pd.DataFrame({
 'Predicted': np.round(y_pred_speed, 1),
 'Actual': np.round(y_test_speed, 1)
})
# Create a table to compare predictions with actual values for time
comparison_df_time = pd.DataFrame({
 'Predicted': np.round(y_pred_time, 1),
 'Actual': np.round(y_test_time, 1)
})
# Calculate the frequency table and proportion table for speed
frequency_table_speed = pd.crosstab(comparison_df_speed['Predicted'], comparison_df_speed['Actual'])
prop_table_speed = np.round(frequency_table_speed / np.sum(frequency_table_speed.values) * 100, 1)
accuracy_speed = np.sum(np.diag(prop_table_speed.values))
# Calculate the frequency table and proportion table for time
frequency_table_time = pd.crosstab(comparison_df_time['Predicted'], comparison_df_time['Actual'])
prop_table_time = np.round(frequency_table_time / np.sum(frequency_table_time.values) * 100, 1)
accuracy_time = np.sum(np.diag(prop_table_time.values))
# Print the results for speed
print("Frequency Table for Speed:\n", frequency_table_speed)
print("\nProportion Table for Speed:\n", prop_table_speed)
print("\nAccuracy for Speed: ", accuracy_speed)
print("\nRMSE for Speed: ", rmse_speed)
print("MAE for Speed: ", mae_speed)
print("MSE for Speed: ", mse_speed)
# Print the results for time
print("\nFrequency Table for Time:\n", frequency_table_time)
print("\nProportion Table for Time:\n", prop_table_time)
print("\nAccuracy for Time: ", accuracy_time)
print("\nRMSE for Time: ", rmse_time)
print("MAE for Time: ", mae_time)
print("MSE for Time: ", mse_time)
SVR Model for Speed: SVR(C=10, gamma=0.1)
SVR Model for Time: SVR(C=10, gamma=0.1)
Frequency Table for Speed:
Actual 0.1 0.2 0.3 0.4 0.5 0.6 1.3 6.8 8.9 9.1 ... \
Predicted ...
0.2 0 1 0 1 0 0 0 0 0 0 ...
0.3 1 0 0 0 0 0 0 0 0 0 ...
0.6 0 0 1 0 0 0 0 0 0 0 ...
0.8 0 0 0 0 0 0 0 0 1 0 ...
1.5 0 0 0 0 1 0 0 0 0 0 ...
1.6 0 0 0 0 0 1 1 0 0 0 ...
3.5 0 0 0 0 0 0 0 0 0 0 ...
7.4 0 0 0 0 0 0 0 0 0 0 ...
8.8 0 0 0 0 0 0 0 0 0 0 ...
17.1 0 0 0 0 0 0 0 0 0 0 ...
18.7 0 0 0 0 0 0 0 1 0 0 ...
18.9 0 0 0 0 0 0 0 0 0 0 ...
19.3 0 0 0 0 0 0 0 0 0 0 ...
20.4 0 0 0 0 0 0 0 0 0 0 ...
22.0 0 0 0 0 0 0 0 0 0 0 ...
22.1 0 0 0 0 0 0 0 0 0 1 ...
22.8 0 0 0 0 0 0 0 0 0 0 ...
23.1 0 0 0 0 0 0 0 0 0 0 ...
23.9 0 0 0 0 0 0 0 0 0 0 ...
24.1 0 0 0 0 0 0 0 0 0 0 ...
25.6 0 0 0 0 0 0 0 0 0 0 ...
25.7 0 0 0 0 0 0 0 0 0 0 ...
26.0 0 0 0 0 0 0 0 0 0 0 ...
26.2 0 0 0 0 0 0 0 0 0 0 ...
26.8 0 0 0 0 0 0 0 0 0 0 ...
27.0 0 0 0 0 0 0 0 0 0 0 ...
Actual 21.5 23.0 24.4 24.9 25.4 27.5 28.1 30.0 33.2 38.0
Predicted
0.2 0 0 0 0 0 0 0 0 0 0
0.3 0 0 0 0 0 0 0 0 0 0
0.6 0 0 0 0 0 0 0 0 0 0
0.8 0 0 0 0 0 0 0 0 0 0
1.5 0 0 0 0 0 0 0 0 0 0
1.6 0 0 0 0 0 0 0 0 0 0
3.5 0 0 0 0 0 0 0 0 0 0
7.4 0 0 0 0 0 0 0 0 0 0
8.8 0 0 0 0 0 0 0 0 0 0
17.1 0 0 0 0 0 0 0 0 0 0
18.7 0 0 0 0 0 0 0 0 0 0
18.9 0 0 0 0 0 0 0 0 0 0
19.3 0 0 0 0 0 0 0 0 0 0
20.4 0 0 1 0 0 0 0 0 0 0
22.0 0 0 0 0 0 0 0 0 0 0
22.1 0 0 0 0 0 0 0 0 0 0
22.8 0 0 0 0 0 0 0 0 1 0
23.1 0 0 0 1 0 0 0 1 0 0
23.9 0 0 0 0 0 0 0 0 0 0
24.1 0 0 0 0 0 0 0 0 0 0
25.6 0 0 0 0 1 0 0 0 0 0
25.7 0 0 0 0 0 1 0 0 0 0
26.0 1 0 0 0 0 0 0 0 0 0
26.2 0 0 0 0 0 0 1 0 0 0
26.8 0 1 0 0 0 0 0 0 0 0
27.0 0 0 0 0 0 0 0 0 0 1
[26 rows x 29 columns]
Proportion Table for Speed:
Actual 0.1 0.2 0.3 0.4 0.5 0.6 1.3 6.8 8.9 9.1 ... \
Predicted ...
0.2 0.0 3.3 0.0 3.3 0.0 0.0 0.0 0.0 0.0 0.0 ...
0.3 3.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
0.6 0.0 0.0 3.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
0.8 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.3 0.0 ...
1.5 0.0 0.0 0.0 0.0 3.3 0.0 0.0 0.0 0.0 0.0 ...
1.6 0.0 0.0 0.0 0.0 0.0 3.3 3.3 0.0 0.0 0.0 ...
3.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
7.4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
8.8 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
17.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
18.7 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.3 0.0 0.0 ...
18.9 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
19.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
20.4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
22.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
22.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.3 ...
22.8 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
23.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
23.9 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
24.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
25.6 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
25.7 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
26.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
26.2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
26.8 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
27.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
Actual 21.5 23.0 24.4 24.9 25.4 27.5 28.1 30.0 33.2 38.0
Predicted
0.2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.6 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.8 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
1.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
1.6 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
3.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
7.4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
8.8 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
17.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
18.7 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
18.9 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
19.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
20.4 0.0 0.0 3.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0
22.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
22.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
22.8 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.3 0.0
23.1 0.0 0.0 0.0 3.3 0.0 0.0 0.0 3.3 0.0 0.0
23.9 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
24.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
25.6 0.0 0.0 0.0 0.0 3.3 0.0 0.0 0.0 0.0 0.0
25.7 0.0 0.0 0.0 0.0 0.0 3.3 0.0 0.0 0.0 0.0
26.0 3.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
26.2 0.0 0.0 0.0 0.0 0.0 0.0 3.3 0.0 0.0 0.0
26.8 0.0 3.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
27.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.3
[26 rows x 29 columns]
Accuracy for Speed: 9.899999999999999
RMSE for Speed: 6.537510694236151
MAE for Speed: 4.690395336377388
MSE for Speed: 42.73904607725205
Frequency Table for Time:
Actual 0.0 0.1 0.2 0.3 0.4 0.5 0.7
Predicted
0.1 7 0 0 1 0 1 0
0.2 0 3 1 1 0 0 0
0.3 0 1 2 4 0 2 0
0.4 0 0 1 1 1 1 0
0.5 0 0 0 0 1 0 1
0.6 0 0 0 0 1 0 0
Proportion Table for Time:
Actual 0.0 0.1 0.2 0.3 0.4 0.5 0.7
Predicted
0.1 23.3 0.0 0.0 3.3 0.0 3.3 0.0
0.2 0.0 10.0 3.3 3.3 0.0 0.0 0.0
0.3 0.0 3.3 6.7 13.3 0.0 6.7 0.0
0.4 0.0 0.0 3.3 3.3 3.3 3.3 0.0
0.5 0.0 0.0 0.0 0.0 3.3 0.0 3.3
0.6 0.0 0.0 0.0 0.0 3.3 0.0 0.0
Accuracy for Time: 46.599999999999994
RMSE for Time: 0.1273545504047673
MAE for Time: 0.10190975740719556
MSE for Time: 0.01621918150880042
# Random forest in Regression Mode
# Random forest Regressor
# Fit the random forest model for predicting speed
rf_speed = RandomForestRegressor(random_state=42)
rf_speed.fit(X_train, y_train_speed)
# Print the summary of the model
print(rf_speed)
# Calculate and plot variable importance for speed prediction
importances_speed = rf_speed.feature_importances_
indices_speed = importances_speed.argsort()[::-1]
feature_names = X_train.columns
# Create a proportion table for speed
proportions_speed = importances_speed / np.sum(importances_speed)
proportion_table_speed = pd.DataFrame({
 'Feature': feature_names[indices_speed],
 'Importance': importances_speed[indices_speed],
 'Proportion (%)': proportions_speed[indices_speed] * 100
})
print("Proportion Table for Speed Prediction:")
print(proportion_table_speed)
plt.figure(figsize=(10, 6))
plt.title("Random Forest - Variable Importance (Speed)")
plt.bar(range(X_train.shape[1]), importances_speed[indices_speed], align="center")
plt.xticks(range(X_train.shape[1]), feature_names[indices_speed], rotation=90)
plt.tight_layout()
plt.show()
# Fit the random forest model for predicting time
rf_time = RandomForestRegressor(random_state=42)
rf_time.fit(X_train, y_train_time)
# Print the summary of the model
print(rf_time)
# Calculate and plot variable importance for time prediction
importances_time = rf_time.feature_importances_
indices_time = importances_time.argsort()[::-1]
# Create a proportion table for time
proportions_time = importances_time / np.sum(importances_time)
proportion_table_time = pd.DataFrame({
 'Feature': feature_names[indices_time],
 'Importance': importances_time[indices_time],
 'Proportion (%)': proportions_time[indices_time] * 100
})
print("Proportion Table for Time Prediction:")
print(proportion_table_time)
plt.figure(figsize=(10, 6))
plt.title("Random Forest - Variable Importance (Time)")
plt.bar(range(X_train.shape[1]), importances_time[indices_time], align="center")
plt.xticks(range(X_train.shape[1]), feature_names[indices_time], rotation=90)
plt.tight_layout()
plt.show()
# Initialize the Random Forest Regressor
rf_speed = RandomForestRegressor(random_state=42)
rf_time = RandomForestRegressor(random_state=42)
# Train the Random Forest Regressor
rf_speed.fit(X_train, y_train_speed)
rf_time.fit(X_train, y_train_time)
# Predict on the test set
y_pred_speed = rf_speed.predict(X_test)
y_pred_time = rf_time.predict(X_test)
# Calculate RMSE and MAE for speed
rmse_speed = np.sqrt(mean_squared_error(y_test_speed, y_pred_speed))
mae_speed = mean_absolute_error(y_test_speed, y_pred_speed)
mse_speed = mean_squared_error(y_test_speed, y_pred_speed)
# Calculate RMSE and MAE for time
rmse_time = np.sqrt(mean_squared_error(y_test_time, y_pred_time))
mae_time = mean_absolute_error(y_test_time, y_pred_time)
mse_time = mean_squared_error(y_test_time, y_pred_time)
# Print the results
print(f"Speed - MSE: {mse_speed:.2f}, RMSE: {rmse_speed:.2f}, MAE: {mae_speed:.2f}")
print(f"Time - MSE: {mse_time:.2f}, RMSE: {rmse_time:.2f}, MAE: {mae_time:.2f}")
RandomForestRegressor(random_state=42)
Proportion Table for Speed Prediction:
 Feature Importance Proportion (%)
0 distance 0.935585 93.558485
1 car_or_bus 0.030248 3.024775
2 rating 0.024756 2.475568
3 rating_bus 0.006091 0.609125
4 rating_weather 0.003320 0.332048
RandomForestRegressor(random_state=42)
Proportion Table for Time Prediction:
 Feature Importance Proportion (%)
0 distance 0.911207 91.120715
1 rating 0.040585 4.058534
2 car_or_bus 0.025376 2.537608
3 rating_bus 0.011456 1.145630
4 rating_weather 0.011375 1.137513
Speed - MSE: 50.71, RMSE: 7.12, MAE: 5.49
Time - MSE: 0.01, RMSE: 0.11, MAE: 0.08
#KNN
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
# Initialize the KNN regressor
knn_speed = KNeighborsRegressor(n_neighbors=5)
knn_time = KNeighborsRegressor(n_neighbors=5)
# Perform cross-validation predictions
y_train_speed_pred = cross_val_predict(knn_speed, X_train, y_train_speed, cv=5)
y_train_time_pred = cross_val_predict(knn_time, X_train, y_train_time, cv=5)
# Fit the model on the full training set
knn_speed.fit(X_train, y_train_speed)
knn_time.fit(X_train, y_train_time)
# Predict on the test set
y_test_speed_pred = knn_speed.predict(X_test)
y_test_time_pred = knn_time.predict(X_test)
# Create a cross table to compare predicted and actual values
cross_table_speed = pd.DataFrame({
 'Actual Speed': y_test_speed,
 'Predicted Speed': y_test_speed_pred
})
cross_table_time = pd.DataFrame({
 'Actual Time': y_test_time,
 'Predicted Time': y_test_time_pred
})
# Display the cross tables
print("\nCross Table for Speed:")
print(cross_table_speed.head())
print("\nCross Table for Time:")
print(cross_table_time.head())
# Calculate errors for Speed
mse_speed = mean_squared_error(y_test_speed, y_test_speed_pred)
rmse_speed = np.sqrt(mse_speed)
mae_speed = mean_absolute_error(y_test_speed, y_test_speed_pred)
# Calculate errors for Time
mse_time = mean_squared_error(y_test_time, y_test_time_pred)
rmse_time = np.sqrt(mse_time)
mae_time = mean_absolute_error(y_test_time, y_test_time_pred)
# Print the error metrics
print(f"Speed Mean Squared Error (MSE): {mse_speed}")
print(f"Speed Root Mean Squared Error (RMSE): {rmse_speed}")
print(f"Speed Mean Absolute Error (MAE): {mae_speed}")
print(f"\nTime Mean Squared Error (MSE): {mse_time}")
print(f"Time Root Mean Squared Error (RMSE): {rmse_time}")
print(f"Time Mean Absolute Error (MAE): {mae_time}")
Cross Table for Speed:
 Actual Speed Predicted Speed
139 25.45 27.182
61 0.31 0.878
153 38.03 26.878
19 20.80 22.276
116 0.37 0.922
Cross Table for Time:
 Actual Time Predicted Time
139 0.19 0.192
61 0.46 0.120
153 0.17 0.244
19 0.43 0.416
116 0.02 0.010
Speed Mean Squared Error (MSE): 38.34753093333333
Speed Root Mean Squared Error (RMSE): 6.192538327159012
Speed Mean Absolute Error (MAE): 4.5768
Time Mean Squared Error (MSE): 0.012239999999999997
Time Root Mean Squared Error (RMSE): 0.11063453348751463
Time Mean Absolute Error (MAE): 0.06866666666666668
# Comparision RMSE, MAE, and MSE values for each model
metrics = {
 'Model': ['Decision Tree', 'SVR', 'Random Forest', 'KNN'],
 'RMSE Speed': [3.45, 2.98, 3.15, 3.21], # Replace with actual RMSE values
 'MAE Speed': [2.56, 2.24, 2.34, 2.42], # Replace with actual MAE values
 'MSE Speed': [11.9, 8.88, 9.92, 10.31], # Replace with actual MSE values
 'RMSE Time': [4.12, 3.67, 3.89, 3.95], # Replace with actual RMSE values
 'MAE Time': [3.23, 2.89, 3.01, 3.12], # Replace with actual MAE values
 'MSE Time': [16.98, 13.47, 15.13, 15.60] # Replace with actual MSE values
}
# Convert to DataFrame
metrics_df = pd.DataFrame(metrics)
# Plot histograms for RMSE, MAE, and MSE for Speed
plt.figure(figsize=(18, 6))
# RMSE Speed
plt.subplot(1, 3, 1)
plt.bar(metrics_df['Model'], metrics_df['RMSE Speed'], color='blue')
plt.title('RMSE for Speed')
plt.ylabel('RMSE')
# MAE Speed
plt.subplot(1, 3, 2)
plt.bar(metrics_df['Model'], metrics_df['MAE Speed'], color='orange')
plt.title('MAE for Speed')
plt.ylabel('MAE')
# MSE Speed
plt.subplot(1, 3, 3)
plt.bar(metrics_df['Model'], metrics_df['MSE Speed'], color='green')
plt.title('MSE for Speed')
plt.ylabel('MSE')
plt.tight_layout()
plt.show()
# Plot histograms for RMSE, MAE, and MSE for Time
plt.figure(figsize=(18, 6))
# RMSE Time
plt.subplot(1, 3, 1)
plt.bar(metrics_df['Model'], metrics_df['RMSE Time'], color='blue')
plt.title('RMSE for Time')
plt.ylabel('RMSE')
# MAE Time
plt.subplot(1, 3, 2)
plt.bar(metrics_df['Model'], metrics_df['MAE Time'], color='orange')
plt.title('MAE for Time')
plt.ylabel('MAE')
# MSE Time
plt.subplot(1, 3, 3)
plt.bar(metrics_df['Model'], metrics_df['MSE Time'], color='green')
plt.title('MSE for Time')
plt.ylabel('MSE')
plt.tight_layout()
plt.show()
In [23]: 
