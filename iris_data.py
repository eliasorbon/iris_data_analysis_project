import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv(url, names=column_names)

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.drop("species", axis=1).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Iris Features", fontsize=16)
plt.tight_layout()
plt.savefig('iris_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a pairplot
g = sns.pairplot(df, hue="species", height=2.5, aspect=1.2)
g.fig.subplots_adjust(top=0.95, right=0.85)  # Adjust the right margin to make room for the legend
g.fig.suptitle("Pairplot of Iris Features", fontsize=16)
g._legend.set_bbox_to_anchor((1.10, 0.5))  # Move the legend outside the plot
plt.tight_layout()
plt.savefig('iris_pairplot.png', dpi=300, bbox_inches='tight')
plt.close()

# Perform PCA
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
x = df.loc[:, features].values
y = df.loc[:, ["species"]].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=["PC1", "PC2"])
finalDf = pd.concat([principalDf, df[["species"]]], axis=1)

# Plot PCA results
plt.figure(figsize=(10, 8))
targets = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
colors = ["r", "g", "b"]
for target, color in zip(targets, colors):
    indicesToKeep = finalDf["species"] == target
    plt.scatter(finalDf.loc[indicesToKeep, "PC1"], 
                finalDf.loc[indicesToKeep, "PC2"], 
                c=color, 
                s=50)

plt.title("PCA of Iris Dataset", fontsize=16)
plt.xlabel(f"First Principal Component ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"Second Principal Component ({pca.explained_variance_ratio_[1]:.2%})")
plt.legend(targets)
plt.tight_layout()
plt.savefig('iris_pca.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate and display feature means by species
feature_means = df.groupby("species")[features].mean()
print("\nFeature means by species:")
print(feature_means)

# Save feature means to CSV
feature_means.to_csv("iris_feature_means.csv")

print("Visualization complete. Check the generated PNG files for results.")
