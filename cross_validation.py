import pandas as pd
import math
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def split_train_test(data_path, test_ratio=0.3):
    # Load the dataset
    data = pd.read_excel(data_path)

    # Separate by class
    # zero_data = data[data['diagnosis'] == 0]
    # one_data = data[data['diagnosis'] == 1]
    zero_data = data[data['Outcome'] == 0]
    one_data = data[data['Outcome'] == 1]

    # Calculate split sizes
    test_zero_count = int(len(zero_data) * test_ratio)
    test_one_count = int(len(one_data) * test_ratio)

    # Randomly sample test data
    test_zero = zero_data.sample(test_zero_count, random_state=None)
    test_one = one_data.sample(test_one_count, random_state=None)

    # Combine test data
    test_data = pd.concat([test_zero, test_one])

    # Remaining data is train data
    train_data = data.drop(test_data.index)
    print(f'test:{test_data}')
    print(f'train:{train_data}')

    return train_data, test_data


def perform_gmm(train_data, k=3):
    # Drop the target column for clustering
    # features = train_data.drop(columns=['diagnosis'])
    features = train_data.drop(columns=['Outcome'])

    # Perform GMM clustering
    gmm = GaussianMixture(n_components=k, random_state=None)
    train_data['Cluster'] = gmm.fit_predict(features)

    # Analyze clusters
    clusters = {}
    for cluster in range(k):
        cluster_data = train_data[train_data['Cluster'] == cluster]
        # zero_count = len(cluster_data[cluster_data['diagnosis'] == 0])
        # one_count = len(cluster_data[cluster_data['diagnosis'] == 1])
        zero_count = len(cluster_data[cluster_data['Outcome'] == 0])
        one_count = len(cluster_data[cluster_data['Outcome'] == 1])
        clusters[cluster] = {'total': len(cluster_data), 'zero': zero_count, 'one': one_count}

    return train_data, clusters


# def classify_test_data(test_data, clusters, train_data, threshold=0.4):
#     # Separate class clusters
#     class_0_cluster = max(clusters, key=lambda x: clusters[x]['zero'])
#     class_1_cluster = max(clusters, key=lambda x: clusters[x]['one'])
#     print(f'class_0_cluster:{class_0_cluster}')
#     print(f'class_1_cluster:{class_1_cluster}')
#
#     # Classify test data based on minimum distance from all points in clusters
#     # features = train_data.drop(columns=['diagnosis', 'Cluster'])
#     # test_features = test_data.drop(columns=['diagnosis'])
#     features = train_data.drop(columns=['Outcome', 'Cluster'])
#     test_features = test_data.drop(columns=['Outcome'])
#     labels = []
#     rejected_data = []
#
#     for _, test_row in test_features.iterrows():
#         test_point = test_row.values
#
#         # Calculate distances from all points in class 0 cluster
#         cluster_0_points = features[train_data['Cluster'] == class_0_cluster].values
#         distances_to_0 = [math.sqrt(sum((test_point - cluster_point) ** 2)) for cluster_point in cluster_0_points]
#         min_dist_to_0 = min(distances_to_0)
#
#         # Calculate distances from all points in class 1 cluster
#         cluster_1_points = features[train_data['Cluster'] == class_1_cluster].values
#         distances_to_1 = [math.sqrt(sum((test_point - cluster_point) ** 2)) for cluster_point in cluster_1_points]
#         min_dist_to_1 = min(distances_to_1)
#
#         if min(min_dist_to_0, min_dist_to_1) < threshold:
#             labels.append(0 if min_dist_to_0 < min_dist_to_1 else 1)
#         else:
#             labels.append(None)  # Rejected
#             rejected_data.append(test_point)
#
#     test_data['label'] = labels
#     rejected_data = pd.DataFrame(rejected_data, columns=test_features.columns)
#     return test_data, rejected_data
def classify_test_data(test_data, clusters, train_data, threshold=0.4):
    # Separate class clusters
    class_0_cluster = max(clusters, key=lambda x: clusters[x]['zero'])
    class_1_cluster = max(clusters, key=lambda x: clusters[x]['one'])
    print(f'class_0_cluster:{class_0_cluster}')
    print(f'class_1_cluster:{class_1_cluster}')
    # Get cluster centroids
    # features = train_data.drop(columns=['diagnosis', 'Cluster'])
    # test_features = test_data.drop(columns=['diagnosis'])
    features = train_data.drop(columns=['Outcome', 'Cluster'])
    centroid_0 = features[train_data['Cluster'] == class_0_cluster].mean().values
    centroid_1 = features[train_data['Cluster'] == class_1_cluster].mean().values
    # Classify test data
    test_features = test_data.drop(columns=['Outcome'])
    labels = []
    rejected_data = []

    for _, row in test_features.iterrows():
        dist_to_0 = math.sqrt(sum((row.values - centroid_0) ** 2))
        dist_to_1 = math.sqrt(sum((row.values - centroid_1) ** 2))

        if min(dist_to_0, dist_to_1) < threshold:
            labels.append(0 if dist_to_0 < dist_to_1 else 1)
        else:
            labels.append(None)  # Rejected
            rejected_data.append(row.values)

    test_data['label'] = labels
    rejected_data = pd.DataFrame(rejected_data, columns=test_features.columns)
    return test_data, rejected_data


def classify_rejected_with_rf(rejected_data, train_data):
    # Prepare train data
    # train_features = train_data.drop(columns=['diagnosis', 'Cluster'])
    # train_labels = train_data['diagnosis']
    train_features = train_data.drop(columns=['Outcome', 'Cluster'])
    train_labels = train_data['Outcome']

    # Prepare rejected data
    rejected_features = rejected_data

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=None)
    rf.fit(train_features, train_labels)

    # Predict on rejected data
    rejected_data['label'] = rf.predict(rejected_features)

    return rejected_data


def evaluate_results(test_data):
    # Combine labels and evaluate
    # true_labels = test_data['diagnosis']
    true_labels = test_data['Outcome']
    predicted_labels = test_data['label']

    # Handle potential NaN values in predictions
    if predicted_labels.isnull().any():
        print("Warning: Some test samples were not classified (NaN values in predictions).")
        test_data = test_data.dropna(subset=['label'])
        # true_labels = test_data['diagnosis']
        true_labels = test_data['Outcome']
        predicted_labels = test_data['label']
    print(f'true_labels:{true_labels}')
    print(f'predicted_labels{predicted_labels}')
    acc = accuracy_score(true_labels, predicted_labels)
    prec = precision_score(true_labels, predicted_labels)
    rec = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot()
    plt.show()


def cross_validate(data_path, n_splits=5):
    # Load dataset
    data = pd.read_excel(data_path)

    # Split features and labels
    # X = data.drop(columns=['diagnosis'])
    # y = data['diagnosis']
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)

    overall_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Processing fold {fold + 1}/{n_splits}...")

        train_data = data.iloc[train_idx].copy()
        test_data = data.iloc[test_idx].copy()

        # Perform GMM on train data
        train_data, clusters = perform_gmm(train_data)

        # Classify test data using GMM
        test_data, rejected_data = classify_test_data(test_data, clusters, train_data)

        # Classify rejected data with Random Forest if needed
        if not rejected_data.empty:
            rejected_data = classify_rejected_with_rf(rejected_data, train_data)
            test_data.update(rejected_data)

        # Evaluate results
        # true_labels = test_data['diagnosis']
        true_labels = test_data['Outcome']
        predicted_labels = test_data['label']

        # Handle NaN values in predictions
        if predicted_labels.isnull().any():
            print(f"Warning: Fold {fold + 1} contains unclassified samples (NaN values).")
            test_data = test_data.dropna(subset=['label'])
            # true_labels = test_data['diagnosis']
            true_labels = test_data['Outcome']
            predicted_labels = test_data['label']

        acc = accuracy_score(true_labels, predicted_labels)
        prec = precision_score(true_labels, predicted_labels)
        rec = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        print(f'true_labels{true_labels}')
        print(f'predicted_labels{predicted_labels}')
        overall_metrics['accuracy'].append(acc)
        overall_metrics['precision'].append(prec)
        overall_metrics['recall'].append(rec)
        overall_metrics['f1'].append(f1)

    print("\nCross-Validation Results:")
    print("Average Accuracy:", sum(overall_metrics['accuracy']) / n_splits)
    print("Average Precision:", sum(overall_metrics['precision']) / n_splits)
    print("Average Recall:", sum(overall_metrics['recall']) / n_splits)
    print("Average F1 Score:", sum(overall_metrics['f1']) / n_splits)


# Workflow
# data_path = "breast_cancer/breast_cancer_normalized.xlsx"
data_path = "diabetes_700/diabetes_normal.xlsx"
cross_validate(data_path)
