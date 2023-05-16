# FIXME: Original
def custom_weight_calculation0(y):
    weights = np.where(True, 1, y)
    return weights


# Class imbalance: If you have imbalanced classes, you can assign higher weights to the minority class. For example, you can use the inverse of class frequencies as weights. The less frequent class will have a higher weight, giving it more importance during training.
# FIXME: Not ideal for this case
def custom_weight_calculation1(y):
    class_weights = len(y) / (len(np.unique(y)) * np.bincount(y))
    weights = class_weights[y]
    return weights


# Misclassification penalties: You can assign different weights based on the cost of misclassifying each class. If certain classes are more important to classify correctly, you can assign them higher weights.
# FIXME: under testing
def custom_weight_calculation2(y):
    weights = np.ones_like(y)
    weights[y == 0] = 1.0  # Class 0 weight
    weights[y == 1] = 2.0  # Class 1 weight
    return weights


# Sample difficulty: You can analyze the features of your samples and assign weights based on their difficulty level. For example, if some samples are harder to classify correctly based on certain features, you can assign them higher weights to provide more emphasis on those challenging cases.
# FIXME: under testing
def custom_weight_calculation3(y):
    # Calculate sample difficulty based on features
    # Modify this logic according to your specific problem
    difficulty = calculate_difficulty(X, y)

    # Assign weights based on difficulty
    weights = 1.0 / difficulty
    return weights


# Expert knowledge: If you have domain expertise or prior knowledge about the data, you can incorporate that into the weight calculation function. For example, if certain samples are known to be more reliable or important, you can assign them higher weights.
# FIXME: under testing
def custom_weight_calculation4(y):
    weights = np.ones_like(y)
    weights[important_samples_indices] = 2.0  # Assign higher weight to important samples
    return weights


# Create the RandomForestClassifier with custom weight calculation
class RF_local(RF):
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = custom_weight_calculation0(y)
        super().fit(X, y, sample_weight=sample_weight)
