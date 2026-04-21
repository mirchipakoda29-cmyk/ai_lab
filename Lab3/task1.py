import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Load Dataset
# ---------------------------
data = pd.read_csv("emails.csv")

# Features and Labels
X = data.drop(columns=['Email No.', 'Prediction']).values
y = data['Prediction'].values

# ---------------------------
# Train Test Split (80/20)
# ---------------------------
indices = np.arange(len(X))
np.random.shuffle(indices)

split = int(0.8 * len(indices))

train_idx = indices[:split]
test_idx = indices[split:]

X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]


# ---------------------------
# Select Top Vocabulary Words
# ---------------------------
def select_vocab(X, vocab_size):
    word_totals = np.sum(X, axis=0)
    top_indices = np.argsort(word_totals)[::-1][:vocab_size]
    return top_indices


# ---------------------------
# Naive Bayes Classifier
# ---------------------------
class NaiveBayes:

    def __init__(self, laplace=True):
        self.laplace = laplace

    def fit(self, X, y):

        self.classes = np.unique(y)
        self.vocab_size = X.shape[1]

        self.class_priors = {}
        self.word_counts = {c: np.zeros(self.vocab_size) for c in self.classes}
        self.total_words = {c: 0 for c in self.classes}

        # Count words per class
        for x, label in zip(X, y):
            self.word_counts[label] += x
            self.total_words[label] += np.sum(x)

        # Prior probabilities
        total_docs = len(y)
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / total_docs

    def predict(self, X):

        predictions = []

        for x in X:

            class_scores = {}

            for c in self.classes:

                score = np.log(self.class_priors[c])

                for i in range(len(x)):

                    freq = x[i]

                    if self.laplace:
                        likelihood = (self.word_counts[c][i] + 1) / \
                                     (self.total_words[c] + self.vocab_size)
                    else:
                        if self.total_words[c] == 0:
                            continue

                        likelihood = self.word_counts[c][i] / self.total_words[c]

                        if likelihood == 0:
                            continue

                    score += freq * np.log(likelihood)

                class_scores[c] = score

            predictions.append(max(class_scores, key=class_scores.get))

        return np.array(predictions)


# ---------------------------
# Confusion Matrix
# ---------------------------
def confusion_matrix(y_true, y_pred):

    classes = np.unique(y_true)
    matrix = pd.DataFrame(0, index=classes, columns=classes)

    for t, p in zip(y_true, y_pred):
        matrix.loc[t, p] += 1

    return matrix


# ---------------------------
# Evaluation Metrics
# ---------------------------
def metrics(cm):

    TP = cm.iloc[1, 1]
    TN = cm.iloc[0, 0]
    FP = cm.iloc[0, 1]
    FN = cm.iloc[1, 0]

    accuracy = (TP + TN) / np.sum(cm.values)
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return accuracy, precision, recall, f1


# ---------------------------
# Experiment Section
# ---------------------------
vocab_sizes = [500, 1000, 2000]

train_accs = []
test_accs = []

for size in vocab_sizes:

    vocab_indices = select_vocab(X_train, size)

    X_train_sub = X_train[:, vocab_indices]
    X_test_sub = X_test[:, vocab_indices]

    model = NaiveBayes(laplace=True)
    model.fit(X_train_sub, y_train)

    # Train Accuracy
    train_pred = model.predict(X_train_sub)
    train_cm = confusion_matrix(y_train, train_pred)
    train_acc, _, _, _ = metrics(train_cm)

    # Test Accuracy
    test_pred = model.predict(X_test_sub)
    test_cm = confusion_matrix(y_test, test_pred)
    test_acc, precision, recall, f1 = metrics(test_cm)

    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print("\nVocabulary Size:", size)
    print("Confusion Matrix:\n", test_cm)
    print("Accuracy:", test_acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


# ---------------------------
# Plot Accuracy Graph
# ---------------------------
plt.plot(vocab_sizes, train_accs, label="Training Accuracy")
plt.plot(vocab_sizes, test_accs, label="Testing Accuracy")

plt.xlabel("Vocabulary Size")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
