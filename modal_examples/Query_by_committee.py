import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner, Committee

# Set our RNG seed for reproducibility.
RANDOM_STATE_SEED = 1
np.random.seed(RANDOM_STATE_SEED)

# loading the iris dataset
iris = load_iris()

# Apply pca for easy visualisations
pca = PCA(n_components=2).fit_transform(iris['data'])

# Isolate the data we'll need for plotting.
x_component, y_component = pca[:, 0], pca[:, 1]

# generate the pool
X_pool = deepcopy(iris['data'])
y_pool = deepcopy(iris['target'])

# ------------------------------------------------------
# Create the committee
# ------------------------------------------------------

# initializing Committee members
n_members = 3
learner_list = list()

for member_idx in range(n_members):
    # initial training data
    n_initial = 5
    train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
    X_train = X_pool[train_idx]
    y_train = y_pool[train_idx]

    # creating a reduced copy of the data with the known instances removed
    X_pool = np.delete(X_pool, train_idx, axis=0)
    y_pool = np.delete(y_pool, train_idx)

    # initializing learner
    learner = ActiveLearner(estimator=RandomForestClassifier(),X_training=X_train, y_training=y_train)
    learner_list.append(learner)

# assembling the committee
committee = Committee(learner_list=learner_list)

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(n_members*4, 4))
    for learner_idx, learner in enumerate(committee):
        plt.subplot(1, n_members, learner_idx + 1)
        plt.scatter(x=pca[:, 0], y=pca[:, 1], c=learner.predict(iris['data']), cmap='viridis', s=50)
        plt.title('Learner no. %d initial predictions' % (learner_idx + 1))
    #plt.show()

unqueried_score = committee.score(iris['data'], iris['target'])
# ------------------------------------------------------------------
# Introduce Active Learning
# -----------------------------------------------------------------

performance_history = [unqueried_score]

# query by committee
n_queries = 15
for idx in range(n_queries):
    query_idx, query_instance = committee.query(X_pool)
    committee.teach(
        X=X_pool[query_idx].reshape(1, -1),
        y=y_pool[query_idx].reshape(1, )
    )
    performance_history.append(committee.score(iris['data'], iris['target']))
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)

# visualizing the final predictions per learner
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(n_members*4, 4))
    for learner_idx, learner in enumerate(committee):
        plt.subplot(1, n_members, learner_idx + 1)
        plt.scatter(x=pca[:, 0], y=pca[:, 1], c=learner.predict(iris['data']), cmap='viridis', s=50)
        plt.title('Learner no. %d predictions after %d queries' % (learner_idx + 1, n_queries))
    #plt.show()

# Plot our performance over time.
fig, ax = plt.subplots(figsize=(5,4), dpi=130)
ax.plot(performance_history)
ax.scatter(range(len(performance_history)), performance_history, s=13)
ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_ylim(bottom=0, top=1)
ax.grid(True)
ax.set_title('Incremental classification accuracy')
ax.set_xlabel('Query iteration')
ax.set_ylabel('Classification Accuracy')
#plt.show()

# Isolate the data we'll need for plotting.
predictions = committee.predict(iris['data'])
print(predictions)
is_correct = (predictions == iris['target'])

# Plot our updated classification results once we've trained our learner.
fig, ax = plt.subplots(figsize=(5,4), dpi=130)
ax.scatter(x=x_component[is_correct],  y=y_component[is_correct],  c='g', marker='+', label='Correct',   alpha=8/10)
ax.scatter(x=x_component[~is_correct], y=y_component[~is_correct], c='r', marker='x', label='Incorrect', alpha=8/10)
ax.set_title('Classification accuracy after {n} queries: {final_acc:.3f}'.format(n=n_queries, final_acc=performance_history[-1]))
ax.legend(loc='lower right')
plt.show()