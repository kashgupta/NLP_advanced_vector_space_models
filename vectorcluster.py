from gensim.models import KeyedVectors
import numpy as np
from sklearn.cluster import KMeans

# This maps from word  -> list of candidates
word2cands = {}

# This maps from word  -> number of clusters
word2num = {}

# Read the words file.
with open("data/dev_input.txt") as f:
    for line in f:
        word, numclus, cands = line.split(" :: ")
        cands = cands.split()
        word2num[word] = int(numclus)
        word2cands[word] = cands

# Load cooccurrence vectors (question 2)
#vec = KeyedVectors.load_word2vec_format("data/coocvec-500mostfreq-window-3.vec.filter")
# Load dense vectors (uncomment for question 3)
vec = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.filter")

def sub_cost(char1, char2):
    if char1 == char2:
        return 0
    else:
        return 2

def edit_distance(str1, str2):
    '''Computes the minimum edit distance between the two strings.

    Use a cost of 1 for all operations.

    See Section 2.4 in Jurafsky and Martin for algorithm details.
    Do NOT use recursion.

    Returns:
    An integer representing the string edit distance
    between str1 and str2
    '''
    n = len(str1)
    m = len(str2)
    D = [[0 for i in range(m+1)] for j in range(n+1)]
    D[0][0] = 0
    for i in range(1,n+1):
        D[i][0] = D[i-1][0] + 1
    for j in range(1,m+1):
        D[0][j] = D[0][j-1] + 1
    for i in range(1,n+1):
        for j in range(1,m+1):
            D[i][j] = min(D[i-1][j]+1, D[i-1][j-1]+sub_cost(str1[i-1],str2[j-1]), D[i][j-1]+1)
    return D[n][m]

output = open('dev_output_dense.txt', 'w')

for word in word2cands:
    cands = word2cands[word]
    numclusters = word2num[word]

    # TODO: get word vectors from vec
    # Cluster them with k-means
    # Write the clusters to file.
    words = []
    X = []
    for cand in cands:
        try:
            vector = vec.get_vector(cand)
        except KeyError:
            min_dist = 100000
            best_match = ''
            for v in vec.vocab:
                dist = edit_distance(cand, v)
                if dist < min_dist:
                    min_dist = dist
                    best_match = v
            vector = vec.get_vector(best_match)
        X.append(vector)
        words.append(cand)

    X = np.array(X)
    kmeans = KMeans(n_clusters=numclusters).fit(X)
    results = [[] for i in range(numclusters)]
    labels = kmeans.labels_
    for i in range(len(labels)):
        results[labels[i]].append(words[i])
    print(results)

    for i in range(numclusters):
        output.write(word + ' :: ' + str(i) + ' :: ' + ''.join(phrase+' ' for phrase in results[i]) + '\n')
        print(word + ' :: ' + str(i) + ' :: ' + ''.join(phrase+' ' for phrase in results[i]) + '\n')


output.close()
