from gensim.models import KeyedVectors
vecfile = 'GoogleNews-vectors-negative300.bin'
vecs = KeyedVectors.load_word2vec_format(vecfile, binary=True)


#dimensionality
vecs.vector_size

#most similar to picnic
vecs.most_similar(positive = ['picnic'], topn=7)

vecs.doesnt_match(['tissue', 'papyrus', 'manila', 'newsprint', 'parchment', 'gazette'])

vecs.most_similar(positive = ['leg', 'throw'], negative = ['jump'], topn=1)