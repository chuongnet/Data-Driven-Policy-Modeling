from sklearn.manifold import TSNE
from sklearn.decomposition import SparsePCA, TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

from gensim.corpora import Dictionary
from gensim.models import LdaModel, Word2Vec, LdaMulticore, TfidfModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from scipy import stats

from string import punctuation
from collections import defaultdict
from nltk.corpus import stopwords
from os import listdir

import unicodedata
import pickle
import spacy
import numpy as np
import matplotlib.pyplot as plt
import re


#### Preprocessing Document, basic functions for other functions use ####

### Prepare text functions
def load_file(fn, is_dump=False):
    if not is_dump:
        file = open(fn, 'r')
        doc = file.read()
        file.close()
    else:
        file = open(fn, 'rb')
        doc = pickle.load(file)
        file.close()
    return doc


def save_file(fn, data, is_dump=False):    
    if not is_dump:
        file = open(fn, 'w')
        file.write(data)
        file.close()
    else:
        file = open(fn, 'wb')
        pickle.dump(data, file)
        file.close()
        
        
def load_spacy(module='en_core_web_sm'):
    nlp = spacy.load(module)
    return nlp


def remove_stopwords(tokens):
    tokens = [w for w in tokens if w not in set(stopwords.words('english'))]
    return tokens


def remove_numerics(tokens):
    tokens = [w for w in tokens if not w.isnumeric()]
    return tokens


def remove_non_ascii(tokenized_sent):
    """ remove non-ascii characters in sentences list"""
    word = [unicodedata.normalize('NFKD', w)
                .encode('ascii', 'ignore')
                .decode('utf-8', 'ignore') for w in tokenized_sent]
    return word


## clean tokens with spacy 
def clean_text(line, nlp, remove_numeric=True, lemmanizer=True):
    """preprocess text with useing spacy."""
    re_punc = re.compile('[%s]' % re.escape(punctuation))
    remove_words = 'eg ie however'.split()
    
    line = nlp(line.lower())
    ## extract token from nlp
    tokens = [token for token in line if not token.is_stop]
    ## word lemmanizer
    if lemmanizer:
        tokens = [token.lemma_ for token in tokens]
    else:
        tokens = [token.text for token in tokens]
    ## remove punctuation
    tokens = [re_punc.sub('', token) for token in tokens]
    ## remove single character
    tokens = [token for token in tokens if len(token) > 1]
    tokens = [token for token in tokens if token not in remove_words]
    tokens = remove_non_ascii(tokens)
    if remove_numeric:
        tokens = [token for token in tokens if not token.isnumeric()]
    
    return tokens


def save_traces(documents, filename='data/tracing.txt'):
    """ Save the tracing list into a tracing.txt file
    load documents list in the data folder.
    list od documents' info: doc_name, increase_num, start, end
    :param documets: the corpus of sentences
    :return: None, saving file into disk with a name tracing.txt
    """
    docs = listdir('raw_docs')
    # documents = docs_list
    infos = [(val, i) for i, val in enumerate(documents) if val in docs]
    tracing = list()
    for i in range(len(infos)):
        subs = list()
        subs.append(infos[i][0])  # title
        subs.append(str(i + 1))  # position
        subs.append(str(infos[i][1] + 1))  # start
        if i == len(infos) - 1:  # end
            subs.append(str(len(documents)))  # trace back never over this point
        else:
            # get the next doc info position
            subs.append(str(infos[i + 1][1]))  # next is the previous doc end
            
        tracing.append(subs)

    tracing = [' '.join(row) for row in tracing]
    tracing = '\n'.join(tracing)
    save_file(filename, tracing)
    
    
## trace back to the original sentences in a document
def tracing_back(line_id, lines):
    """@return: (title, index, sentence)"""
    traces = load_file('data/tracing.txt').split('\n')
    traces = [line.split() for line in traces]
    #print(traces)
    for trace in traces:
        ## search where sentence in the doc        
        if (line_id + int(trace[1])) - int(trace[-1]) < 0: 
            ix = line_id + int(trace[1])   # array index starts from 0
            doc = (ix, lines[ix], trace[0])
            break
        else:
            doc = None
    
    return doc
    

## TSNE transform vector
def tsne_transform(vectors, n_comp=2, loading=True, filename='pickled_data/tsne_vectors.pkl'):
    if loading:
        vectors = load_file(filename, is_dump=True)
    else:
        tsne = TSNE(n_components=n_comp, init='pca')
        vectors = tsne.fit_transform(vectors)
        save_file(filename, vectors, is_dump=True)
    return vectors


def tsne_transform_only(vectors, n_comp=2):
    tsne = TSNE(n_components=n_comp, init='pca')
    vectors = tsne.fit_transform(vectors)
    return vectors


#### Document Model #####

## Embedding document model, doc2vec provide a vector space 
def doc2vec_model(tokens, filename='pickled_data/doc2vecmodel_project.h5', epochs=100, loading=True):
    if loading:
        emb_model = Doc2Vec.load(filename)
    else:
        max_length = max([len(line) for line in tokens]) + 1
        win_size = max_length//2
        tagdocs = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokens)]
        emb_model = Doc2Vec(tagdocs, 
                            window=win_size,
                            alpha=1e3,
                            min_count=1, 
                            vector_size=max_length,
                            dbow_words=1,
                            dm_concat=1,
                            workers=3,
                            dm=0, #DBOW
                            hs=1,                            
                            epochs=epochs)
        
        emb_model.save(filename)
    vectors = emb_model.dv.get_normed_vectors()
    return vectors


#### Word Model #####

## mapping the original sentences into clustered sentences' groups
def mapping_original_line(sorted_keyclusters, lines):
    """ Mapping original sentence into clustered groups
        @return: topic(key, cluster) list
    """
    #titles = load_file('data/tracing.txt').split('\n')
    #titles = titles[0]
    #lines = 
    cluster_original_lines = list()
    for topic in sorted_keyclusters:
        topics = list()
        for key, cluster in topic:
            ## map ix into lines
            cat = list()
            for ix in cluster:
                line = tracing_back(ix, lines)
                if line is None:
                    print("No sentence found.")
                    break # out of loop
                else:
                    cat.append(line)
                    
            topics.append((key, cat))
            
        cluster_original_lines.append(topics)
    
    return cluster_original_lines


## identify action statement, ner with spacy
def lexical_lines(nlp, original_keylines):
    """ spacy model, noun_chunks for a line, ner list
        @return: noun phrases in the sentence of a cluster
    """
    re_sub = re.compile('nsubj(\w+)?')
    re_aux = re.compile('aux(\w+)?')
    objs = 'dobj pobj pcomp compound acomp'.split()
    prep = 'prep agent'.split()
    extcl = 'ccomp xcomp advcl conj dative acl recl'.split() 
    dets = 'det prep punct'.split()
    objts = 'dobj pobj'.split()
    
    noun_phrases = list()
    
    for topic in original_keylines:
        topics = list()
        for key, cluster in topic:
            cat = list()
            for ix, line, title in cluster:
                doc = nlp(line)
                chunks = list()
                ## processing doc with nlp
                ## action stmt, subj, and verb 
                action = False
                subj = list()
                verb = ''
                predicate = list()
                i = 0 # token position
                for token in doc:
                    i += 1 # increase for every token
                    # analyst verb tense
                    if token.dep_ == 'ROOT': # identify the main verb
                        # collect nsubj and amod, amodpass
                        span = doc[0 : token.i]
                        for chunk in span.noun_chunks:
                            header = [chunk.root.head.text]
                            mul = [t for t in chunk if t.dep_ not in dets]
                            sb = True
                            for c in mul:
                                if c.dep_ in objts:
                                    muls = [t.text for t in mul]
                                    muls = header + muls
                                    subj.append((c.dep_, '_'.join(muls)))
                                    sb = False
                                elif re_sub.search(c.dep_):
                                    muls = [t.text for t in mul]
                                    subj.append((c.dep_, '_'.join(muls)))
                                    sb = False
                            if sb:
                                mul = [t.text for t in mul]
                                subj.append((c.dep_, '_'.join(mul)))
                        
                        # main verb in past tense
                        verb = token.text
                        if 'Fin' not in token.morph.get('VerbForm'):
                            if token.lemma_.lower() != 'be':
                                if token.lemma_.lower() != 'have':
                                    flg = False # verb form: participle, gerund, infinitive
                                    for lef in token.lefts:
                                        if re_aux.search(lef.dep_):
                                            flg = True
                                            break
                                if flg:
                                    # verb in action form
                                    #verb = token.text
                                    action = True
                                else:
                                    # Past tense
                                    for rig in token.rights:
                                        if rig.dep_ in extcl: # if extra clause direct to verb
                                            for c in rig.children:
                                                if re_sub.search(c.dep_):
                                                    action = True
                                                elif c.dep_ in objs or c.dep_ in prep:
                                                    action = True
                                        elif token.lemma_.lower() != 'have':
                                            for rig in token.children:
                                                if rig.dep_ in objs or rig.dep_ in prep:
                                                    action = True
                        ## root, else 
                        elif token.lemma_.lower() != 'have':
                            ## verb in finitive and contains an action form
                            # when sent is infinitive
                            for lef in token.lefts:
                                if re_aux.search(lef.dep_):
                                    # full action stmt
                                    if token.lemma_.lower() != 'be':
                                        action = True
                                        verb = token.text
                                else:
                                    # when sent is a present stmt
                                    if token.lemma_.lower() != 'be':
                                        for rig in token.rights:
                                            if rig.dep_ in extcl: # has extra clause
                                                for c in rig.children:
                                                    if re_sub.search(c.dep_):
                                                        action = True
                                                        verb = token.text                                            
                                                    elif c.dep_ in objs or c.dep_ in prep:
                                                        action = True
                                                        verb = token.text
                                                    elif c.dep_ in extcl:
                                                        action = True
                                                        verb = token.text
                                            elif rig.dep_ in objs or rig.dep_ in prep:
                                                # when direct object and sub-clause
                                                action = True
                                                verb = token.text
                                    
                        # root, else
                        elif token.lemma_.lower() == 'have': # if verb is 'to have'
                            verb = token.text
                            for rig in token.rights:
                                # check on the left of child if has an aux
                                haveto = False
                                for lef in rig.lefts:
                                    if re_aux.search(lef.dep_):
                                        if lef.text == 'to':
                                            haveto = True
                                if haveto:
                                    if rig.dep_ in extcl:
                                        action = True
                        
                        ## analyst predicate phrase, begin with ROOT
                        length = len(list(doc))
                        span = doc[i+1 : length] # skip verb
                        for chunk in span.noun_chunks:
                            header = [chunk.root.head.text]
                            mul = [t for t in chunk if t.dep_ not in dets]
                            sb = True
                            for w in mul:
                                if w.dep_ in objts:
                                    muls = [t.text for t in mul]
                                    muls = header + muls
                                    predicate.append((w.dep_, '_'.join(muls)))
                                    sb = False
                                elif re_sub.search(w.dep_):
                                    muls = [t.text for t in mul]# + header
                                    predicate.append((w.dep_, '_'.join(muls)))
                                    sb = False
                            if sb:
                                mul = [t.text for t in mul]
                                predicate.append((w.dep_, '_'.join(mul)))
                                
                        ## more, but close for testing 
                        break # close of token loop.
                                    
                stmt = [action, subj, verb, predicate]
                chunks.append(stmt)
                ## detect ner
                chunks.append([[ent.text, ent.label_] for ent in doc.ents])
                                
                ## add a line back to cluster
                cat.append((ix, chunks))
            topics.append((key, cat))
            
        noun_phrases.append(topics)
        
    return noun_phrases


## collect all entities and arrange into categories
def line_entity_categories(syntactic_lines):
    """ @return: category, list of (line_ix, entity)"""
    cat_entities = defaultdict(list)
    for topic in syntactic_lines:
        for key, cluster in topic:
            for ix, line in cluster:
                if len(line[1]) > 0:
                    ents = line[1]
                    for val, cat in ents:
                        v = [(ix, val)]
                        cat_entities[cat] += v
    return cat_entities        


## embedding multiword ngrams
## auto detect common phrases, supporting word2vec model
def search_probability_phrases(tokened_lines):
    """ Searching word-phrases base on probability of word-neighbors """
    phrases = Phrases(tokened_lines,
                          min_count=1,
                          threshold=0.3,
                          scoring='npmi',
                          connector_words=ENGLISH_CONNECTOR_WORDS)
    
    line_phrases = [phrases[line] for line in tokened_lines]
    
    return line_phrases


## multi-word ngrams model
## ngrams model find possible pairs of words in meaning
def multiword_model(token_phrases, filename='pickled_data/multiword_model_project.h5', epochs=100, loading=True):
    
    if loading:
        model = Word2Vec.load(filename)
    else:
        max_size = max([len(line) for line in token_phrases]) + 1
        winsize = max_size//2
        
        model = Word2Vec(sentences=token_phrases,
                         min_count=1,
                         window=winsize,
                         vector_size=max_size,
                         alpha=1e3,
                         sg=1, #Skip-gram
                         hs=1,
                         cbow_mean=0,
                         workers=3,                         
                         epochs=epochs)
        model.save(filename)
    return model


#### Topic (Scenario) Model ####

## train lda topic model, 
def ldatrain(tokens, epochs=100, topics=5, filename='pickled_data/ldamodel_project.h5', loading=True):
    """@return: list topics' score for sentences.
        and topics list with keywords
    """
    dct = Dictionary(tokens)
    bow = [dct.doc2bow(doc) for doc in tokens]
    max_size = len(tokens)
    
    if loading:
        ldamodel = LdaMulticore.load(filename)
    else:
        ldamodel = LdaMulticore(corpus=bow, 
                                num_topics=topics, 
                                id2word=dct, 
                                chunksize=max_size,
                                iterations=max_size,
                                eta='auto',
                                eval_every=1,
                                passes=epochs,
                                workers=3)
        ldamodel.save(filename)

    topics = ldamodel.top_topics(bow, topn=10)
    ldacorpus = ldamodel[bow] # sentence and topics
    ldacorpus = [sorted(doc, key=lambda x: -x[1]) for doc in ldacorpus]
    
    return ldacorpus, topics


#### Supported Model Functions ####
#### sorted main features, based tfidfModel, visual-clusters

def doc_main_feature(tokenized):
    """get word which has high tfidf score of the sentences in documents"""
    dic = Dictionary(tokenized)
    corpus = [dic.doc2bow(line) for line in tokenized]
    tf_model = TfidfModel(corpus)

    tfidf_corpus = list()
    scores = [[v[1] for v in doc] for doc in tf_model.__getitem__(corpus)]

    for i in range(len(scores)):
        tfidf_corpus.append(list(zip(*(tokenized[i], scores[i]))))

    # sort tfidf descending
    tfidf_corpus = [sorted(line, key=lambda x: -x[1]) for line in tfidf_corpus]

    #tfidf_corpus
    docs_features = []
    for i in range(len(tfidf_corpus)):
        if len(tfidf_corpus[i]) == 0:
            docs_features.append('')
        else:
            docs_features.append(tfidf_corpus[i][0][0])
    
    return docs_features, tfidf_corpus


## distance its own cluster to its centroid
def index_min_distance(indexes, vectors, centroid): 
    """index of minimum distance between centroid and its cluster points"""
    centroid = centroid.reshape(1, -1)
    distances = pairwise_distances(vectors, centroid, metric='euclidean')
    ## calculate expected point.
    expected = np.sum(distances)/len(vectors)
    pairs = sorted(zip(indexes, distances), key=lambda x: x[1])
    ix = 0
    for i in range(len(pairs)):
        if pairs[i][1] - expected < 0:
            continue
        else:
            ix = i - 1
            break
    pairs = pairs[:ix]
    index = [pair[0] for pair in pairs] # the list of reducing index
    
    return index


def tfidf_vectors(tokens):
    """ @return: vector space from tf-idf model"""
    tf = TfidfVectorizer()
    vectors = tf.fit_transform(tokens)
    
    return vectors
    
    
def tf_scores(words):
    """fit words into term frequency"""
    tf = TfidfVectorizer()
    X = tf.fit_transform(words)
    features = tf.get_feature_names()
    svd = TruncatedSVD(n_components=1) #
    scores = svd.fit(X)
    comp = scores.components_.tolist()[0]  # converts ndarray to list 
    sorted_w = sorted(zip(features, comp), key=lambda x: -x[1]) # sort on comp with descending
    return sorted_w


# print out full information clusters, nearest centroid, most-frequent word
def print_doc_clusters(model, documents, key_features, vectorized):
    """represents clusters infor. """
    key_features = np.asarray(key_features)
    documents = np.asarray(documents)
    labels = model.labels_
    centroids = model.cluster_centers_
    clusters = list() # list of index of each cluster
    
    for k in range(len(centroids)):
        # stores index into list
        cluster = list()
        
        index = np.where(labels == k)
        vectors = vectorized[index]
        # based on sorted indexes, get documents, key-features 
        sorted_ix = index_min_distance(index, vectors, centroids[k])
        
        index = sorted_ix[0]
        docs = documents[index]
        nearest = docs[0] # doc is closed to centroid
        keys = key_features[index]
        nearest_key = keys[0]
                
        p_words = tf_scores(docs)
        pop_w = p_words[:5]        
        
        # save all cluster infor.
        cluster.append(pop_w)
        cluster.append((index[0], nearest_key))
        cluster.append(index)
        
        clusters.append(cluster)
        
        print('\n================================================>')
        print(f'Cluster {k}: \n \t *topics: {pop_w}')
        print(f'\t *Key centroid: {nearest}  <<{nearest_key}>>')
        print('>>> List documents:')
        for i in range(len(docs)):
            print(docs[i], f'<<{keys[i]}>>')
        #print('\n')
        
    return clusters


### Visualization models
def visualization_kmeans(vectorizer, model, n_clusters=2, flatten=True):
    #kmeans = KMeans(n_clusters=n_components)
    yhat = model.fit_predict(vectorizer)
    clusters = np.unique(yhat)
    if flatten:
        #ax = plt.figure()
        centroids = model.cluster_centers_
            
        if vectorizer.shape[1] > 2:
            #tsne = TSNE(perplexity=20, n_components=2, init='pca')
            tsne = PCA(n_components=2)
            vectorizer = tsne.fit_transform(vectorizer)
            centroids = tsne.fit_transform(centroids)
            
        for centroid in centroids:
            plt.plot(centroid[0], centroid[1], color='black', marker='X')
        
        for cluster in clusters:
            row_ix = np.where(yhat == cluster)
            plt.scatter(vectorizer[row_ix, 0], vectorizer[row_ix, 1])
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
    else:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(projection='3d')
        centroids = model.cluster_centers_

        if vectorizer.shape[1] > 3:
            #tsne = TSNE(perplexity=20, n_components=3, init='pca')
            tsne = PCA(n_components=3)
            vectorizer = tsne.fit_transform(vectorizer)
            centroids = tsne.fit_transform(centroids)
           
        for centroid in centroids:
            ax.plot(centroid[0], centroid[1], centroid[2], color='black', marker='X')

        for cluster in clusters:
            row_ix = np.where(yhat == cluster)
            ax.scatter(vectorizer[row_ix, 0], vectorizer[row_ix, 1], vectorizer[row_ix, 2])
        
        ax.set_xlabel('pca1')
        ax.set_ylabel('pca2')
        ax.set_zlabel('pca3')
    plt.title('Documents Clustering, k= %s' % n_clusters)
    plt.show()
    return yhat


def visualization_kmeans2(vectorizer, yhat, centroids, n_clusters=2, flatten=True):
    #kmeans = KMeans(n_clusters=n_components)
    #yhat = model.fit_predict(vectorizer)
    clusters = np.unique(yhat)
    if flatten:
        #ax = plt.figure()
        #centroids = model.cluster_centers_
            
        if vectorizer.shape[1] > 2:
            #tsne = TSNE(perplexity=20, n_components=2, init='pca')
            tsne = PCA(n_components=2)
            vectorizer = tsne.fit_transform(vectorizer)
            centroids = tsne.fit_transform(centroids)
            
        for centroid in centroids:
            plt.plot(centroid[0], centroid[1], color='black', marker='x')
        
        for cluster in clusters:
            row_ix = np.where(yhat == cluster)
            plt.scatter(vectorizer[row_ix, 0], vectorizer[row_ix, 1])
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
    else:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(projection='3d')
        centroids = model.cluster_centers_

        if vectorizer.shape[1] > 3:
            #tsne = TSNE(perplexity=20, n_components=3, init='pca')
            tsne = PCA(n_components=3)
            vectorizer = tsne.fit_transform(vectorizer)
            centroids = tsne.fit_transform(centroids)
           
        for centroid in centroids:
            ax.plot(centroid[0], centroid[1], centroid[2], color='black', marker='X')

        for cluster in clusters:
            row_ix = np.where(yhat == cluster)
            ax.scatter(vectorizer[row_ix, 0], vectorizer[row_ix, 1], vectorizer[row_ix, 2])
        
        ax.set_xlabel('pca1')
        ax.set_ylabel('pca2')
        ax.set_zlabel('pca3')
    plt.title('Scenario Sentences Clustering, k= %s' % n_clusters)
    plt.show()
    return yhat


def clustering_topics(km_model, vectors_topics, k, loading=True, filename='pickled_data/kmeans_models_topic_'):
    yhats = list()
    topics_centroids = list() # list centroids shape(10, 67) for each topic
    
    for i, vec in enumerate(vectors_topics):
        if loading:
            fn = filename + str(i)
            model = load_file(fn, is_dump=True)
            yhat = model.predict(vec)
            yhats.append(yhat)
            centroids = model.cluster_centers_
            topics_centroids.append(centroids)
            vec = np.asarray(vec)
            visualization_kmeans2(vec, yhat, centroids, k)
        else:            
            yhat = km_model.fit_predict(vec)
            ## save model
            fn = filename + str(i)
            save_file(fn, km_model, is_dump=True)
            
            yhats.append(yhat)
            centroids = km_model.cluster_centers_
            topics_centroids.append(centroids)
            vec = np.asarray(vec)
            visualization_kmeans2(vec, yhat, centroids, k)
    
    return yhats, topics_centroids


## plot the data
def plotting( x_data, y_data, kind='plot', title='Sentences', xlabel='X', ylabel='Y'):
    if kind is 'plot':
        plt.plot(x_data, y_data)
    elif kind is 'bar':
        plt.bar(x_data, y_data)
    elif kind is 'scatter':
        plt.scatter(x_data, y_data)
    else:
        print('plotting type does not set')
        
    plt.xticks(x_data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    

## create centroids (k)
def initial_centroids(n, step=0):
    """max k centroids generation 
    based on binomial dist. max scores, which p depends on the N of samples
    """
    if n < 160:
        p = 0.1
    elif n < 800:
        p = 0.07
    elif n > 800:
        p = 0.03
    
    if step == 0:
        scores = np.asarray([stats.binom.pmf(k, n, p) for k in np.arange(1, n)])
        k = np.where(scores == max(scores))[0][0]
    else:
        k = len(np.arange(1, n, step))
    
    return k + 1


def optimal_centroids(vectors, kmax=20, ranges=None):
    """multiple runs model with each k"""
    silscores = list()
    wss = list()  # within-cluster sum of squared error
    if ranges is None:
        for k in range(2, kmax):
            model = KMeans(n_clusters=k, max_iter=1000).fit(vectors)
            #yhat = model.fit_predict(vectors)
            labels = model.labels_
            #print(labels)
            silscores.append(silhouette_score(vectors, labels, metric='euclidean'))
            wss.append(model.inertia_)
    else:
        for k in ranges:
            model = KMeans(n_clusters=k, max_iter=1000).fit(vectors)
            #yhat = model.fit_predict(vectors)
            labels = model.labels_
            #print(labels)
            silscores.append(silhouette_score(vectors, labels, metric='euclidean'))
            wss.append(model.inertia_)

    return silscores, wss

                          
## representation tools                   
def tsne_plot(X, components=2, title='word'):
    tsne = TSNE(perplexity=30, n_components=components, init='pca', random_state=42)
    result = tsne.fit_transform(X)
    plt.scatter(result[:, 0], result[:, 1], label=title)
    plt.legend()
    plt.title(title + ' distribution')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.show()
    return tsne
    
    
def sparse_plot(X, components=2, title='word'):
    pca = SparsePCA(n_components=components)
    result = pca.fit_transform(X)
    plt.scatter(result[:, 0], result[:, 1], label=title)
    plt.legend()
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title(title + ' distribution')
    plt.show()
    return pca


def svd_plot(X, components=2, title='word'):
    svd = TruncatedSVD(n_components=components)
    result = svd.fit_transform(X)
    plt.scatter(result[:, 0], result[:, 1], label=title)
    plt.legend()
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title(title + ' distribution')
    plt.show()
    return svd


def pca_plot(X, components=2, title='word'):
    pca = PCA(n_components=components)
    result = pca.fit_transform(X)
    plt.scatter(result[:, 0], result[:, 1], label=title)
    plt.legend()
    plt.title(title + ' distribution')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.show()
    return pca
