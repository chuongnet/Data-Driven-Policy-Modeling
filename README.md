# Data-Driven-Policy-Modeling
The Policy analysis and Modeling with the data (unstructured text) driven techniques, Deep Learning and Natural Language Processing (NLP)
## 1. Supervisors
- Prof. Dr. Maria A. Wimmer - wimmer@uni-koblenz.de
(Professorin für E-Government Institut für Wirtschafts- und Verwaltungsinformatik Universität Koblenz-Landau http://www.uni-koblenz.de/agvinf)
- Dr. Ulf Lotzmann - ulf@uni-koblenz.de
## 2. Description
The Data Driven Policy Modeling is a Research Lab for the Electronic Government. The project is focused on both semantic and lexical analysis of tracking important information of the unstructured text such as research papers, research topics analysis and research reports to extract keywords in the relevant sentences (e.g. actors, objects, and their expectations,actions, conditions, and interdependencies)
## 2.1. Policy Models
Modeling and simulation of political-societal issues, e.g. threats for environment or health (Wimmer & Ulf). The model is based on the data analysis to represent the problems and their relationships under CCD structural diagrams (Consistent Conceptual Description). Those diagrams are finally transformed into statements, which are inspected and analyzed by stakeholders.
## 2.2. Text Mining and Data Analysis
Data Mining is a main technique to process all information of the unstructured text. The data, e.g. research papers, and reports, which are read and splited into individual sentences. Then, they are reprocessed to remove punctuations, special characters, and numbers, which makes sentences clean for extracting the main points (keywords) from the data. Those words are dominant in the whole document.
## 3. Methodologies
The key of the data driven is a main information extraction for further analysis from whole documents, which aims to solve problems or stakeholder's questions. The information is words or word phrases, that are given topic ideas from the text. Furthermore, that can be used to classify relevant or irrelevant sentences from the text, and their relationship identification between topics. 
## 3.1. Main topics of whole corpus
In sentiment analysis, Main topics detection for a whole document is the first task when working with unstructured text to know key points that the document is mentioning. The main topics are the keywords or word phrases that are dominant in the whole document and they are allocated and identified by the Latent Dirichlet Allocation (LDA) algorithm computation. The LDA is a deep learning technique for Natural Language Processing (NLP). 
## 3.2. Relevant or Irrelevant classification
Topics scenario is the collection of sentences that are related to the topic based on the topic score (nearest neighbors). Which is classificated by KMean clustering technique.  Classification of the range of relevant sentences in a single cluster by observing the Expected Mean Difference and the Mean of Error, which is based on the vector space transformation of the sentence. 
## 3.3. Actors Classification 
The actor is a keyword or word phrase that has the strongest relation to the topic scenario. The word or word phrase has the highest similarity score to the topic and is dominant over the cluster. 
## 3.4. Word classes and relationship classification
In lexical sentence analysis, identifying the subject, verb and object in the sentence, and the other noun, or noun phrase is the key process in political analysis. The sentence’s verb or verb phrase is known as the relationship and the main point to classify an action behavior of the sentence.
## 4. Support tools
- Gensim, Word similarity
- Spacy, word lexical identification
- Sklearn, machine learning techniques
- Scipy, stats computation
- XML tree, xml file constructor
