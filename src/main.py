import pandas as pd
from WordsTokenizer import WordsTokenizer
from SemanticsClassifier import SemanticsClassifier


df_sem = pd.read_csv('../data/keyword_names.csv')
df_search = pd.read_csv('../data/keyword_search.csv')
#df_sem = pd.read_csv('../data/test_sem.csv')
#df_search = pd.read_csv('../data/test_search.csv')

# ТОКЕНИЗАЦИЯ ЗАПРОСОВ ПО УНИКАЛЬНЫМ СЛОВАМ
tokenizer = WordsTokenizer()
tokenizer.fit(df_sem['keyword_name'].values)

cos_matrix = tokenizer.transform(df_search['search'].values)
sem_cos_matrix = tokenizer.transform(df_sem['keyword_name'])

classifier = SemanticsClassifier(tokenizer=tokenizer, be_verbose=True)
classifier.train(df_sem['keyword_name'].values)
predictions = classifier.predict(df_search['search'].values)

df_search['predictions'] = predictions
df_search.to_csv('../results/result.csv')
