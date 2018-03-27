import pandas as pd
from WordsTokenizer import WordsTokenizer
from SemanticsClassifier import SemanticsClassifier


#df_sem = pd.read_csv('../data/keyword_names.csv')
#df_search = pd.read_csv('../data/keyword_search.csv')
#df_sem = pd.read_csv('../data/test_sem.csv')
#df_search = pd.read_csv('../data/test_search.csv')
df_sem = pd.read_csv('../data/yandex_keyword.csv')
df_search = pd.read_csv('../data/yandex_data_utf.csv', low_memory=False)

# ТОКЕНИЗАЦИЯ ЗАПРОСОВ ПО УНИКАЛЬНЫМ СЛОВАМ
tokenizer = WordsTokenizer()
tokenizer.fit(df_sem['keyword_name'].values)

cos_matrix = tokenizer.transform(df_search['search'].values)
sem_cos_matrix = tokenizer.transform(df_sem['keyword_name'])

classifier = SemanticsClassifier(tokenizer=tokenizer, be_verbose=False, a=0.7)
classifier.train(df_sem['keyword_name'].values)
predictions = classifier.predict(df_search['search'].values)
df_search['predictions'] = predictions
print("--------------------------")
print("Predictions are ready, writing to file...")
print("--------------------------")
df_search.to_csv('../results/result.csv')
