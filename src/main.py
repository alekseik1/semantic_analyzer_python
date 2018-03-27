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
predictions = []
for i in range(0, df_search.shape[0], 1000):
    # Если выпираем, то
    if 1000+i > df_search.size:
        pred_data = df_search['search'].values[i:]
    else:
        pred_data = df_search['search'].values[i:1000 + i]

    tmp = classifier.predict(pred_data)
    predictions += tmp.tolist()
    print("--------------------------")
    print("Made from {} to {}".format(i, i+1000))
    print("--------------------------")

df_search['predictions'] = predictions
print("--------------------------")
print("Predictions are ready, writing to file...")
print("--------------------------")
df_search.to_csv('../results/result.csv')
