import pandas as pd
from WordsTokenizer import WordsTokenizer
from SemanticsClassifier import SemanticsClassifier


def print_usage():
    print("Предсказания семантики на кофейной гуще! Выберите опцию:")
    print("1. Указать путь до файла с семантикой")
    print("2. Указать путь до файла с запросами")
    print("3. Указать параметры модели. Старая при этом будет удалена")
    print("4. Указать параметры токенайзера. Старый токенайзер и старая МОДЕЛЬ при этом будут удалены, но параметры "
          "модели сохранятся")
    print("5. Указать путь для вывода результатов")
    print("6. Запусить предсказания")
    print("?. Показать это окно еще раз")
    print("q. Выйти из программы")


def read_file():
    print("Введите путь к файлу...")
    s = input()
    try:
        df = pd.read_csv(s)
    except:
        print("Не удалось открыть файл. Попытаться еще раз? (y/n)")
        s = input().lower()
        if s == 'y':
            return read_file()
        else:
            return None
    print('Файл успешно считан! Возврат в главное меню...')
    print_delimeter()
    print()
    print_usage()
    return df


def print_delimeter():
    print('------------------------------------')


def read_param_model():
    print("Введите параметры a, n_jobs, be_verbose, p через пробел. Их описание:")
    print("a -- Вес косинусного расстояние (для Левенштейна будет 1-a)")
    print("n_jobs -- число потоков")
    # FIXME: Сделать обработку be_verbose
    print("be_verbose -- Писать больше информации. Пока не работает: пишет много всегда")
    print("p -- минимальный порог расстояния, ниже которого запрос будет воспринят как 'Unknown'. От 0 до 1")
    params = input().split()
    try:
        a = float(params[0])
        n_jobs = int(params[1])
        be_verbose = bool(params[2])
        p = float(params[3])
    except:
        print("Некорректный ввод. Хотите попробовать еще раз? (y/n)")
        ans = input().lower()
        if ans == 'y':
            read_param_model()
        else:
            return None
    return a, n_jobs, be_verbose, p


def read_param_tokenizer():
    print("Введите параметры p, n_jobs через пробел. Их описание:")
    print("p -- максимальное нормированное расстояние Левенштейна (степень схожести)"
          " между двумя словами, при котором их считать одним и тем же словом. От 0 до 1")
    print("n_jobs -- число потоков")
    params = input().split()
    try:
        p = float(params[0])
        n_jobs = int(params[1])
    except:
        print("Некорректный ввод. Хотите попробовать еще раз? (y/n)")
        ans = input().lower()
        if ans == 'y':
            read_param_model()
        else:
            return None
    return p, n_jobs


if __name__ == "__main__":
    print_usage()
    tokenizer = WordsTokenizer()
    model = SemanticsClassifier()
    out_path = '../results/result.csv'
    df_sem, df_search = None, None
    while True:
        n = input()
        if n not in ['1', '2', '3', '4', '5', '6', '?', 'q']:
            print("Некорректный ввод, повторите еще раз. Если нужна помощь, введите '?' без кавычек")
            continue
        if n == '1':
            df_sem = read_file()
            if df_sem is None:
                print("Файл не был считан. Выберите другую опцию")
                print_delimeter()
                print()
                print_usage()
        if n == '2':
            df_search = read_file()
            if df_search is None:
                print("Файл не был считан. Выберите другую опцию")
                print_delimeter()
                print()
                print_usage()
        if n == '3':
            par1 = read_param_model()
            model = SemanticsClassifier(a=par1[0], n_jobs=par1[1], be_verbose=par1[2], p=par1[3], tokenizer=tokenizer)
            print("Параметры модели установлены. Возврат в главное меню...")
            print_delimeter()
            print()
            print_usage()
        if n == '4':
            par2 = read_param_tokenizer()
            tokenizer = WordsTokenizer(p=par2[0], n_jobs=par2[1])
            model = SemanticsClassifier(a=par1[0], n_jobs=par1[1], be_verbose=par1[2], p=par1[3], tokenizer=tokenizer)
            print("Параметры установлены. Возврат в главное меню...")
            print_delimeter()
            print()
            print_usage()
        if n == '5':
            print("Введите путь файла для вывода...")
            out_path = input()
            print("Путь установлен. Возврат в главное меню...")
            print_delimeter()
            print()
            print_usage()
        if n == '6':
            # TODO: Провека на то, что пользователь ввел все файлы!!!
            if df_sem is None or df_search is None:
                print("Не все файлы указаны. Возврат в главное меню...")
                print_delimeter()
                print()
                print_usage()
                continue
            tokenizer.fit(df_search['search'].values)
            model.train(df_sem['keyword_name'].values)
            predictions = model.predict(df_search['search'].values)
            df_search['predictions'] = predictions
            print_delimeter()
            print("Предсказания готовы. Записываем в файл...")
            df_search.to_csv(out_path)
            print('Готово! Возврат в главное меню...')
            print_delimeter()
            print()
            print_usage()
        if n == '?':
            print_usage()
        if n == 'q':
            print("Благодарим за использование наших услуг!")
            break