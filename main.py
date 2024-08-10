import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import datetime
import re

# Функция для парсинга данных из строки
def parse_line(line):
    pattern = r"User ID- (\d+), Username- (\S+), Q1- (\d+), Q2- (\d+), Q3- (\d+), Q4- (\d+), Q5- (\d+), Q6- (\d+), Q7- (\d+), Q8- (\d+), Q9- (\d+), Q10- (\d+)"
    match = re.search(pattern, line)
    if match:
        return {
            'User ID': int(match.group(1)),
            'Username': match.group(2),
            'Q1': int(match.group(3)),
            'Q2': int(match.group(4)),
            'Q3': int(match.group(5)),
            'Q4': int(match.group(6)),
            'Q5': int(match.group(7)),
            'Q6': int(match.group(8)),
            'Q7': int(match.group(9)),
            'Q8': int(match.group(10)),
            'Q9': int(match.group(11)),
            'Q10': int(match.group(12)),
        }
    else:
        print(f"Не удалось распарсить строку: {line}")
        return None

# Чтение файла построчно
data = []
with open('utf-8logs.txt', 'r', encoding='utf-8') as file:
    for line in file:
        parsed_line = parse_line(line)
        if parsed_line is not None:
            data.append(parsed_line)

# Создание DataFrame
df = pd.DataFrame(data)

# Проверка содержания DataFrame
print("Существующие столбцы в DataFrame:")
print(df.columns)

print("Первые строки DataFrame:")
print(df.head())

# Проверка количества распарсенных строк
print(f"Количество успешно распарсенных строк: {len(df)}")

# Преобразование типов данных столбцов, если они существуют
for col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']:
    if col in df.columns:
        df[col] = df[col].astype(int)
    else:
        print(f"Столбец {col} отсутствует в DataFrame")








# Дополнительный код для дальнейшей обработки DataFrame (если требуется)
# Выводим DataFrame для проверки
print("Начальные данные:")
print(df.head())

# Распределение ответов по каждому вопросу
questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
for q in questions:
    plt.figure()
    df[q].value_counts().sort_index().plot(kind='bar')
    plt.title(f'Распределение ответов для {q}')
    plt.xlabel('Ответ')
    plt.ylabel('Частота')
    plt.show()

# Корреляционная матрица
correlation_matrix = df[questions].corr()

# Визуализация корреляционной матрицы
plt.figure(figsize=(10, 8))
plt.matshow(correlation_matrix, fignum=1)
plt.xticks(range(len(questions)), questions, rotation=90)
plt.yticks(range(len(questions)), questions)
plt.colorbar()
plt.title('Корреляционная матрица ответов')
plt.show()

# Вывод корреляционной матрицы
print("Корреляционная матрица:")
print(correlation_matrix)

# Определение оптимального количества кластеров с помощью метода локтя
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(df[questions])
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Метод локтя для определения оптимального количества кластеров')
plt.xlabel('Количество кластеров')
plt.ylabel('Инерция')
plt.show()

# Кластеризация с оптимальным количеством кластеров (например, 3)
kmeans = KMeans(n_clusters=3, random_state=0).fit(df[questions])
df['Cluster'] = kmeans.labels_

# Выводим DataFrame с метками кластеров
print("Данные с метками кластеров:")
print(df.head())

# Визуализация кластеров
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Q1', y='Q2', hue='Cluster', palette='viridis')
plt.title('Визуализация кластеров на основе Q1 и Q2')
plt.show()