import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import datetime

# Исходные данные
data = [
    {"User ID": 201382457, "Username": "mell1433", "Q1": 4, "Q2": 2, "Q3": 3, "Q4": 2, "Q5": 4, "Q6": 3, "Q7": 4, "Q8": 1, "Q9": 2, "Q10": 4},
    {"User ID": 5929292461, "Username": "OStasya", "Q1": 3, "Q2": 2, "Q3": 3, "Q4": 4, "Q5": 4, "Q6": 2, "Q7": 2, "Q8": 4, "Q9": 4, "Q10": 4, "end_time": datetime.datetime(2024, 8, 6, 15, 7, 46, 85618)},
    {"User ID": 806241975, "Username": "Marie_Paramonova", "Q1": 4, "Q2": 1, "Q3": 4, "Q4": 4, "Q5": 4, "Q6": 2, "Q7": 4, "Q8": 4, "Q9": 1, "Q10": 1, "end_time": datetime.datetime(2024, 8, 6, 15, 7, 46, 92619)},
    # (добавьте все остальные записи сюда)
]

# Создание DataFrame
df = pd.DataFrame(data)

# Удаление дубликатов
df.drop_duplicates(inplace=True)

# Проверка и заполнение пропущенных значений
df.fillna(value={"Username": "Unknown"}, inplace=True)

# Преобразование типов данных
for col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']:
    df[col] = df[col].astype(int)

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