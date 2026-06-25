# Прогнозирование ценовой категории автомобиля

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)

## О проекте

Проект по прогнозированию ценовой категории автомобиля на основе характеристик из объявлений о продаже. Задача: по признакам автомобиля (год выпуска, пробег, регион, состояние и др.) определить его ценовую категорию. Результат: обучены и сравнены три модели машинного обучения. Лучшая модель (Random Forest) достигла точности 72.9% на кросс-валидации.

## Данные

Датасет содержит объявления о продаже автомобилей. Признаки: год выпуска, пробег, производитель, модель, тип топлива, коробка передач, регион, текстовое описание и др. Целевая переменная: price_category - категория цены автомобиля.

## Этапы работы

1. Очистка данных: удаление дубликатов, обработка пропусков (>20% - удаление признаков, <20% - заполнение), устранение выбросов методом IQR
2. Feature Engineering: создание новых признаков (возрастная категория, длина описания, частота упоминания модели, временные характеристики), кодирование категориальных признаков (One-Hot Encoding), стандартизация числовых признаков (StandardScaler)
3. Моделирование: Random Forest Classifier, Logistic Regression, MLPClassifier (нейронная сеть)
4. Оценка качества: кросс-валидация (5 фолдов), сравнение точности моделей

## 📈 Результаты

| Модель | Средняя точность (CV) | Стабильность |
|--------|----------------------|--------------|
| **Random Forest** | **0.729** | Высокая (σ=0.006) |
| MLPClassifier | 0.726 | Высокая (σ=0.006) |
| Logistic Regression | 0.704 | Высокая (σ=0.008) |

**Лучшая модель:** Random Forest Classifier (сохранена в `models/model.pickle`)

## Как запустить:

## Как запустить

1. Клонировать репозиторий
2. Установить зависимости при необходимости
3. Запустить Jupyter Notebook: `vehicles.ipynb`
   
## Использование модели (python):

import pickle
  
import pandas as pd

with open('models/model.pickle', 'rb') as file:
  model = pickle.load(file)

  prediction = model.predict(prepared_data)

## Технологии

- Python 3.9+
- Pandas
- Matplotlib, Missingno
- Sklearn
- Jupyter Notebook

## Контакты

**Автор:** Андрей Тырлышкин

*   [GitHub](https://github.com/andreytyrlyshkin)
*   Telegram: @Andrey_Tyrlyshkin
*   Email: andreytyrlyshkin2000@gmail.com
