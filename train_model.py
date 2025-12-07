#!/usr/bin/env python3
"""
Скрипт для обучения нейросетевой модели рекомендаций
Запуск: python train_model.py
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import json

def create_training_data():
    """Создание тренировочных данных"""
    print("Создание тренировочных данных...")
    
    # Примерные данные (в реальности нужно больше)
    data = []
    
    # Технические профили
    for i in range(50):
        data.append({
            'math_skill': np.random.uniform(4, 5),
            'programming_skill': np.random.uniform(4, 5),
            'communication_skill': np.random.uniform(2, 4),
            'creativity': np.random.uniform(2, 4),
            'analytical_thinking': np.random.uniform(4, 5),
            'memory': np.random.uniform(3, 5),
            'attention_to_detail': np.random.uniform(4, 5),
            'teamwork': np.random.uniform(3, 4),
            'leadership': np.random.uniform(2, 4),
            'stress_tolerance': np.random.uniform(3, 5),
            'adaptability': np.random.uniform(3, 5),
            'curiosity': np.random.uniform(4, 5),
            'math_interest': np.random.uniform(4, 5),
            'physics_interest': np.random.uniform(3, 5),
            'chemistry_interest': np.random.uniform(2, 4),
            'biology_interest': np.random.uniform(2, 4),
            'it_interest': np.random.uniform(4, 5),
            'economics_interest': np.random.uniform(2, 4),
            'humanities_interest': np.random.uniform(2, 3),
            'art_interest': np.random.uniform(1, 3),
            'sports_interest': np.random.uniform(2, 4),
            'desired_salary': np.random.randint(120000, 200000),
            'preferred_work_env': np.random.choice([1, 2]),  # 1-офис, 2-удаленка
            'work_life_balance': np.random.randint(3, 5),
            'chosen_field': np.random.choice(['Data Science', 'Computer Science', 'Software Engineering'])
        })
    
    # Гуманитарные профили
    for i in range(50):
        data.append({
            'math_skill': np.random.uniform(2, 4),
            'programming_skill': np.random.uniform(1, 3),
            'communication_skill': np.random.uniform(4, 5),
            'creativity': np.random.uniform(3, 5),
            'analytical_thinking': np.random.uniform(3, 5),
            'memory': np.random.uniform(4, 5),
            'attention_to_detail': np.random.uniform(3, 5),
            'teamwork': np.random.uniform(4, 5),
            'leadership': np.random.uniform(3, 5),
            'stress_tolerance': np.random.uniform(3, 4),
            'adaptability': np.random.uniform(4, 5),
            'curiosity': np.random.uniform(4, 5),
            'math_interest': np.random.uniform(2, 4),
            'physics_interest': np.random.uniform(1, 3),
            'chemistry_interest': np.random.uniform(2, 4),
            'biology_interest': np.random.uniform(3, 5),
            'it_interest': np.random.uniform(1, 3),
            'economics_interest': np.random.uniform(3, 5),
            'humanities_interest': np.random.uniform(4, 5),
            'art_interest': np.random.uniform(3, 5),
            'sports_interest': np.random.uniform(2, 4),
            'desired_salary': np.random.randint(80000, 150000),
            'preferred_work_env': np.random.choice([2, 3]),  # 2-удаленка, 3-гибрид
            'work_life_balance': np.random.randint(4, 5),
            'chosen_field': np.random.choice(['Psychology', 'Linguistics', 'Design'])
        })
    
    # Творческие профили
    for i in range(30):
        data.append({
            'math_skill': np.random.uniform(2, 3),
            'programming_skill': np.random.uniform(1, 3),
            'communication_skill': np.random.uniform(3, 5),
            'creativity': np.random.uniform(4, 5),
            'analytical_thinking': np.random.uniform(2, 4),
            'memory': np.random.uniform(3, 5),
            'attention_to_detail': np.random.uniform(4, 5),
            'teamwork': np.random.uniform(3, 5),
            'leadership': np.random.uniform(2, 4),
            'stress_tolerance': np.random.uniform(3, 4),
            'adaptability': np.random.uniform(4, 5),
            'curiosity': np.random.uniform(4, 5),
            'math_interest': np.random.uniform(1, 3),
            'physics_interest': np.random.uniform(1, 2),
            'chemistry_interest': np.random.uniform(2, 3),
            'biology_interest': np.random.uniform(2, 4),
            'it_interest': np.random.uniform(2, 4),
            'economics_interest': np.random.uniform(2, 4),
            'humanities_interest': np.random.uniform(3, 5),
            'art_interest': np.random.uniform(4, 5),
            'sports_interest': np.random.uniform(2, 4),
            'desired_salary': np.random.randint(70000, 120000),
            'preferred_work_env': np.random.choice([2, 3, 4]),  # разные варианты
            'work_life_balance': np.random.randint(4, 5),
            'chosen_field': np.random.choice(['Design', 'Psychology', 'Linguistics'])
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/training_data.csv', index=False)
    print(f"Создано {len(df)} тренировочных примеров")
    
    return df

def build_model(input_dim, num_classes):
    """Создание архитектуры нейросети"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # Первый скрытый слой
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Второй скрытый слой
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Третий скрытый слой
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        # Выходной слой
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy',
                keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy'),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy')]
    )
    
    return model

def main():
    """Обучение модели"""
    print("="*80)
    print("ОБУЧЕНИЕ НЕЙРОСЕТЕВОЙ МОДЕЛИ ДЛЯ РЕКОМЕНДАЦИЙ")
    print("="*80)
    print()
    
    # Создаем папки
    Path('data').mkdir(exist_ok=True)
    Path('career_model').mkdir(exist_ok=True)
    
    # Создаем или загружаем данные
    if Path('data/training_data.csv').exists():
        print("Загружаю существующие тренировочные данные...")
        df = pd.read_csv('data/training_data.csv')
    else:
        print("Создаю новые тренировочные данные...")
        df = create_training_data()
    
    print(f"   Всего примеров: {len(df)}")
    print(f"   Направлений: {df['chosen_field'].nunique()}")
    print(f"   Признаков: {len(df.columns) - 1}")
    
    # Подготовка данных
    feature_columns = [col for col in df.columns if col != 'chosen_field']
    X = df[feature_columns].values
    y = df['chosen_field'].values
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nРазделение данных:")
    print(f"   Обучающая выборка: {X_train.shape[0]} примеров")
    print(f"   Валидационная выборка: {X_val.shape[0]} примеров")
    print(f"   Классов: {len(label_encoder.classes_)}")
    
    # Создание модели
    print("\nСоздание нейросетевой архитектуры...")
    model = build_model(X.shape[1], len(label_encoder.classes_))
    
    # Вывод архитектуры
    model.summary()
    
    # Callbacks для обучения
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='career_model/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Обучение
    print("\nНачинаю обучение нейросети...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Сохранение модели и препроцессоров
    print("\nСохраняю модель и вспомогательные файлы...")
    
    # Сохраняем полную модель
    model.save('career_model/career_model.h5')
    
    # Сохраняем препроцессоры
    joblib.dump(scaler, 'career_model/scaler.pkl')
    joblib.dump(label_encoder, 'career_model/label_encoder.pkl')
    
    # Сохраняем метаданные
    metadata = {
        'training_date': pd.Timestamp.now().isoformat(),
        'num_samples': len(df),
        'num_classes': len(label_encoder.classes_),
        'classes': label_encoder.classes_.tolist(),
        'features': feature_columns,
        'input_shape': X.shape[1],
        'model_architecture': 'Dense(128)-Dense(64)-Dense(32)',
        'final_accuracy': float(history.history['val_accuracy'][-1]),
        'final_top3_accuracy': float(history.history['val_top3_accuracy'][-1])
    }
    
    with open('career_model/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Сохраняем историю обучения
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('career_model/training_history.csv', index=False)
    
    print("\n" + "="*80)
    print("МОДЕЛЬ УСПЕШНО ОБУЧЕНА И СОХРАНЕНА!")
    print("="*80)
    print(f"\nПапка с моделью: career_model/")
    print(f"   • career_model.h5 - нейросеть")
    print(f"   • scaler.pkl - нормализатор признаков")
    print(f"   • label_encoder.pkl - кодировщик направлений")
    print(f"   • training_history.csv - история обучения")
    print(f"   • model_metadata.json - информация о модели")
    
    print(f"\nРезультаты обучения:")
    print(f"   Final Accuracy: {metadata['final_accuracy']:.2%}")
    print(f"   Top-3 Accuracy: {metadata['final_top3_accuracy']:.2%}")
    
    print(f"\nТеперь можно использовать модель:")
    print(f"   python career_ai.py пример_студента.csv")

if __name__ == "__main__":
    main()