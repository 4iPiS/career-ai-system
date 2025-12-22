#!/usr/bin/env python3
"""
@file career_ai.py
@brief Основной скрипт системы CareerAI для рекомендации направления обучения
@details Использует нейронную сеть для анализа ответов студента и выдает рекомендации
по 8 направлениям обучения с детальной информацией о каждом направлении
@author Разработчик
@version 2.1
@date 2025
"""

import sys
import os

# =============== ОТКЛЮЧЕНИЕ ВСЕХ WARNINGS И АСИНХРОННЫХ ОПЕРАЦИЙ ===============
import warnings

warnings.filterwarnings('ignore')

# КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Отключаем все асинхронные операции
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Отключаем GPU - источник асинхронных операций
os.environ['ABSL_LOG_LEVEL'] = 'ERROR'

# Устанавливаем синхронный режим для всего
os.environ['TF_DISABLE_MLIR'] = '1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

# =============== ИМПОРТ БИБЛИОТЕК ===============
import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path
import json
from datetime import datetime

# Импортируем TensorFlow ПОСЛЕ установки переменных окружения
import tensorflow as tf
from tensorflow import keras

# КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Настраиваем TensorFlow для полностью синхронной работы
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Отключаем все асинхронные оптимизации
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_soft_device_placement(True)

# Отключаем message о deprecated функциях
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# =============== ОТКЛЮЧЕНИЕ ДОПОЛНИТЕЛЬНЫХ СООБЩЕНИЙ ===============
try:
    # Отключаем прогресс-бары и другие сообщения
    from tqdm import tqdm

    tqdm._instances.clear()
except:
    pass

# ОСТАЛЬНОЙ КОД БЕЗ ИЗМЕНЕНИЙ...
# ==================== КОНФИГУРАЦИЯ БАЗЫ ДАННЫХ ПРОФЕССИЙ ====================

FIELD_DATABASE = {
    'Data Science': {
        'category': 'ИТ и анализ данных',
        'avg_salary': 180000,
        'employment_rate': 98,
        'study_years': 4,
        'human_description': 'Вы будете находить скрытые закономерности в данных, строить умные модели для предсказаний и помогать компаниям принимать решения на основе данных. Это одна из самых востребованных и высокооплачиваемых профессий в IT.',
        'skills_needed': ['Python', 'Статистика', 'SQL', 'Машинное обучение', 'Визуализация данных'],
        'personality': 'Аналитический склад ума, внимательность к деталям, любознательность, терпение',
        'pros': ['Высокая зарплата', 'Востребованность', 'Интересные задачи', 'Работа в любой индустрии'],
        'cons': ['Много математики', 'Нужно постоянно учиться', 'Сложные алгоритмы'],
        'next_steps': ['Начните с Python на Stepik', 'Пройдите курс статистики', 'Попробуйте Kaggle'],
        'universities': ['ВШЭ', 'МГУ', 'МИФИ', 'ИТМО']
    },

    'Computer Science': {
        'category': 'ИТ и программирование',
        'avg_salary': 170000,
        'employment_rate': 97,
        'study_years': 4,
        'human_description': 'Вы будете создавать программы с нуля, оптимизировать алгоритмы и глубоко понимать, как работают компьютеры. Это фундаментальное IT-образование, которое даст вам базу для любой IT-специальности.',
        'skills_needed': ['Алгоритмы', 'Структуры данных', 'C++/Java', 'Операционные системы', 'Сети'],
        'personality': 'Логическое мышление, терпение, системный подход, любовь к решению задач',
        'pros': ['Широкие возможности', 'Фундаментальные знания', 'Можно уйти в любую IT-специальность'],
        'cons': ['Сложная теория', 'Много абстрактных понятий', 'Требует усидчивости'],
        'next_steps': ['Изучите основы алгоритмов', 'Начните с языка Python или C++', 'Решайте задачи на LeetCode'],
        'universities': ['МГУ', 'МИФИ', 'МФТИ', 'СПбГУ']
    },

    'Software Engineering': {
        'category': 'Разработка ПО',
        'avg_salary': 160000,
        'employment_rate': 96,
        'study_years': 4,
        'human_description': 'Вы будете проектировать и создавать приложения, сайты и программы, которые используют миллионы людей. От маленьких мобильных приложений до огромных корпоративных систем.',
        'skills_needed': ['Java/Python/C#', 'Git', 'Базы данных', 'Архитектура ПО', 'Тестирование'],
        'personality': 'Командная работа, внимание к деталям, креативность в решении проблем',
        'pros': ['Много вакансий', 'Творческая работа', 'Быстрый карьерный рост'],
        'cons': ['Ненормированный график в стартапах', 'Много правок', 'Постоянные дедлайны'],
        'next_steps': ['Выберите язык (Python/Java)', 'Изучите Git и GitHub', 'Сделайте первый проект'],
        'universities': ['ИТМО', 'МИРЭА', 'МГТУ им. Баумана', 'УрФУ']
    },

    'Psychology': {
        'category': 'Социальные науки',
        'avg_salary': 120000,
        'employment_rate': 88,
        'study_years': 5,
        'human_description': 'Вы будете помогать людям разбираться с их проблемами, изучать поведение и мышление. Это профессия для тех, кто хочет понимать людей и помогать им становиться счастливее.',
        'skills_needed': ['Эмпатия', 'Наблюдательность', 'Коммуникация', 'Анализ поведения', 'Этика'],
        'personality': 'Доброта, терпение, умение слушать, тактичность, эмоциональный интеллект',
        'pros': ['Работа с людьми', 'Возможность помогать', 'Интерес к человеческой природе'],
        'cons': ['Эмоциональное выгорание', 'Долгое образование', 'Сложно найти первую работу'],
        'next_steps': ['Почитайте книги по психологии', 'Пройдите курсы по коммуникации',
                       'Поговорите с практикующим психологом'],
        'universities': ['МГУ', 'ВШЭ', 'СПбГУ', 'РГГУ']
    },

    'Design': {
        'category': 'Творчество и искусство',
        'avg_salary': 110000,
        'employment_rate': 85,
        'study_years': 4,
        'human_description': 'Вы будете создавать красивые и удобные интерфейсы, логотипы, айдентику брендов. Это профессия на стыке искусства и технологий, где ваши идеи становятся реальностью.',
        'skills_needed': ['Чувство стиля', 'Figma/Photoshop', 'Композиция', 'Колористика', 'Типографика'],
        'personality': 'Креативность, внимательность к деталям, насмотренность, чувство прекрасного',
        'pros': ['Творческая работа', 'Видимый результат', 'Гибкий график', 'Можно работать удаленно'],
        'cons': ['Субъективная оценка', 'Много правок от клиентов', 'Требует постоянного вдохновения'],
        'next_steps': ['Освойте Figma', 'Изучите основы композиции и цвета', 'Создайте первое портфолио'],
        'universities': ['Британская высшая школа дизайна', 'Профакадемия', 'Строгановка', 'НИУ ВШЭ']
    },

    'Economics': {
        'category': 'Бизнес и финансы',
        'avg_salary': 140000,
        'employment_rate': 92,
        'study_years': 4,
        'human_description': 'Вы будете анализировать рынки, строить экономические модели, прогнозировать тренды и помогать компаниям принимать финансовые решения. Это профессия для стратегов и аналитиков.',
        'skills_needed': ['Математика', 'Анализ данных', 'Excel', 'Понимание рынков', 'Финансовая грамотность'],
        'personality': 'Аналитический ум, стратегическое мышление, интерес к бизнесу, внимательность',
        'pros': ['Широкий выбор карьеры', 'Высокий доход', 'Престиж', 'Работа в банках и корпорациях'],
        'cons': ['Высокая конкуренция', 'Стресс', 'Ответственность за финансовые решения'],
        'next_steps': ['Пройдите курс по экономике', 'Читайте бизнес-литературу', 'Изучите Excel и статистику'],
        'universities': ['ВШЭ', 'МГУ', 'Финансовый университет', 'РЭУ им. Плеханова']
    },

    'Medicine': {
        'category': 'Здравоохранение',
        'avg_salary': 150000,
        'employment_rate': 99,
        'study_years': 6,
        'human_description': 'Вы будете спасать жизни, лечить болезни и помогать людям сохранять здоровье. Это одна из самых благородных и важных профессий, которая требует полной отдачи.',
        'skills_needed': ['Биология', 'Химия', 'Анатомия', 'Внимательность', 'Эмпатия'],
        'personality': 'Ответственность, стрессоустойчивость, желание помогать, внимательность',
        'pros': ['Помощь людям', 'Стабильность', 'Уважение в обществе', 'Постоянное развитие'],
        'cons': ['Долгое обучение', 'Высокая ответственность', 'Ненормированный график', 'Эмоциональные нагрузки'],
        'next_steps': ['Углубленно изучайте биологию и химию', 'Пообщайтесь с врачами', 'Пройдите курсы первой помощи'],
        'universities': ['Первый МГМУ', 'РНИМУ', 'СПбГМУ', 'МГМСУ']
    },

    'Linguistics': {
        'category': 'Гуманитарные науки',
        'avg_salary': 100000,
        'employment_rate': 89,
        'study_years': 4,
        'human_description': 'Вы будете изучать, как устроены языки, работать переводчиком или заниматься компьютерной лингвистикой. Это профессия для любителей языков и культур.',
        'skills_needed': ['Иностранные языки', 'Анализ текста', 'Память', 'Коммуникация', 'Культурология'],
        'personality': 'Любознательность, внимательность к деталям, любовь к языкам, терпение',
        'pros': ['Работа с языками', 'Возможность путешествовать', 'Разнообразие задач'],
        'cons': ['Конкуренция с носителями', 'Много рутинной работы', 'Требует постоянной практики'],
        'next_steps': ['Углубленно изучайте иностранные языки', 'Читайте литературу в оригинале', 'Попробуйте перевод'],
        'universities': ['МГУ', 'ВШЭ', 'МГЛУ', 'СПбГУ']
    }
}


# ==================== КЛАСС НЕЙРОННОЙ СЕТИ ====================

class CareerAdvisorAI:
    def __init__(self, model_dir='career_model'):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.fields_info = FIELD_DATABASE

    def load_model(self):
        if not self.model_dir.exists():
            print(f"ОШИБКА: Папка с моделью не найдена: {self.model_dir}")
            print("       Сначала обучите модель: python train_model.py")
            sys.exit(1)

        try:
            model_path = self.model_dir / 'best_model.h5'
            if not model_path.exists():
                model_path = self.model_dir / 'career_model.h5'

            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Явно указываем синхронную загрузку
            self.model = keras.models.load_model(model_path, compile=False)

            # Компилируем модель с простыми настройками
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
            self.label_encoder = joblib.load(self.model_dir / 'label_encoder.pkl')

            metadata_path = self.model_dir / 'model_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

        except Exception as e:
            print(f"ОШИБКА загрузки модели: {e}")
            print("       Попробуйте переобучить модель: python train_model.py")
            sys.exit(1)

    def analyze_student_profile(self, csv_path):
        try:
            student_data = pd.read_csv(csv_path)

        except Exception as e:
            print(f"ОШИБКА чтения файла: {e}")
            sys.exit(1)

        X_processed = self._prepare_student_data(student_data)

        predictions = self.model.predict(X_processed, verbose=0)[0]

        top_indices = np.argsort(predictions)[::-1][:5]

        recommendations = []
        for i, idx in enumerate(top_indices[:5]):
            field_name = self.label_encoder.inverse_transform([idx])[0]
            probability = predictions[idx] * 100

            info = self.fields_info.get(field_name, {
                'category': 'Общее',
                'avg_salary': 100000,
                'employment_rate': 85,
                'human_description': 'Перспективное направление',
            })

            recommendations.append({
                'rank': i + 1,
                'field': field_name,
                'probability': round(probability, 1),
                'match_level': self._get_match_level(probability),
                'category': info.get('category', ''),
                'avg_salary': info.get('avg_salary', 0),
                'avg_salary_formatted': f"{info.get('avg_salary', 0):,} руб.",
                'employment_rate': info.get('employment_rate', 0),
                'employment_formatted': f"{info.get('employment_rate', 0)}%",
                'study_years': info.get('study_years', 4),
                'description': info.get('human_description', ''),
                'skills': info.get('skills_needed', []),
                'personality': info.get('personality', ''),
                'pros': info.get('pros', []),
                'cons': info.get('cons', []),
                'next_steps': info.get('next_steps', []),
                'universities': info.get('universities', [])
            })

        all_recommendations = []
        for i, idx in enumerate(top_indices):
            field_name = self.label_encoder.inverse_transform([idx])[0]
            probability = predictions[idx] * 100
            all_recommendations.append({
                'rank': i + 1,
                'field': field_name,
                'probability': round(probability, 1),
                'probability_raw': float(probability),
                'field_encoded': int(idx)
            })

        return recommendations, student_data, all_recommendations, predictions

    def _prepare_student_data(self, student_df):
        expected_features = [
            'math_skill', 'programming_skill', 'communication_skill',
            'creativity', 'analytical_thinking', 'memory',
            'attention_to_detail', 'teamwork', 'leadership',
            'stress_tolerance', 'adaptability', 'curiosity',
            'math_interest', 'physics_interest', 'chemistry_interest',
            'biology_interest', 'it_interest', 'economics_interest',
            'humanities_interest', 'art_interest', 'sports_interest',
            'desired_salary', 'preferred_work_env', 'work_life_balance'
        ]

        processed_data = {}

        for feature in expected_features:
            if feature in student_df.columns:
                processed_data[feature] = student_df[feature].iloc[0]
            else:
                default_values = {
                    'desired_salary': 120000,
                    'preferred_work_env': 2,
                    'work_life_balance': 3.5,
                }
                processed_data[feature] = default_values.get(feature, 3.0)

        processed_df = pd.DataFrame([processed_data])[expected_features]

        X_processed = self.scaler.transform(processed_df.values.astype('float32'))

        return X_processed

    def _get_match_level(self, probability):
        if probability >= 85:
            return "ИДЕАЛЬНОЕ СОЧЕТАНИЕ"
        elif probability >= 70:
            return "ОТЛИЧНОЕ СОЧЕТАНИЕ"
        elif probability >= 55:
            return "ХОРОШЕЕ СОЧЕТАНИЕ"
        elif probability >= 40:
            return "СРЕДНЕЕ СОЧЕТАНИЕ"
        else:
            return "НИЗКОЕ СОЧЕТАНИЕ"


# ==================== СОЗДАНИЕ CSV ФАЙЛА С ТОП-5 РЕКОМЕНДАЦИЯМИ ====================

def create_top5_recommendations_csv(all_recommendations, csv_path, output_dir="output"):
    """
    Создает CSV файл ТОЛЬКО с топ-5 рекомендациями в числовом формате
    и сохраняет его в папке output

    @param all_recommendations: список из 5 рекомендаций
    @param csv_path: путь к исходному файлу
    @param output_dir: папка для сохранения результатов

    @return: путь к созданному CSV файлу
    """
    # Создаем папку output, если она не существует
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Получаем имя исходного файла без расширения для использования в имени выходного файла
    original_filename = Path(csv_path).stem

    # Создаем словарь для одной строки данных
    row_data = {}

    # Для каждой из 5 рекомендаций добавляем 3 колонки
    for i, rec in enumerate(all_recommendations[:5]):
        prefix = f"rec_{i + 1}"

        # 1. Индекс профессии (0-7)
        row_data[f"{prefix}_field_encoded"] = rec['field_encoded']

        # 2. Вероятность в процентах (0.0-100.0)
        row_data[f"{prefix}_probability"] = rec['probability_raw']

        # 3. Уровень совпадения (1-5)
        probability = rec['probability_raw']
        if probability >= 85:
            match_code = 5  # ИДЕАЛЬНОЕ СОЧЕТАНИЕ
        elif probability >= 70:
            match_code = 4  # ОТЛИЧНОЕ СОЧЕТАНИЕ
        elif probability >= 55:
            match_code = 3  # ХОРОШЕЕ СОЧЕТАНИЕ
        elif probability >= 40:
            match_code = 2  # СРЕДНЕЕ СОЧЕТАНИЕ
        else:
            match_code = 1  # НИЗКОЕ СОЧЕТАНИЕ

        row_data[f"{prefix}_match_level_code"] = match_code

    # Создаем DataFrame с одной строкой
    df = pd.DataFrame([row_data])

    # Формируем имя файла
    filename = f"career_top5_{original_filename}_{timestamp}.csv"
    output_file = output_path / filename

    # Сохраняем в CSV файл
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"{filename}")

    return output_file


# ==================== СОЗДАНИЕ ПОДРОБНОГО ОТЧЕТА В JSON ====================

def create_detailed_report(recommendations, student_data, all_recommendations, output_dir="output"):
    """
    Создает подробный отчет в JSON формате и сохраняет в папке output

    @param recommendations: детальные рекомендации
    @param student_data: исходные данные студента
    @param all_recommendations: все рекомендации
    @param output_dir: папка для сохранения

    @return: путь к созданному JSON файлу
    """
    # Создаем папку output, если она не существует
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Подготовка данных для JSON
    report = {
        "timestamp": timestamp,
        "generated_at": datetime.now().isoformat(),
        "analysis": {
            "total_fields_analyzed": 8,
            "top_recommendations_count": 5,
            "best_match": recommendations[0] if recommendations else None
        },
        "recommendations": recommendations,
        "all_predictions": all_recommendations,
        "student_profile_summary": {
            "skills_assessed": len(student_data.columns),
            "profile_data": student_data.iloc[0].to_dict() if not student_data.empty else {}
        }
    }

    # Формируем имя файла
    filename = f"career_detailed_report_{timestamp}.json"
    output_file = output_path / filename

    # Сохраняем в JSON файл
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return output_file


# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def main():
    parser = argparse.ArgumentParser(description='ИИ для выбора направления обучения')
    parser.add_argument('csv_file', help='CSV файл с профилем студента')
    parser.add_argument('--detailed', action='store_true', help='Создать подробный отчет в JSON')
    parser.add_argument('--output-dir', default='output',
                        help='Папка для сохранения результатов (по умолчанию: output)')

    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"ОШИБКА: Файл не найден: {args.csv_file}")
        sys.exit(1)

    # Создаем папку для выходных файлов
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Создаем контекст для синхронного выполнения
    tf.config.set_visible_devices([], 'GPU')  # Гарантированно отключаем GPU

    advisor = CareerAdvisorAI()
    advisor.load_model()

    try:
        recommendations, student_data, all_recommendations, predictions = advisor.analyze_student_profile(args.csv_file)

        # Создаем CSV файл ТОЛЬКО с топ-5 рекомендациями
        csv_file = create_top5_recommendations_csv(all_recommendations, csv_path, args.output_dir)

        # Если запрошен подробный отчет, создаем JSON
        if args.detailed:
            json_file = create_detailed_report(recommendations, student_data, all_recommendations, args.output_dir)

    except Exception as e:
        print(f"ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()