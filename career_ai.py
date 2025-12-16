#!/usr/bin/env python3
"""
@file career_ai.py
@brief Основной скрипт системы CareerAI для рекомендации направления обучения
@details Использует нейронную сеть для анализа ответов студента и выдает рекомендации
по 8 направлениям обучения с детальной информацией о каждом направлении
@author Разработчик
@version 2.0
@date 2025
"""

import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import argparse
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
        'next_steps': ['Почитайте книги по психологии', 'Пройдите курсы по коммуникации', 'Поговорите с практикующим психологом'],
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
        
        print("Загружаю искусственный интеллект для рекомендаций...")
        
        try:
            model_path = self.model_dir / 'best_model.h5'
            if not model_path.exists():
                model_path = self.model_dir / 'career_model.h5'
            
            self.model = keras.models.load_model(model_path)
            self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
            self.label_encoder = joblib.load(self.model_dir / 'label_encoder.pkl')
            
            metadata_path = self.model_dir / 'model_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    print(f"УСПЕХ: ИИ загружен (точность: {metadata.get('test_accuracy', 0):.1%})")
            else:
                print(f"УСПЕХ: ИИ загружен из {model_path.name}")
            
            print(f"       Знает {len(self.label_encoder.classes_)} направлений")
            
        except Exception as e:
            print(f"ОШИБКА загрузки модели: {e}")
            print("       Попробуйте переобучить модель: python train_model.py")
            sys.exit(1)
    
    def analyze_student_profile(self, csv_path):
        print(f"Анализирую профиль студента из {csv_path}")
        
        try:
            student_data = pd.read_csv(csv_path)
            print(f"Найдено {len(student_data)} ответов")
            
        except Exception as e:
            print(f"ОШИБКА чтения файла: {e}")
            sys.exit(1)
        
        X_processed = self._prepare_student_data(student_data)
        
        print("ИИ анализирует ваш профиль...")
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

# ==================== КЛАСС ИНТЕРФЕЙСА ====================

class CareerAdvisorInterface:
    def __init__(self, advisor):
        self.advisor = advisor
    
    def print_header(self, student_name=None):
        print("\n" + "="*80)
        print("УМНЫЙ ВЫБОР НАПРАВЛЕНИЯ ОБУЧЕНИЯ")
        print("="*80)
        
        if student_name:
            print(f"\nСтудент: {student_name}")
        
        print("\nИИ анализирует ваши навыки, интересы и личность...")
        print("="*80)
    
    def print_recommendation(self, rec, detailed=True):
        print(f"\nРЕКОМЕНДАЦИЯ #{rec['rank']}:")
        print(f"="*60)
        print(f"НАПРАВЛЕНИЕ:  {rec['field']}")
        print(f"СОВПАДЕНИЕ:   {rec['match_level']} ({rec['probability']}%)")
        print(f"КАТЕГОРИЯ:    {rec['category']}")
        print(f"СРЕДНЯЯ ЗП:   {rec['avg_salary_formatted']}")
        print(f"ТРУДОУСТРОЙСТВО: {rec['employment_formatted']}")
        print(f"СРОК ОБУЧЕНИЯ: {rec['study_years']} лет")
        
        if detailed:
            print(f"\nОПИСАНИЕ:")
            print(f"   {rec['description']}")
            
            print(f"\nНЕОБХОДИМЫЕ НАВЫКИ:")
            for skill in rec['skills']:
                print(f"   - {skill}")
            
            print(f"\nПОРТРЕТ СПЕЦИАЛИСТА:")
            print(f"   {rec['personality']}")
            
            print(f"\nПРЕИМУЩЕСТВА:")
            for pro in rec['pros']:
                print(f"   + {pro}")
            
            print(f"\nСЛОЖНОСТИ:")
            for con in rec['cons']:
                print(f"   - {con}")
            
            print(f"\nСЛЕДУЮЩИЕ ШАГИ:")
            for i, step in enumerate(rec['next_steps'], 1):
                print(f"   {i}. {step}")
            
            print(f"\nВУЗЫ ДЛЯ ПОСТУПЛЕНИЯ:")
            for uni in rec['universities']:
                print(f"   - {uni}")
    
    def print_all_recommendations(self, recommendations, student_data, csv_path):
        self.print_header()
        
        print("\nВАШИ ОТВЕТЫ:")
        print("-"*40)
        key_answers = student_data.iloc[0].to_dict()
        for key, value in list(key_answers.items())[:10]:
            print(f"{key:<25}: {value}")
        
        print("\n" + "="*80)
        print("ТОП-5 РЕКОМЕНДАЦИЙ ИСКУССТВЕННОГО ИНТЕЛЛЕКТА")
        print("="*80)
        
        for i, rec in enumerate(recommendations):
            self.print_recommendation(rec, detailed=(i == 0))
            
            if i < len(recommendations) - 1:
                print("\n" + "-"*60)
        
        print("\n" + "="*80)
        print("ПОЛНЫЙ СПИСОК РЕКОМЕНДАЦИЙ")
        print("="*80)
        print("Ранг | Направление          | Совпадение  | Уровень")
        print("-"*60)
        
        for i, rec in enumerate(recommendations):
            print(f"  {rec['rank']:<3} | {rec['field']:<20} | {rec['probability']:>6.1f}%     | {rec['match_level']}")

# ==================== СОЗДАНИЕ CSV ФАЙЛА С ТОП-5 РЕКОМЕНДАЦИЯМИ ====================

def create_top5_recommendations_csv(all_recommendations, csv_path):
    """
    Создает CSV файл ТОЛЬКО с топ-5 рекомендациями в числовом формате
    
    @param all_recommendations: список из 5 рекомендаций
    @param csv_path: путь к исходному файлу
    
    @return: создает CSV файл с 3 колонками на каждую рекомендацию
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Создаем словарь для одной строки данных
    row_data = {}
    
    # Для каждой из 5 рекомендаций добавляем 3 колонки
    for i, rec in enumerate(all_recommendations[:5]):
        prefix = f"rec_{i+1}"
        
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
    
    # Сохраняем в CSV файл
    filename = f"career_top5_recommendations_{timestamp}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"\nСоздан файл с топ-5 рекомендациями: {filename}")
    print("\nСОДЕРЖАНИЕ ФАЙЛА:")
    print("="*60)
    print(df.to_string())
    
    print(f"\nСтруктура данных (15 колонок):")
    print("-"*60)
    print("rec_1_field_encoded    - индекс первой рекомендации (0-7)")
    print("rec_1_probability      - вероятность первой рекомендации (%)")
    print("rec_1_match_level_code - уровень совпадения первой (1-5)")
    print("...")
    print("rec_5_field_encoded    - индекс пятой рекомендации (0-7)")
    print("rec_5_probability      - вероятность пятой рекомендации (%)")
    print("rec_5_match_level_code - уровень совпадения пятой (1-5)")
    
    return filename

# ==================== ГЕНЕРАЦИЯ ШАБЛОНА CSV ====================

def create_template_csv():
    print("\n" + "="*80)
    print("СОЗДАНИЕ ШАБЛОНА ДЛЯ ТЕСТИРОВАНИЯ")
    print("="*80)
    
    template_data = {
        'math_skill': [4.5],
        'programming_skill': [4.0],
        'communication_skill': [3.5],
        'creativity': [3.0],
        'analytical_thinking': [4.5],
        'memory': [3.8],
        'attention_to_detail': [4.2],
        'teamwork': [3.7],
        'leadership': [3.2],
        'stress_tolerance': [3.5],
        'adaptability': [4.0],
        'curiosity': [4.3],
        'math_interest': [4.5],
        'physics_interest': [3.0],
        'chemistry_interest': [2.5],
        'biology_interest': [2.0],
        'it_interest': [4.2],
        'economics_interest': [3.0],
        'humanities_interest': [2.5],
        'art_interest': [2.0],
        'sports_interest': [3.5],
        'desired_salary': [150000],
        'preferred_work_env': [2],
        'work_life_balance': [3.8]
    }
    
    df_template = pd.DataFrame(template_data)
    
    filename = "student_profile_template.csv"
    df_template.to_csv(filename, index=False)
    
    print(f"\nШаблон создан: {filename}")
    print("\nСТРУКТУРА ФАЙЛА:")
    print("-"*40)
    print(df_template.to_string())
    
    print(f"\nИНСТРУКЦИЯ:")
    print("1. Измените значения в файле {filename}")
    print("2. Шкала оценок: 1.0 (низкий) - 5.0 (высокий)")
    print("3. desired_salary: желаемая зарплата в рублях")
    print("4. preferred_work_env: 1=офис, 2=гибрид, 3=удаленка")
    print("5. work_life_balance: 1.0-5.0 (важность баланса)")
    print(f"\nЗапустите: python career_ai.py {filename}")

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def main():
    parser = argparse.ArgumentParser(description='ИИ для выбора направления обучения')
    parser.add_argument('csv_file', nargs='?', help='CSV файл с профилем студента')
    parser.add_argument('--create-template', action='store_true', 
                       help='Создать шаблон CSV для тестирования')
    parser.add_argument('--student-name', type=str, default=None,
                       help='Имя студента для отчета')
    
    args = parser.parse_args()
    
    if args.create_template:
        create_template_csv()
        return
    
    if not args.csv_file:
        print("ОШИБКА: Укажите CSV файл с профилем студента")
        print("Использование: python career_ai.py student_profile.csv")
        print("Или создайте шаблон: python career_ai.py --create-template")
        sys.exit(1)
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"ОШИБКА: Файл не найден: {args.csv_file}")
        print(f"Создайте шаблон: python career_ai.py --create-template")
        sys.exit(1)
    
    print("\nЗАПУСК ИСКУССТВЕННОГО ИНТЕЛЛЕКТА")
    print("="*80)
    
    advisor = CareerAdvisorAI()
    advisor.load_model()
    
    interface = CareerAdvisorInterface(advisor)
    
    try:
        recommendations, student_data, all_recommendations, predictions = advisor.analyze_student_profile(args.csv_file)
        
        interface.print_all_recommendations(recommendations, student_data, args.csv_file)
        
        print(f"\nАНАЛИЗ ЗАВЕРШЕН! Удачи в выборе профессии!")
        
        # Создаем CSV файл ТОЛЬКО с топ-5 рекомендациями
        csv_file = create_top5_recommendations_csv(all_recommendations, csv_path)
        
        print(f"\nРезультаты сохранены в файле: {csv_file}")
        
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        print("Проверьте формат CSV файла или создайте новый шаблон:")
        print("python career_ai.py --create-template")

if __name__ == "__main__":
    main()