# Career AI — профориентация (Университет «Дубна»)

Веб-система рекомендует **направления подготовки бакалавриата** [Государственного университета «Дубна»](https://uni-dubna.ru) по профилю навыков и интересов. Используется нейросеть (multilabel) по **19 направлениям**.

**Источник направлений:** [каталог специальностей](https://dubna.postupi.online/vuz/universitet-dubna/specialnosti/bakalavr/)  
**Сайт вуза:** [uni-dubna.ru](https://uni-dubna.ru) | [Абитуриентам](https://uni-dubna.ru/abitur)

---

## Возможности

- Адаптивный веб-опросник (FastAPI)
- Топ рекомендаций с объяснениями и кодами ФГОС
- CLI: рекомендации по CSV-профилю
- Анонимный сбор ответов в `data/real/responses.csv` (для будущего дообучения)
- Запуск через Docker или локально

---

## Быстрый старт

### Docker (рекомендуется)

Нужны [Docker](https://www.docker.com/) и Docker Compose.

```bash
docker compose up --build
```

Откройте http://localhost:7777

При **первом** запуске контейнер обучит модель (5–15 минут), затем поднимет сайт. Модель и анкеты сохраняются в томах `career_model` и `career_data`.

Остановка:

```bash
docker compose down
```

---

### Локально

```bash
pip install -r requirements.txt
python train_model.py          # один раз: обучение модели
python main.py                 # веб: http://127.0.0.1:7777
```

#### Рекомендации по CSV

```bash
python career_ai.py examples/student_it_profile.csv
python career_ai.py examples/student_it_profile.csv --detailed --output-dir output
```

#### Консольный опросник

```bash
python ask_questions.py
python career_ai.py profiles/student_YYYYMMDD_HHMMSS.csv
```

---

## Формат CSV (27 признаков)

Одна строка данных. Оценки — **1.0–5.0**, зарплата — в рублях.

- Шаблон: `examples/student_template.csv`
- Пример IT-профиля: `examples/student_it_profile.csv`

| Признак | Описание |
|--------|----------|
| `math_modeling` … `linguistics_skill` | Предметные навыки и интересы (1–5) |
| `ecology`, `electronics`, `management`, `public_administration` | Доп. области |
| `logical_thinking` … `teamwork_skill` | Общие качества (1–5) |
| `desired_salary` | Желаемая зарплата (руб.) |

Недостающие колонки заполняются значениями по умолчанию (3.0 / 100000).  
Полный список — в `config/model_config.py` (`FEATURE_LABELS`).

---

## Направления (19)

Список и коды ФГОС — в `config/model_config.py` (`DUBNA_DIRECTION_KEYS`, `DUBNA_CODES`).

Примеры: Программная инженерия (09.03.04), Физика (03.03.02), Ядерные физика и технологии (14.03.02), Психология (37.03.01), Менеджмент, Юриспруденция, Лингвистика и др.

---

## Структура проекта

```
career-ai-system/
├── config/
│   ├── model_config.py      # Направления, признаки (единый источник)
│   └── dubna_directions.py  # Карточки направлений
├── career_core.py           # Модель, предсказания, объяснения
├── train_model.py           # Обучение
├── career_ai.py             # CLI по CSV
├── ask_questions.py         # Консольный опросник
├── main.py                  # FastAPI + веб-UI
├── data_collector.py        # Сохранение ответов в CSV
├── templates/index.html
├── static/                  # Логотип и статика
├── examples/
├── docker-compose.yml
├── Dockerfile
├── tests/test_smoke.py
└── requirements.txt
```

Папки `career_model/` и `data/` создаются при работе и в Git не коммитятся.

---

## Сбор реальных данных

После прохождения опроса на сайте:

1. В `data/real/responses.csv` добавляется строка: 27 признаков + топ-5 рекомендаций (без ФИО и контактов).
2. Если пользователь отправил отзыв — в ту же строку дописываются `chosen_fields` и `feedback_score`.

---

## Тесты

```bash
pytest tests/ -v
```

Без обученной модели часть тестов пропускается.

---

## Требования

- Python 3.8+
- TensorFlow ≥ 2.10, pandas, scikit-learn, FastAPI, uvicorn (см. `requirements.txt`)

---

## Важно

Данные для обучения **синтетические** — модель отражает заложенные правила «навык → направление». Для реального применения нужна валидация на анкетах абитуриентов.
