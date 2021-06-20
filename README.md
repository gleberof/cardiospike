# CardioSpike

Решение кейса [CardioSpike](https://leadersofdigital.ru/event/63008/case/706486) хакатона [Цифровой Прорыв](https://leadersofdigital.ru/) команды "Звездочка", занявшее первое место.

Задача кейса состояла в детектировании аномалий в сердечных RR-ритмах, специфичных для больных COVID-19.

Пример детектирования аномалий моделью:

![Пример детектирования аномалий моделью](img/rr.png)

Состав команды: [Глеб Ерофеев](https://github.com/gleberof), [Даниил Гафни](https://github.com/danielgafni), [Сергей Фиронов](https://github.com/ifserge), [Михаил Марьин](https://github.com/muxaulmarin).

## Демонстрация
[Веб-приложение](http://сердечный-друг.рф/)

[Онлайн-документация REST API](http://сердечный-друг.рф:5000)

## Реализована функциональность
 - Разработана модель машинного обучения, позволяющая производить детектирование аномалий в RR-ритмах больных COVID-2019
 - Модель представляет собой связку градиентных бустингой и нейронных сетей, предсказания которых совмещаются при помощи линейной модели.
 - Реализована распределенная оптимизация гиперпараметров для нейронных сетей
 - Реализован REST API для детектирования аномалий
 - Реализовано веб-приложение для взаимодействия с REST API

## Особенности проекта
 - Нейронная сеть включает в себя SOTA технологию Attention
 - Система распределенного подбора гиперпараметров нейронной сети способна горизонтально расширяться практички неограниченно
 - Система обучения с использованием двойной вложенной кросс-валидации обеспечивает стабильность модели и способность корректно обобщать данные
 - Присутствуют легкие, но качественные модели LightGBM
 - Совмещение предсказания различных моделей обеспечивает дополнительную стабильность результата

## Осной стек технологий:
Python, Optuna, Hydra, Pytorch, Pytorch-Lightning, Sklearn, Pandas, Numpy, Plotly, Flask, FastAPI, Poetry


### Локальный запуск веб-приложения
Скопировать `.env.example` в `.env`:
```bash
cp .env.example .env
```
Поднять приложение:
```bash
docker-compose up
```
Оно станет доступным в браузере (по умолчанию - `localhost:8000`).

## Установка

### Установка Python нужной версии

Установить [pyenv](https://pipenv-fork.readthedocs.io/en/latest/install.html#installing-pipenv)
Следуйте инструкциям (добавить переменные в `~/.bashrc`/`~/.zshrc`)

```bash
pyenv install 3.8.6
```

### Установка проекта
Установить [poetry](https://python-poetry.org/)
```bash
poetry install
poetry run pre-commit install  # для разработки
```
Скопировать `.env.example` в `.env`:
```bash
cp .env.example .env
```
