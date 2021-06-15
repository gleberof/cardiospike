# Cardio Spike
https://leadersofdigital.ru/cabinet/63008/hackathon#your-case
# Структура репозитория

# Установка

## Установка Python нужной версии

Установить [pyenv](https://pipenv-fork.readthedocs.io/en/latest/install.html#installing-pipenv)
Следуйте инструкциям (добавить переменные в `~/.bashrc`/`~/.zshrc`)

```bash
pyenv install 3.8.6
```

## Установка проекта
Установить [poetry](https://python-poetry.org/)
```bash
poetry install
poetry run pre-commit install
```

Установите `nvidia-container-toolkit` для работы с Docker и GPU
