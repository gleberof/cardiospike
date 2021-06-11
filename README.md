# Cardio Spike
https://leadersofdigital.ru/cabinet/63008/hackathon#your-case
# Структура репозитория


# Установка проекта
[Установка pyenv](https://pipenv-fork.readthedocs.io/en/latest/install.html#installing-pipenv)

Следуйте инструкциям (добавить переменные в bashrc/zshrc)

Установка нужной версии python в папке проекта
```bash
pyenv install 3.8.6
cd cardiohack
pyenv local 3.8.6
```

Установка poetry
```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
poetry install
poetry shell
pre-commit install
```
