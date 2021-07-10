# sort imports
isort --quiet . pytorch_widedeep tests examples setup.py
# Black code style
black . pytorch_widedeep tests examples setup.py
# flake8 standards
flake8 . --max-complexity=10 --max-line-length=127 --ignore=E203,E266,E501,E722,F401,F403,F405,W503,C901,F811
# mypy
mypy pytorch_widedeep --ignore-missing-imports --no-strict-optional