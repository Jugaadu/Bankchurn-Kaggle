

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
train:
	sh run.sh ${model}
test:
	python -m pytest -vv -cov src.test_train.py


format:
	black src/*.py

lint:
	pylint --disable=R,C src.train.py

all: 
