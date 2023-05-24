FROM python:3.9

WORKDIR /deepmrm 

COPY . /deepmrm

RUN python setup.py install

RUN pip install --no-cache-dir --upgrade -r requirements.txt

ENTRYPOINT ["python", "./deepmrm/predict/make_prediction.py"]
