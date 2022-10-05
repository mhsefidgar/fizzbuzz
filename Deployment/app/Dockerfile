FROM python:3.8-slim-buster

WORKDIR /app

RUN apt-get update
RUN apt install -y libgl1-mesa-glx



ADD . /app
RUN pip install --default-timeout=100 future

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

ARG USER_NAME=app
ARG USER_UID=1000
ARG PASSWD=password

RUN useradd -m -s /bin/bash -u $USER_UID $USER_NAME && \
    gpasswd -a $USER_NAME sudo && \
    echo "${USER_NAME}:${PASSWD}" | chpasswd && \
    echo "${USER_NAME} ALL=(ALL) ALL" >> /etc/sudoers

COPY ./ /app
RUN chown -R ${USER_NAME}:${USER_NAME} /app


EXPOSE 8000

# Execute
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0" ,"--port" ,"8000"]