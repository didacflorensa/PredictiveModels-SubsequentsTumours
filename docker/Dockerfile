FROM python:3
COPY ./python-analysis/requeriments.txt /tmp/requeriments.txt
RUN  pip install --no-cache-dir -r  /tmp/requeriments.txt

# create a python user
ENV USER python-user

## Copy your working files over
COPY ./results /home/$USER/results
COPY ./data /home/$USER/Data
COPY ./python /home/$USER/python-analysis


WORKDIR /home/$USER/python-analysis

CMD [ "python" "./your-daemon-or-script.py"]
