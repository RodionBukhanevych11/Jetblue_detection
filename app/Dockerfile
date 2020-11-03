FROM python:3.7 
ARG model_parameter_location
ARG model_parameter_name
ARG port
ARG host
ENV JETBLUE_PORT=$port
ENV JETBLUE_HOST=$host
ENV MODEL_WEIGHTS=/app/$model_parameter_name
RUN pip3 install -r requirements.txt
RUN mkdir -p /app
COPY ./jetblue_server.py /app
$model_location/$model_weights_name /app/
./run-model-service.sh /
EXPOSE $port
ENTRYPOINT ["/run-model-service.sh"]