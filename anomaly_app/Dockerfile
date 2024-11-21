# Get python 3 image from dockerhub
FROM python:3

RUN wget -O /usr/local/bin/dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.5/dumb-init_1.2.5_x86_64
RUN chmod +x /usr/local/bin/dumb-init

# Copy repo contents into working directory
COPY . /anomaly_app
COPY ./requirements.txt /anomaly_app/requirements.txt
WORKDIR /anomaly_app

# Install prometheus client library
RUN pip install prometheus_client
RUN pip install -r requirements.txt

# Dumb init
ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]

# Run the application
CMD ["python", "anomaly_app.py"]
