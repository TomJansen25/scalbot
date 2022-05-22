# Scalbot
A Scalping Bot implemented in Python.

## Getting Started

### Dependencies and installing

* PC with [Poetry](https://python-poetry.org/) installed
* Run:
```shell
git clone https://github.com/TomJansen25/scalbot.git
cd scalbot
poetry install --no-dev
```

### Executing program

* Fill in required environment variables like in `.env.example` in a `.env` file
* Run the following to run the example script to run a bot:
```shell
poetry shell
python run_bot.py
```

## Deploy on GCP
```shell
gcloud pubsub topics create TOPICNAME
gcloud scheduler jobs create pubsub JOBNAME --schedule "* * * * *" --topic TOPICNAME --message-body "EARN MONEY" --location europe-west3
gcloud functions deploy FUNCTIONNAME --entry-point execute-me --region europe-west3 --runtime python39 --trigger-topic TOPICNAME
```

## Authors

Contributors names and contact info

- [Tom Jansen](https://github.com/TomJansen25)
