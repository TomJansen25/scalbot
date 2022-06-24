.PHONY: generate-requirements
generate-requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

.PHONY: python-format
python-format:
	poetry run isort .
	poetry run black .
	poetry run mypy scalbot/

.PHONY: gcp-deploy-bot-dev
gcp-deploy-bot-dev:
	$TOPICNAME = "scalbot-bot-topic-dev"
	$JOBNAME = "scalbot-minute-job-dev"
	gcloud scheduler jobs pause $JOBNAME
	gcloud functions deploy scalbot-init-func --region europe-west3 --entry-point run_bybit_bot --runtime python39 --trigger-topic $TOPICNAME --memory 512MB
	gcloud scheduler jobs resume $JOBNAME --location europe-west3


.PHONY: gcp-deploy-bot-prod
gcp-deploy-bot-prod:
	$TOPICNAME = "scalbot-bot-topic-prod"
	$JOBNAME = "scalbot-minute-job-prod"
	gcloud scheduler jobs pause $JOBNAME
	gcloud functions deploy scalbot-run-bot-prod --region europe-west3 --entry-point run_bybit_bot --runtime python39 --trigger-topic $TOPICNAME --memory 512MB
	gcloud scheduler jobs resume $JOBNAME --location europe-west3
