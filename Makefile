.PHONY: python-format
python-format:
	poetry run isort .
	poetry run black .
	poetry run mypy scalbot/

.PHONY: deploy-gcp-dev
deploy-gcp-dev:
	$TOPICNAME = "scalbot-topic-123"
	$JOBNAME = "scalbot-job-123"
	gcloud scheduler jobs pause $JOBNAME
	gcloud functions deploy scalbot-init-func --region europe-west3 --entry-point scalbot_init_func --runtime python39 --trigger-topic $TOPICNAME
	gcloud scheduler jobs resume $JOBNAME --location europe-west3