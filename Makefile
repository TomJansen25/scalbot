BOT_TOPICNAME=scalbot-bot-topic
BOT_JOBNAME=scalbot-bot-job
BOT_FUNCTIONNAME=scalbot-bot-function

MAILER_TOPICNAME=scalbot-mailer-topic
MAILER_JOBNAME=scalbot-mailer-job
MAILER_FUNCTIONNAME=scalbot-mailer-function

INVALID_CHECK_TOPICNAME=scalbot-invalid-check-topic
INVALID_CHECK_JOBNAME=scalbot-invalid-check-job
INVALID_CHECK_FUNCTIONNAME=scalbot-invalid-check-function


###-------------------- GENERAL CODE STUFF --------------------###

.PHONY: generate-requirements
generate-requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

.PHONY: python-format
python-format:
	poetry run isort .
	poetry run black .
	poetry run mypy scalbot/

###-------------------- CREATE INITIAL GCP THINGS --------------------###

.PHONY: gcp-create-dev
gcp-create-dev:
	gcloud pubsub topics create dev-$(BOT_TOPICNAME)
	gcloud pubsub topics create dev-$(MAILER_TOPICNAME)
	gcloud pubsub topics create dev-$(INVALID_CHECK_TOPICNAME)
	gcloud scheduler jobs create pubsub dev-$(BOT_JOBNAME) --schedule "* * * * *" --topic dev-$(BOT_TOPICNAME) --message-body "Scalbot" --location europe-west3
	gcloud scheduler jobs create pubsub dev-$(MAILER_JOBNAME) --schedule "30 23 * * *" --topic dev-$(MAILER_TOPICNAME) --message-body "Mailer" --location europe-west3
	gcloud scheduler jobs create pubsub dev-$(INVALID_CHECK_JOBNAME) --schedule "0 */12 * * *" --topic dev-$(INVALID_CHECK_TOPICNAME) --message-body "Invalid Check" --location europe-west3
	gcloud functions deploy dev-$(BOT_FUNCTIONNAME) --region europe-west3 --entry-point run_bybit_bot --runtime python39 --trigger-topic dev-$(BOT_TOPICNAME) --memory 512MB
	gcloud functions deploy dev-$(MAILER_FUNCTIONNAME) --region europe-west3 --entry-point send_daily_summary --runtime python39 --trigger-topic dev-$(MAILER_TOPICNAME) --memory 512MB

###-------------------- DEVELOPMENT DEPLOYMENTS --------------------###

.PHONY: gcp-deploy-bot-dev
gcp-deploy-bot-dev:
	gcloud scheduler jobs pause dev-$(BOT_JOBNAME) --location europe-west3
	gcloud functions deploy dev-$(BOT_FUNCTIONNAME) --region europe-west3 --entry-point run_bybit_bot --runtime python39 --trigger-topic dev-$(BOT_TOPICNAME) --memory 512MB
	gcloud scheduler jobs resume dev-$(BOT_JOBNAME) --location europe-west3

.PHONY: gcp-deploy-invalid-check-dev
gcp-deploy-invalid-check-dev:
	gcloud functions deploy dev-$(INVALID_CHECK_FUNCTIONNAME) --region europe-west3 --entry-point cancel_invalid_or_expired_orders --runtime python39 --trigger-topic dev-$(INVALID_CHECK_TOPICNAME) --memory 512MB

.PHONY: gcp-deploy-mailer-dev
gcp-deploy-mailer-dev:
	gcloud functions deploy dev-$(MAILER_FUNCTIONNAME) --region europe-west3 --entry-point send_daily_summary --runtime python39 --trigger-topic dev-$(MAILER_TOPICNAME) --memory 512MB

###-------------------- PRODUCTION DEPLOYMENTS --------------------###

.PHONY: gcp-deploy-bot-prod
gcp-deploy-bot-prod:
	gcloud scheduler jobs pause prod-$(BOT_JOBNAME) --location europe-west3
	gcloud functions deploy prod-$(BOT_FUNCTIONNAME) --region europe-west3 --entry-point run_bybit_bot --runtime python39 --trigger-topic prod-$(BOT_TOPICNAME) --memory 512MB
	gcloud scheduler jobs resume prod-$(BOT_JOBNAME) --location europe-west3

.PHONY: gcp-deploy-invalid-check-prod
gcp-deploy-invalid-check-prod:
	gcloud functions deploy prod-$(INVALID_CHECK_FUNCTIONNAME) --region europe-west3 --entry-point cancel_invalid_or_expired_orders --runtime python39 --trigger-topic prod-$(INVALID_CHECK_TOPICNAME) --memory 512MB

.PHONY: gcp-deploy-mailer-prod
gcp-deploy-mailer-prod:
	gcloud functions deploy prod-$(MAILER_FUNCTIONNAME) --region europe-west3 --entry-point send_daily_summary --runtime python39 --trigger-topic prod-$(MAILER_TOPICNAME) --memory 512MB
