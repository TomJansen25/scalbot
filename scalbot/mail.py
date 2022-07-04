import os
import re
import smtplib
import ssl
from abc import ABC
from email.message import EmailMessage
from string import Template
from typing import Literal, Optional, Union

from loguru import logger
from pydantic import BaseModel, validator

from scalbot.utils import get_project_dir, setup_logging

setup_logging()

VALID_EMAIL_REGEX = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"


def email_address_validator(email_address: str) -> str:
    """

    :param email_address:
    :return:
    """
    assert re.match(VALID_EMAIL_REGEX, email_address)
    return email_address


class Email(ABC, BaseModel):
    email_sender: str
    email_password: str
    email_receiver: Union[str, list[str]]
    message_template: Optional[str] = None
    message_content: Optional[str] = None
    email_message: Optional[EmailMessage] = None

    class Config:
        arbitrary_types_allowed = True

    _validate_email_sender = validator("email_sender", allow_reuse=True)(
        email_address_validator
    )
    _validate_email_receiver = validator(
        "email_receiver", allow_reuse=True, each_item=True
    )(email_address_validator)

    def __init__(self, email_sender: str, email_receiver: Union[str, list[str]]):
        if email_sender.endswith("@gmail.com"):
            email_password = os.getenv("GMAIL_PASSWORD")
        else:
            email_password = os.getenv("EMAIL_PASSWORD")

        super().__init__(
            email_sender=email_sender,
            email_password=email_password,
            email_receiver=email_receiver,
        )

    def set_message_template(self, template: Literal["daily_trade_summary", ""]):
        if template == "daily_trade_summary":
            file_path = get_project_dir().joinpath("config", "daily_summary_mail.html")
        else:
            err = (
                f"Provided Template ({template}) is not supported yet... Please choose on of "
                f"the following: 'daily_trade_summary'."
            )
            logger.error(err)
            raise KeyError(err)

        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        self.message_template = html_content

    def fill_message_template(
        self, variable_substitutes: dict[str, Union[str, int, float]]
    ):

        if not self.message_template:
            err = "Message Template is currently empty, first load in a template before filling it with variables"
            logger.error(err)
            raise KeyError(err)

        filled_template = Template(self.message_template).safe_substitute(
            variable_substitutes
        )
        self.message_content = filled_template

    def prepare_message(self, subject: str, message_type: Literal["html", ""]):
        if not self.email_sender or not self.email_receiver or not self.message_content:
            err = f"Not all required Class variables are set, message can not be prepared..."
            logger.error(err)
            raise KeyError(err)

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.email_sender
        msg["To"] = self.email_receiver
        msg.set_content(self.message_content, message_type)
        self.email_message = msg

    def send_email(self):
        """ """
        if self.email_sender.endswith("@gmail.com"):
            ssl_context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl_context) as smtp:
                smtp.login(self.email_sender, self.email_password)
                smtp.send_message(self.email_message)
        else:
            raise ValueError(f"Only Gmail accounts accepted for now...")

        logger.info("Email successfully sent!")
