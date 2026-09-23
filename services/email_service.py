import logging
import smtplib
from email.message import EmailMessage

logger = logging.getLogger(__name__)


class EmailService:
    def __init__(self, user: str, app_password: str):
        self.user = user
        self.app_password = app_password

    @property
    def configured(self):
        return bool(self.user and self.app_password)

    def send(self, recipients, subject, body):
        if not self.configured:
            raise RuntimeError("Email is not configured")
        message = EmailMessage()
        message["From"] = self.user
        message["To"] = ", ".join(recipients)
        message["Subject"] = subject
        message.set_content(body)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as smtp:
            smtp.login(self.user, self.app_password)
            smtp.send_message(message)

    def send_safely(self, recipients, subject, body):
        try:
            self.send(recipients, subject, body)
            return True
        except Exception:
            logger.exception("Email notification failed for recipients=%s", recipients)
            return False
