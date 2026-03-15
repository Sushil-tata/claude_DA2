"""
Alerting Module - Multi-Channel Notifications

Sends alerts to multiple channels:
- Slack (webhooks)
- PagerDuty (API)
- Email (SMTP)

Alert Severity Levels:
- INFO: Daily summaries, successful runs
- WARNING: Moderate drift (PSI 0.1-0.25), minor issues
- CRITICAL: Severe drift (PSI > 0.25), model failures, data quality issues

Usage:
    alerter = AlertManager(config)
    alerter.send_alert(
        severity="CRITICAL",
        title="Feature Drift Detected",
        message="PSI = 0.35 for transaction_amount",
        details={...}
    )
"""

import logging
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class SlackAlerter:
    """
    Send alerts to Slack via webhook.
    """

    def __init__(self, webhook_url: str):
        """
        Initialize Slack alerter.

        Args:
            webhook_url: Slack webhook URL
        """
        self.webhook_url = webhook_url

    def send(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send alert to Slack.

        Args:
            severity: Alert severity level
            title: Alert title
            message: Alert message
            details: Additional details (optional)

        Returns:
            True if successful
        """
        # Emoji based on severity
        emoji_map = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🔴"
        }
        emoji = emoji_map.get(severity, "📢")

        # Color based on severity
        color_map = {
            AlertSeverity.INFO: "#36a64f",  # Green
            AlertSeverity.WARNING: "#ff9900",  # Orange
            AlertSeverity.CRITICAL: "#ff0000"  # Red
        }
        color = color_map.get(severity, "#808080")

        # Build message
        text = f"{emoji} *{title}*"

        # Build attachment with details
        attachment = {
            "color": color,
            "title": title,
            "text": message,
            "footer": "Decision Agent Monitoring",
            "ts": int(datetime.now().timestamp())
        }

        if details:
            fields = []
            for key, value in details.items():
                fields.append({
                    "title": key,
                    "value": str(value),
                    "short": True
                })
            attachment["fields"] = fields

        payload = {
            "text": text,
            "attachments": [attachment]
        }

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"✓ Slack alert sent: {title}")
                return True
            else:
                logger.error(f"Slack alert failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class PagerDutyAlerter:
    """
    Send alerts to PagerDuty via Events API v2.
    """

    def __init__(self, integration_key: str):
        """
        Initialize PagerDuty alerter.

        Args:
            integration_key: PagerDuty integration key
        """
        self.integration_key = integration_key
        self.events_url = "https://events.pagerduty.com/v2/enqueue"

    def send(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send alert to PagerDuty.

        Args:
            severity: Alert severity level
            title: Alert title
            message: Alert message
            details: Additional details

        Returns:
            True if successful
        """
        # Map severity to PagerDuty severity
        severity_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.CRITICAL: "critical"
        }
        pd_severity = severity_map.get(severity, "error")

        # Build event
        event = {
            "routing_key": self.integration_key,
            "event_action": "trigger",
            "payload": {
                "summary": title,
                "severity": pd_severity,
                "source": "decision_agent_monitoring",
                "timestamp": datetime.now().isoformat(),
                "custom_details": {
                    "message": message,
                    **(details or {})
                }
            }
        }

        try:
            response = requests.post(
                self.events_url,
                json=event,
                timeout=10
            )

            if response.status_code == 202:
                logger.info(f"✓ PagerDuty alert sent: {title}")
                return True
            else:
                logger.error(f"PagerDuty alert failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False


class EmailAlerter:
    """
    Send alerts via email (SMTP).
    """

    def __init__(self, smtp_config: Dict[str, Any]):
        """
        Initialize email alerter.

        Args:
            smtp_config: SMTP configuration:
                - host: SMTP server host
                - port: SMTP server port
                - username: SMTP username
                - password: SMTP password
                - from_address: Sender email address
        """
        self.smtp_config = smtp_config

    def send(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        recipients: Optional[List[str]] = None
    ) -> bool:
        """
        Send alert via email.

        Args:
            severity: Alert severity level
            title: Alert title
            message: Alert message
            details: Additional details
            recipients: Email recipients

        Returns:
            True if successful
        """
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        if not recipients:
            logger.warning("No email recipients specified. Skipping email alert.")
            return False

        try:
            # Build email
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{severity.value.upper()}] {title}"
            msg["From"] = self.smtp_config["from_address"]
            msg["To"] = ", ".join(recipients)

            # Build email body
            body = f"""
Alert: {title}
Severity: {severity.value.upper()}
Time: {datetime.now().isoformat()}

Message:
{message}
"""

            if details:
                body += "\n\nDetails:\n"
                for key, value in details.items():
                    body += f"  {key}: {value}\n"

            body += "\n---\nDecision Agent Monitoring System"

            # Attach body
            msg.attach(MIMEText(body, "plain"))

            # Send email
            with smtplib.SMTP(self.smtp_config["host"], self.smtp_config["port"]) as server:
                if self.smtp_config.get("use_tls", True):
                    server.starttls()

                if self.smtp_config.get("username") and self.smtp_config.get("password"):
                    server.login(
                        self.smtp_config["username"],
                        self.smtp_config["password"]
                    )

                server.send_message(msg)

            logger.info(f"✓ Email alert sent to {len(recipients)} recipients")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class AlertManager:
    """
    Orchestrates alerts across multiple channels.

    Routes alerts based on severity:
    - INFO → Slack only
    - WARNING → Slack + Email
    - CRITICAL → Slack + PagerDuty + Email
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert manager.

        Args:
            config: Alerting configuration:
                - slack_webhook: Slack webhook URL (optional)
                - pagerduty_integration_key: PagerDuty key (optional)
                - email: Email config (optional)
                - email_recipients: List of email addresses
                - severity_routing: Custom severity → channels mapping
        """
        self.config = config

        # Initialize alerters
        self.slack = None
        if config.get("slack_webhook"):
            self.slack = SlackAlerter(config["slack_webhook"])

        self.pagerduty = None
        if config.get("pagerduty_integration_key"):
            self.pagerduty = PagerDutyAlerter(config["pagerduty_integration_key"])

        self.email = None
        if config.get("email"):
            self.email = EmailAlerter(config["email"])

        self.email_recipients = config.get("email_recipients", [])

        logger.info("AlertManager initialized:")
        logger.info(f"  Slack: {'enabled' if self.slack else 'disabled'}")
        logger.info(f"  PagerDuty: {'enabled' if self.pagerduty else 'disabled'}")
        logger.info(f"  Email: {'enabled' if self.email else 'disabled'}")

    def send_alert(
        self,
        severity: str,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """
        Send alert to appropriate channels based on severity.

        Args:
            severity: Alert severity ("info", "warning", "critical")
            title: Alert title
            message: Alert message
            details: Additional details

        Returns:
            Dict of {channel: success} results
        """
        severity_enum = AlertSeverity(severity.lower())

        logger.info(f"Sending {severity_enum.value.upper()} alert: {title}")

        results = {}

        # Route based on severity
        if severity_enum == AlertSeverity.INFO:
            # INFO → Slack only
            if self.slack:
                results["slack"] = self.slack.send(severity_enum, title, message, details)

        elif severity_enum == AlertSeverity.WARNING:
            # WARNING → Slack + Email
            if self.slack:
                results["slack"] = self.slack.send(severity_enum, title, message, details)

            if self.email and self.email_recipients:
                results["email"] = self.email.send(
                    severity_enum, title, message, details, self.email_recipients
                )

        elif severity_enum == AlertSeverity.CRITICAL:
            # CRITICAL → Slack + PagerDuty + Email
            if self.slack:
                results["slack"] = self.slack.send(severity_enum, title, message, details)

            if self.pagerduty:
                results["pagerduty"] = self.pagerduty.send(severity_enum, title, message, details)

            if self.email and self.email_recipients:
                results["email"] = self.email.send(
                    severity_enum, title, message, details, self.email_recipients
                )

        logger.info(f"Alert results: {results}")
        return results
