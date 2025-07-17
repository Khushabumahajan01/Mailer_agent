import imaplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def save_draft_imap(imap_host, imap_port, username, password, to_addr, subject, body):
    # Build the MIME email
    msg = MIMEMultipart()
    msg["From"] = username
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Connect to IMAP
    mail = imaplib.IMAP4_SSL(imap_host, imap_port)
    mail.login(username, password)

    # For Gmail, use "[Gmail]/Drafts". For others, use "Drafts"
    drafts_folder = '"[Gmail]/Drafts"' if "gmail" in imap_host else '"Drafts"'

    # Append to Drafts
    mail.append(
        drafts_folder,
        "",
        imaplib.Time2Internaldate(time.time()),
        msg.as_bytes()
    )
    mail.logout()