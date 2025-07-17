import imaplib
import email
from email.utils import parseaddr
from bs4 import BeautifulSoup  
from datetime import datetime, timedelta
from typing import List
import re


SYSTEM_EMAIL_PATTERNS = [
    r"^no[-_]?reply@", r"^mailer-daemon@", r"^postmaster@", r"^bounce@",
    r"^noreply@", r"^automated@", r"^do[-_]?not[-_]?reply@"
]

def fetch_emails(imap_config) -> List[dict]:
    try:
        mail = imaplib.IMAP4_SSL(imap_config.host, imap_config.port)
        mail.login(imap_config.username, imap_config.password)
    except imaplib.IMAP4.error as e:
        print(f"IMAP login failed: {e}")
        raise

    emails = []
    unread = getattr(imap_config.filters, 'unread', False)

    # Calculate SINCE date for last 60 days
    since_days = getattr(imap_config.filters, 'since_days', 60)
    since_date = (datetime.now() - timedelta(days=since_days)).strftime("%d-%b-%Y")

    for folder in imap_config.folders:
        print(f"Selecting folder: {folder}")
        status, _ = mail.select(folder)
        if status != 'OK':
            print(f"Failed to select folder: {folder}")
            continue

        search_criteria = []
        search_criteria.append(f'SINCE {since_date}')
        if unread:
            search_criteria.insert(0, 'UNSEEN')
        else:
            search_criteria.insert(0, 'ALL')

        print(f"Search criteria: {search_criteria}")
        result, data = mail.search(None, *search_criteria)
        print(f"Search result: {result}, data: {data}")

        if result != 'OK' or not data or not data[0]:
            print(f"No emails found in folder {folder} with criteria {search_criteria}")
            continue

        email_ids = data[0].split()
        for num in email_ids:
            # Fetch FLAGS to determine read/unread status
            _, flags_data = mail.fetch(num, '(FLAGS)')
            flags = flags_data[0].decode() if flags_data and flags_data[0] else ""
            is_unread = '\\Seen' not in flags

            _, msg_data = mail.fetch(num, '(RFC822)')
            msg = email.message_from_bytes(msg_data[0][1])
            subject = decode_mime_header(msg.get("Subject", ""))
            from_raw = msg.get("From", "")
            from_name, from_email = parseaddr(from_raw)

            # Extract "To", "Cc", and "Bcc" addresses
            to_raw = msg.get("To", "")
            cc_raw = msg.get("Cc", "")
            bcc_raw = msg.get("Bcc", "")

            from email.utils import getaddresses
            to_list = [addr for name, addr in getaddresses([to_raw])] if to_raw else []
            cc_list = [addr for name, addr in getaddresses([cc_raw])] if cc_raw else []
            bcc_list = [addr for name, addr in getaddresses([bcc_raw])] if bcc_raw else []

            if is_system_email(from_email):
                print(f"Skipping system email: {from_email}")
                continue

            plain_body, html_body = extract_both_contents(msg)
            # Parse the date for sorting and display
            date_tuple = email.utils.parsedate_tz(msg.get("Date"))
            timestamp = email.utils.mktime_tz(date_tuple) if date_tuple else 0
            dt = datetime.fromtimestamp(timestamp)
            date_str = f"{dt.strftime('%a')} {dt.month}/{dt.day}/{dt.year} {dt.strftime('%I:%M %p')}" if timestamp else ""

            emails.append({
                "from": {
                    "name": decode_mime_header(from_name) if from_name else "",
                    "email": from_email or ""
                },
                "to": to_list or [],
                "cc": cc_list or [],
                "bcc": bcc_list or [],
                "subject": subject or "",
                "body": {
                    "plain": plain_body or "",
                    "html": html_body or ""
                },
                "folder": folder,
                "date": date_str or "",
                "unread": is_unread
            })

            if unread:
                mail.store(num, '+FLAGS', '\\Seen')

    mail.logout()
    # Sort emails by timestamp descending (latest first)
    emails.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    # Remove timestamp from output if not needed
    for email_obj in emails:
        email_obj.pop("timestamp", None)
    return emails


def extract_main_content(msg):
    plain_text = None
    html_text = None

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))
            if content_type == "text/plain" and "attachment" not in content_disposition:
                plain_text = decode_payload(part)
            elif content_type == "text/html" and "attachment" not in content_disposition:
                html_text = decode_payload(part)
    else:
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            plain_text = decode_payload(msg)
        elif content_type == "text/html":
            html_text = decode_payload(msg)

    if plain_text:
        return plain_text.strip()
    elif html_text:
        return BeautifulSoup(html_text, "html.parser").get_text().strip()
    return ""


def decode_payload(part):
    try:
        payload = part.get_payload(decode=True)
        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, errors='ignore')
    except Exception as e:
        print(f"Failed to decode payload: {e}")
        return ""


def decode_mime_header(value):
    from email.header import decode_header
    if not value:
        return ""
    try:
        decoded_parts = decode_header(value)
        decoded_string = ""
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                decoded_string += part.decode(encoding or "utf-8", errors="ignore")
            else:
                decoded_string += part
        return decoded_string
    except Exception as e:
        print(f"Failed to decode header: {e}")
        return value


def is_system_email(email_address: str) -> bool:
    if not email_address:
        return True
    for pattern in SYSTEM_EMAIL_PATTERNS:
        if re.match(pattern, email_address.lower()):
            return True
    return False


def extract_both_contents(msg):
    plain_text = None
    html_text = None

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))
            if content_type == "text/plain" and "attachment" not in content_disposition:
                plain_text = decode_payload(part)
            elif content_type == "text/html" and "attachment" not in content_disposition:
                html_text = decode_payload(part)
    else:
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            plain_text = decode_payload(msg)
        elif content_type == "text/html":
            html_text = decode_payload(msg)

    return (plain_text.strip() if plain_text else ""), (html_text.strip() if html_text else "")

def group_emails_into_threads(emails):
    # Simple grouping by normalized subject (remove Re:/Fwd:), can be improved
    import re
    threads = {}
    def normalize_subject(subject):
        return re.sub(r"^(re:|fwd:)\s*", "", subject.strip(), flags=re.I)
    for email in emails:
        key = normalize_subject(email.get("subject", ""))
        if key not in threads:
            threads[key] = []
        threads[key].append(email)
    # Sort each thread chronologically (oldest first)
    for thread in threads.values():
        thread.sort(key=lambda x: x.get("date"))
    # Return as list of threads
    return [
        {
            "thread_subject": k,
            "messages": v
        }
        for k, v in threads.items()
    ]
