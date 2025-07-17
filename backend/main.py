from fastapi import FastAPI, HTTPException, Body, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from models import EmailConfigPayload, IMAPConfig, IMAPFilters, SMTPConfig
from email_reader import fetch_emails
from email_sender import send_email
from openai_processor import summarize_text, generate_reply, extract_action_item
from imap_draft import save_draft_imap
import requests
import json, os
from rag_utils import extract_text_from_excel, extract_text_from_pdf, chunk_text, get_embedding, build_faiss_index, save_faiss_index, load_faiss_index
import numpy as np
import faiss

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config_store = {}

email_draft_log = []
email_sent_log = []
forwarded_emails_log = []

CONFIG_PATH = "imap_config.json"
KNOWLEDGE_PATH = "knowledge_base.json"
VECTOR_INDEX_PATH = "knowledge_faiss.index"
FORWARDING_RULES_PATH = "forwarding_rules.json"
ACTION_ITEMS_PATH = "email_action_items.json"

knowledge_base = []  # List of {"filename": ..., "content": ...}
vector_index = None
vector_chunks = []
vector_chunk_sources = []  # To map chunk to file and text

forwarding_rules = {
    "keyword_map": [],
    "forward_to": "",
    "forward_to_history": []
}

def save_config():
    if 'imap' in config_store:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config_store['imap'].dict(), f)

def load_config():
    if os.path.exists(CONFIG_PATH):
        from models import IMAPConfig
        with open(CONFIG_PATH) as f:
            config_store['imap'] = IMAPConfig(**json.load(f))

def save_knowledge_base():
    with open(KNOWLEDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

def load_knowledge_base():
    global knowledge_base
    if os.path.exists(KNOWLEDGE_PATH):
        with open(KNOWLEDGE_PATH, encoding="utf-8") as f:
            knowledge_base[:] = json.load(f)

def save_forwarding_rules():
    with open(FORWARDING_RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(forwarding_rules, f, ensure_ascii=False, indent=2)

def load_forwarding_rules():
    global forwarding_rules
    if os.path.exists(FORWARDING_RULES_PATH):
        with open(FORWARDING_RULES_PATH, encoding="utf-8") as f:
            forwarding_rules.update(json.load(f))

def save_action_items(data):
    with open(ACTION_ITEMS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_action_items():
    if os.path.exists(ACTION_ITEMS_PATH):
        with open(ACTION_ITEMS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []

load_knowledge_base()  # Load knowledge base at startup
load_forwarding_rules()

@app.post("/configure-imap")
async def configure_imap(imap: IMAPConfig):
    # Validate port
    if not (1 <= imap.port <= 65535):
        raise HTTPException(status_code=400, detail="Invalid IMAP port.")
    # Store IMAP config in memory
    config_store['imap'] = imap

    # Automatically set SMTP config for Gmail (or your provider)
    smtp_host = "smtp.gmail.com"
    smtp_port = 587
    config_store['smtp'] = SMTPConfig(host=smtp_host, port=smtp_port)

    save_config()

    return {"message": "IMAP (and SMTP) configuration saved successfully."}

@app.post("/configure-smtp")
async def configure_smtp(smtp: SMTPConfig):
    # Validate port
    if not (1 <= smtp.port <= 65535):
        raise HTTPException(status_code=400, detail="Invalid SMTP port.")
    config_store['smtp'] = smtp
    return {"message": "SMTP configuration saved successfully."}

@app.get("/fetch-mails")
async def fetch_mails():
    """
    Fetch emails and return plain content along with a concise LLM-derived action item phrase for each mail.
    """
    imap_config = config_store.get("imap")
    if not imap_config:
        raise HTTPException(status_code=400, detail="IMAP configuration not found. Please configure first.")

    try:
        emails = fetch_emails(imap_config)
        action_items_data = load_action_items()
        existing_keys = set(
            (item["subject"], item["from"].get("email", ""), item["date"])
            for item in action_items_data
        )
        merged_emails = []
        for email in emails:
            key = (email.get("subject", ""), email.get("from", {}).get("email", ""), email.get("date", ""))
            subject = email.get("subject", "")
            plain_body = email.get("body", {}).get("plain", "") if isinstance(email.get("body", ""), dict) else str(email.get("body", ""))
            # Try to find action_item in saved data, else generate
            action_item = None
            for item in action_items_data:
                if (item["subject"], item["from"].get("email", ""), item["date"]) == key:
                    action_item = item.get("action_item")
                    break
            if not action_item:
                action_item = get_action_item_from_email(subject, plain_body)
                email_with_action = {
                    "from": email.get("from"),
                    "to": email.get("to"),
                    "cc": email.get("cc"),
                    "bcc": email.get("bcc"),
                    "subject": subject,
                    "folder": email.get("folder"),
                    "body": plain_body,  # Only plain text!
                    "date": email.get("date"),
                    "unread": email.get("unread"),
                    "action_item": action_item
                }
                action_items_data.append(email_with_action)
            merged_emails.append({
                "from": email.get("from"),
                "to": email.get("to"),
                "cc": email.get("cc"),
                "bcc": email.get("bcc"),
                "subject": subject,
                "folder": email.get("folder"),
                "body": plain_body,
                "date": email.get("date"),
                "unread": email.get("unread"),
                "action_item": action_item
            })
        save_action_items(action_items_data)
        return {"count": len(merged_emails), "emails": merged_emails}
    except Exception as e:
        import traceback
        print("FETCH MAILS ERROR:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fetch-and-draft-emails")
async def fetch_and_draft_emails():
    imap_config = config_store.get("imap")
    if not imap_config:
        raise HTTPException(status_code=400, detail="IMAP configuration not found. Please configure first.")

    try:
        emails = fetch_emails(imap_config)
        results = []

        for email_data in emails:
            body_content = email_data["body"]["plain"] or ""
            if should_forward_email(body_content):
                forward_email(email_data)
            subject = email_data["subject"]
            sender_info = email_data["from"]
            sender_name = sender_info.get("name") or sender_info.get("email")
            from_email = "[Your Email]"  # Replace with actual sender if available
            to_email = sender_info.get("email")  # Reply to the sender
            body_content = email_data["body"]["plain"] or ""

            draft = generate_reply(subject, sender_name, body_content, to_email=to_email, from_email=from_email)

            draft_entry = {
                "from": email_data["from"],
                "to": email_data.get("to", []),
                "cc": email_data.get("cc", []),
                "bcc": email_data.get("bcc", []),
                "subject": subject,
                "content_type": "plain",
                "folder": email_data["folder"],
                "body": body_content,
                "draft_reply": draft
            }
            results.append(draft_entry)
            email_draft_log.append(draft_entry)  # Log the draft

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/send-draft/{draft_index}")
async def send_draft_email(draft_index: int):
    smtp_config = config_store.get("smtp")
    imap_config = config_store.get("imap")
    if not smtp_config or not imap_config:
        raise HTTPException(status_code=400, detail="SMTP or IMAP configuration not found. Please configure first.")

    if draft_index < 0 or draft_index >= len(email_draft_log):
        raise HTTPException(status_code=404, detail="Draft not found.")

    draft = email_draft_log[draft_index]
    from_info = draft.get("from", {})
    to_email = from_info.get("email")
    if not to_email:
        raise HTTPException(status_code=400, detail="Draft does not have a valid sender to reply to.")

    subject = draft.get("subject", "")
    body = draft.get("draft_reply", draft.get("body", ""))

    smtp_username = imap_config.username
    smtp_password = imap_config.password

    try:
        send_email(smtp_config, to_email, subject, body, smtp_username, smtp_password)
        sent_entry = {
            "to": to_email,
            "subject": subject,
            "body": body
        }
        email_sent_log.append(sent_entry)
        # --- Remove the draft after sending ---
        email_draft_log.pop(draft_index)
        return {"message": "Draft email sent successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/email-history/drafts")
async def get_draft_history():
    return {
        "count": len(email_draft_log),
        "drafts": email_draft_log
    }

@app.get("/email-history/sent")
async def get_sent_history():
    return {
        "count": len(email_sent_log),
        "sent": email_sent_log
    }

@app.get("/email-history/all")
async def get_all_history():
    return {
        "draft_count": len(email_draft_log),
        "sent_count": len(email_sent_log),
        "drafts": email_draft_log,
        "sent": email_sent_log
    }

@app.delete("/email-history/drafts")
async def delete_draft(request: Request):
    data = await request.json()
    idx = data.get("index")
    if idx is None or not (0 <= idx < len(email_draft_log)):
        raise HTTPException(status_code=404, detail="Draft not found.")
    email_draft_log.pop(idx)
    return {"message": "Draft deleted.", "drafts": email_draft_log}


@app.post("/save-draft-to-imap")
async def save_draft_to_imap(payload: dict = Body(...)):
    """
    Save a draft to the user's IMAP Drafts folder.
    Expects: {
      "to": "recipient@example.com",
      "subject": "Subject",
      "body": "Draft body"
    }
    """
    imap_config = config_store.get("imap")
    if not imap_config:
        raise HTTPException(status_code=400, detail="IMAP configuration not found. Please configure first.")

    try:
        save_draft_imap(
            imap_host=imap_config.host,
            imap_port=imap_config.port,
            username=imap_config.username,
            password=imap_config.password,
            to_addr=payload["to"],
            subject=payload["subject"],
            body=payload["body"]
        )
        return {"message": "Draft saved to IMAP Drafts folder."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gmail-profile")
async def gmail_profile(request: Request):
    # In production, get the user's OAuth token from session or database
    # For demo, use mock data or fetch only email from config
    imap = config_store.get("imap")
    email = imap.username if imap else "unknown@gmail.com"

    # If you have an OAuth access token, fetch real profile info:
    access_token = getattr(imap, "oauth_token", None)
    if access_token:
        headers = {"Authorization": f"Bearer {access_token}"}
        resp = requests.get(
            "https://people.googleapis.com/v1/people/me?personFields=names,emailAddresses,photos",
            headers=headers
        )
        if resp.ok:
            data = resp.json()
            name = data.get("names", [{}])[0].get("displayName", "")
            photo = data.get("photos", [{}])[0].get("url", "")
            email = data.get("emailAddresses", [{}])[0].get("value", email)
            return {
                "email": email,
                "name": name,
                "photo": photo,
                "about": f"This is your Mailer Agent profile for {email}."
            }

    # Fallback: just show email from config and a generic photo
    return {
        "email": email,
        "name": "Your Name",
        "photo": "https://www.gravatar.com/avatar?d=mp&s=120",
        "about": f"This is your Mailer Agent profile for {email}."
    }

@app.get("/fetch-threads")
async def fetch_threads():
    imap_config = config_store.get("imap")
    if not imap_config:
        raise HTTPException(status_code=400, detail="IMAP configuration not found. Please configure first.")

    try:
        emails = fetch_emails(imap_config)
        action_items_data = load_action_items()  # Load saved action items
        # Merge action items into emails
        for email in emails:
            key = (email.get("subject", ""), email.get("from", {}).get("email", ""), email.get("date", ""))
            action_item = None
            for item in action_items_data:
                if (item["subject"], item["from"].get("email", ""), item["date"]) == key:
                    action_item = item.get("action_item")
                    break
            if not action_item:
                subject = email.get("subject", "")
                plain_body = email.get("body", {}).get("plain", "") if isinstance(email.get("body", ""), dict) else str(email.get("body", ""))
                action_item = get_action_item_from_email(subject, plain_body)
            email["action_item"] = action_item

        from email_reader import group_emails_into_threads
        threads = group_emails_into_threads(emails)
        return {"count": len(threads), "threads": threads}
    except Exception as e:
        import traceback
        print("FETCH THREADS ERROR:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-knowledge")
async def upload_knowledge(file: UploadFile = File(...)):
    import tempfile
    suffix = ".xlsx" if file.filename.endswith(".xlsx") else ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    if file.filename.endswith(".xlsx"):
        content = extract_text_from_excel(tmp_path)
    elif file.filename.endswith(".pdf"):
        content = extract_text_from_pdf(tmp_path)
    else:
        return {"error": "Unsupported file type"}
    knowledge_base.append({"filename": file.filename, "content": content})
    save_knowledge_base()

    # --- Vector DB logic ---
    global vector_index, vector_chunks, vector_chunk_sources
    chunks = chunk_text(content)
    chunk_embeddings = np.vstack([get_embedding(chunk) for chunk in chunks])
    if vector_index is None:
        vector_index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
        vector_index.add(chunk_embeddings)
        vector_chunks = chunks
        vector_chunk_sources = [(file.filename, chunk) for chunk in chunks]
    else:
        vector_index.add(chunk_embeddings)
        vector_chunks.extend(chunks)
        vector_chunk_sources.extend([(file.filename, chunk) for chunk in chunks])
    save_faiss_index(vector_index, VECTOR_INDEX_PATH)
    return {"message": "Knowledge uploaded", "filename": file.filename}

# --- Add RAG draft generation endpoint ---
def retrieve_relevant_context(email_body):
    # Simple keyword search; for production, use embeddings/vector search
    best_match = ""
    max_overlap = 0
    email_words = set(email_body.lower().split())
    for doc in knowledge_base:
        doc_words = set(doc["content"].lower().split())
        overlap = len(email_words & doc_words)
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = doc["content"]
    return best_match

@app.post("/generate-draft-with-rag")
async def generate_draft_with_rag(email: dict = Body(...)):
    from openai_processor import generate_reply
    email_body = email.get("body", "")
    context = retrieve_relevant_context_vector(email_body)
    prompt = f"Use the following context from company documents to answer the email:\n\n{context}\n\nEmail:\n{email_body}"
    draft = generate_reply(email.get("subject", ""), email.get("from", {}).get("name", ""), prompt)
    return {"draft": draft}

@app.get("/fetch-and-generate-drafts")
async def fetch_and_generate_drafts():
    """
    You are a mailer agent that reads mails from the user's IMAP inbox, you will act like a assistaince
    that will read the mails and generate drafts for them.
    consider the following:
    Read the data that is been given to you, you should use RAG and then compose a draft... make it formal and professional.
    do not miss the details and make the draft that can be understood by the user
    If the emails are asking for any query of details specific questions. provide the output in a concise but professional way,
    do not make the response unclear or unprofessional.
    """
    
    imap_config = config_store.get("imap")
    if not imap_config:
        raise HTTPException(status_code=400, detail="IMAP configuration not found. Please configure first.")

    try:
        emails = fetch_emails(imap_config)
        results = []

        # Use subject+from.email+date as a unique key for each email
        def email_key(email):
            return (
                email.get("subject", ""),
                email.get("from", {}).get("email", ""),
                email.get("date", "")
            )

        # Build a set of keys for already drafted emails
        drafted_keys = set(
            (d.get("subject", ""), d.get("from", {}).get("email", ""), d.get("date", ""))
            for d in email_draft_log
        )

        new_draft_count = 0

        for email_data in emails:
            key = email_key(email_data)
            if key in drafted_keys:
                continue  # Skip already drafted emails

            subject = email_data["subject"]
            sender_info = email_data["from"]
            sender_name = sender_info.get("name") or sender_info.get("email")
            from_email = "[Your Email]"  # Replace with actual sender if available
            to_email = sender_info.get("email")
            body_content = email_data["body"]["plain"] or ""

            # --- Forwarding logic ---
            to_emails = should_forward_email(email_data)
            if to_emails:
                forward_email(email_data, to_emails)

            # --- RAG logic ---
            context = retrieve_relevant_context(body_content)
            if context.strip():
                prompt = f"Use the following context from company documents to answer the email:\n\n{context}\n\nEmail:\n{body_content}"
            else:
                prompt = body_content

            draft = generate_reply(subject, sender_name, prompt, to_email=to_email, from_email=from_email)

            draft_entry = {
                "from": email_data["from"],
                "to": email_data.get("to", []),
                "cc": email_data.get("cc", []),
                "bcc": email_data.get("bcc", []),
                "subject": subject,
                "content_type": "plain",
                "folder": email_data["folder"],
                "date": email_data["date"],
                "body": body_content,
                "draft_reply": draft,
                "rag_used": bool(context.strip())
            }
            results.append(draft_entry)
            email_draft_log.append(draft_entry)  # Save to history
            new_draft_count += 1

        if new_draft_count == 0:
            return {"count": 0, "message": "All mails draft already generated."}
        else:
            return {"count": new_draft_count, "drafts": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/remove-knowledge")
async def remove_knowledge(filename: str = Body(...)):
    global knowledge_base
    knowledge_base = [k for k in knowledge_base if k["filename"] != filename]
    save_knowledge_base()
    return {"message": "Knowledge removed", "filename": filename}

@app.get("/knowledge-list")
async def knowledge_list():
    return {"files": [k["filename"] for k in knowledge_base]}

if os.path.exists(VECTOR_INDEX_PATH):
    vector_index = load_faiss_index(VECTOR_INDEX_PATH)
    # You should also persist and load vector_chunks and vector_chunk_sources for full mapping

def retrieve_relevant_context_vector(email_body, top_k=1):
    if vector_index is None or not vector_chunks:
        return ""
    query_emb = get_embedding(email_body).reshape(1, -1)
    D, I = vector_index.search(query_emb, top_k)
    return "\n".join([vector_chunks[i] for i in I[0]])

@app.post("/set-forwarding")
async def set_forwarding_rule(data: dict = Body(...)):
    """
    Expects: { "keyword_map": { "payment": ["a@b.com"], "account": ["c@d.com"] } }
    """
    forwarding_rules["keyword_map"] = data.get("keyword_map", {})
    save_forwarding_rules()
    return {"message": "Forwarding rules updated."}

@app.get("/get-forwarding")
async def get_forwarding_rule():
    return forwarding_rules

def should_forward_email(email_data):
    keyword_map = forwarding_rules.get("keyword_map", {})
    subject = email_data.get("subject", "").lower()
    body_field = email_data.get("body", "")
    if isinstance(body_field, dict):
        body = body_field.get("plain", "").lower()
    else:
        body = str(body_field).lower()
    action_item = email_data.get("action_item", "").lower()
    matched_emails = set()
    for keyword, emails in keyword_map.items():
        keyword_lower = keyword.lower()
        # Match keyword/phrase in subject, body, or action_item
        if (keyword_lower in subject or
            keyword_lower in body or
            keyword_lower in action_item):
            matched_emails.update(emails)
    return list(matched_emails)

def forward_email(email_data, to_emails):
    smtp_config = config_store.get("smtp")
    imap_config = config_store.get("imap")
    if not smtp_config or not imap_config or not to_emails:
        return False

    subject = email_data.get("subject", "")
    if not subject.lower().startswith("fwd:"):
        subject = "Fwd: " + subject

    # Handle both dict and str for body
    body_field = email_data.get("body", "")
    if isinstance(body_field, dict):
        original_body = body_field.get("plain", "")
    else:
        original_body = str(body_field)

    # Add original sender info at the top of the forwarded body
    original_from = email_data.get("from", {})
    original_from_str = f"{original_from.get('name', '')} <{original_from.get('email', '')}>"
    original_date = email_data.get("date", "")
    original_to = ", ".join(email_data.get("to", [])) if isinstance(email_data.get("to", []), list) else str(email_data.get("to", ""))
    forwarder_email = imap_config.username

    forwarded_body = (
        f"---------- Forwarded message ---------\n"
        f"From: {original_from_str}\n"
        f"Date: {original_date}\n"
        f"To: {original_to}\n"
        f"Subject: {subject}\n\n"
        f"{original_body}"
    )

    smtp_username = imap_config.username
    smtp_password = imap_config.password
    forwarded = False
    for to_email in to_emails:
        try:
            # Add the forwarder in CC
            send_email(
                smtp_config,
                to_email,
                subject,
                forwarded_body,
                smtp_username,
                smtp_password,
                cc=[forwarder_email]  # <-- CC the forwarder
            )
            forwarded_emails_log.append({
                "from": email_data.get("from"),
                "to": to_email,
                "cc": [forwarder_email],
                "subject": subject,
                "date": email_data.get("date"),
                "body": forwarded_body
            })
            forwarded = True
        except Exception as e:
            print(f"Forwarding to {to_email} failed:", e)
    return forwarded


@app.get("/unforwarded-mails")
async def get_unforwarded_mails():
    # Compute unforwarded mails: emails that match forwarding rules but have not been forwarded yet
    imap_config = config_store.get("imap")
    if not imap_config:
        raise HTTPException(status_code=400, detail="IMAP configuration not found. Please configure first.")

    try:
        emails = fetch_emails(imap_config)
        unforwarded_mails = []
        for email_data in emails:
            to_emails = should_forward_email(email_data)
            # Check if this email has already been forwarded
            already_forwarded = any(
                fwd.get("subject") == email_data.get("subject") and
                fwd.get("date") == email_data.get("date") and
                fwd.get("to") == to_emails
                for fwd in forwarded_emails_log
            )
            if to_emails and not already_forwarded:
                unforwarded_mails.append(email_data)
        return {"count": len(unforwarded_mails), "mails": unforwarded_mails}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forwarded-emails")
async def get_forwarded_emails():
    return {"count": len(forwarded_emails_log), "emails": forwarded_emails_log}

@app.post("/remove-forward-to-email")
async def remove_forward_to_email(data: dict = Body(...)):
    email = data.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="No email provided.")
    if "forward_to_history" in forwarding_rules and email in forwarding_rules["forward_to_history"]:
        forwarding_rules["forward_to_history"].remove(email)
        # If the current forward_to is being removed, clear it
        if forwarding_rules.get("forward_to") == email:
            forwarding_rules["forward_to"] = ""
        save_forwarding_rules()
        return {"message": "Email removed from history."}
    raise HTTPException(status_code=404, detail="Email not found in history.")

@app.post("/forward-email")
async def forward_email_api(email: dict = Body(...)):
    """
    Forward the given email to the appropriate addresses based on forwarding rules.
    Expects the full email object in the request body.
    """
    to_emails = should_forward_email(email)
    if not to_emails:
        raise HTTPException(status_code=400, detail="No matching forwarding rule for this email.")
    success = forward_email(email, to_emails)
    if success:
        return {"message": "Email forwarded successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to forward email.")

def get_action_item_from_email(subject, body):
    base_prompt = f"""
                "You have to read the email and underatand in detail about the content and motive of the email, "
                "after that you will bring the key insights of the email that will be reffered to it's action items in this case."
                "For eg- Email that has a content about starting a new project should be labeled as action items of - New Content Starting."
                "The action items has to be concised and and clearly state the nature and the key points of mail. "    

              Email Subject: {subject}
              Email Body: {body}

                Respond with only the action item sentence, nothing else.
"""
    for _ in range(3):
        result = extract_action_item(base_prompt)
        if len(result.split()) >= 3:
            return result
        # Strengthen the prompt if the result is too short
        base_prompt += "\n Respond with a of 4 to 5 words."
    return result  # Return whatever was last generated

@app.get("/fetch-mails-with-action-items")
async def fetch_mails_with_action_items():
    imap_config = config_store.get("imap")
    if not imap_config:
        raise HTTPException(status_code=400, detail="IMAP configuration not found. Please configure first.")

    try:
        emails = fetch_emails(imap_config)
        action_items_data = load_action_items()
        existing_keys = set(
            (item["subject"], item["from"].get("email", ""), item["date"])
            for item in action_items_data
        )
        new_items = []
        for email in emails:
            key = (email.get("subject", ""), email.get("from", {}).get("email", ""), email.get("date", ""))
            if key in existing_keys:
                continue  # Skip already processed
            subject = email.get("subject", "")
            plain_body = email.get("body", {}).get("plain", "") if isinstance(email.get("body", ""), dict) else str(email.get("body", ""))
            action_item = get_action_item_from_email(subject, plain_body)
            email_with_action = {
                "from": email.get("from"),
                "to": email.get("to"),
                "cc": email.get("cc"),
                "bcc": email.get("bcc"),
                "subject": subject,
                "folder": email.get("folder"),
                "body": plain_body,  # Only plain text!
                "date": email.get("date"),
                "unread": email.get("unread"),
                "action_item": action_item
            }
            action_items_data.append(email_with_action)
            new_items.append(email_with_action)
        if new_items:
            save_action_items(action_items_data)
        return {"count": len(action_items_data), "emails": action_items_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.post("/relevant-forward")
async def relevant_forward(
    purpose: str = Body(...),
    recipient: str = Body(...)
):
    """
    Forwards all emails semantically matching the given purpose to the recipient.
    """
    imap_config = config_store.get("imap")
    if not imap_config:
        raise HTTPException(status_code=400, detail="IMAP configuration not found. Please configure first.")

    emails = fetch_emails(imap_config)
    purpose_emb = get_embedding(purpose)
    threshold = 0.75  # You can tune this

    forwarded = []
    for email in emails:
        # Combine subject and body for best context
        subject = email.get("subject", "")
        body_field = email.get("body", "")
        if isinstance(body_field, dict):
            body = body_field.get("plain", "")
        else:
            body = str(body_field)
        email_text = f"{subject}\n{body}"
        email_emb = get_embedding(email_text)
        sim = cosine_similarity(purpose_emb, email_emb)
        if sim >= threshold:
            forward_email(email, [recipient])
            forwarded.append({
                "subject": subject,
                "to": recipient,
                "date": email.get("date"),
                "similarity": sim
            })

    return {"forwarded_count": len(forwarded), "forwarded": forwarded}

@app.get("/grouped-action-items")
async def grouped_action_items():
    """
    Returns a list of senders, each with their email ID and a list of action items.
    Each action item includes its id and text, so the frontend can fetch the full email by id.
    Example:
    [
      {
        "sender_email": "someone@example.com",
        "emails": [
          {
            "id": "...",
            "action_item": "Action phrase here"
          },
          ...
        ]
      },
      ...
    ]
    """
    imap_config = config_store.get("imap")
    if not imap_config:
        raise HTTPException(status_code=400, detail="IMAP configuration not found. Please configure first.")

    try:
        emails = fetch_emails(imap_config)
        action_items_data = load_action_items()
        grouped = {}

        for email in emails:
            sender_email = email.get("from", {}).get("email", "")
            subject = email.get("subject", "")
            date = email.get("date", "")
            email_id = f"{subject}|{sender_email}|{date}"

            # Find action item
            action_item = None
            for item in action_items_data:
                if (item["subject"], item["from"].get("email", ""), item["date"]) == (subject, sender_email, date):
                    action_item = item.get("action_item")
                    break
            if not action_item:
                plain_body = email.get("body", {}).get("plain", "") if isinstance(email.get("body", ""), dict) else str(email.get("body", ""))
                action_item = get_action_item_from_email(subject, plain_body)

            if sender_email not in grouped:
                grouped[sender_email] = []
            grouped[sender_email].append({
                "id": email_id,
                "action_item": action_item
            })

        # Convert to list format for JSON response
        result = [
            {
                "sender_email": sender,
                "emails": items
            }
            for sender, items in grouped.items()
        ]
        return {"count": len(result), "senders": result}
    except Exception as e:
        import traceback
        print("GROUPED ACTION ITEMS ERROR:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-email-by-id")
async def get_email_by_id(id: str):
    """
    Fetch a single email by its unique id (subject|sender_email|date).
    """
    imap_config = config_store.get("imap")
    if not imap_config:
        raise HTTPException(status_code=400, detail="IMAP configuration not found. Please configure first.")

    subject, sender_email, date = id.split("|", 2)
    emails = fetch_emails(imap_config)
    for email in emails:
        if (
            email.get("subject", "") == subject and
            email.get("from", {}).get("email", "") == sender_email and
            email.get("date", "") == date
        ):
            return {"email": email}
    raise HTTPException(status_code=404, detail="Email not found.")

