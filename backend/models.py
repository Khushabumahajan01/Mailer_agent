from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Optional
from datetime import datetime

class SMTPConfig(BaseModel):
    host: str = Field(..., example="smtp.gmail.com")
    port: int = Field(..., example=587)

class IMAPFilters(BaseModel):
    unread: Optional[bool] = Field(True)
    from_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    to_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")

    @validator("from_date", "to_date")
    def validate_date_format(cls, v):
        if v:
            try:
                datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Date must be in YYYY-MM-DD format")
        return v

class IMAPConfig(BaseModel):
    host: str = Field(..., example="imap.gmail.com")
    port: int = Field(..., example=993)
    username: EmailStr
    password: str
    folders: Optional[List[str]] = Field(default_factory=lambda: ["INBOX", "Sent", "Sent Mail", "[Gmail]/Sent Mail"])
    filters: Optional[IMAPFilters] = None  # <-- Add filters here

class EmailConfigPayload(BaseModel):
    smtp: SMTPConfig
    imap: IMAPConfig
