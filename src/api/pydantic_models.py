"""Pydantic schemas for the prediction API.

Request and response schemas will be expanded in Task 6.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str

