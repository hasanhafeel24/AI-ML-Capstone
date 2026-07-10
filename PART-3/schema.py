from pydantic import BaseModel


class StudyResponse(BaseModel):
    topic: str
    difficulty: str
    summary: str