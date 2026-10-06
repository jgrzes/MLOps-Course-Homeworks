from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError("Input text cannot be empty.")
        if not isinstance(value, str):
            raise TypeError("Input text must be a string.")
        return value


class PredictResponse(BaseModel):
    prediction: str
