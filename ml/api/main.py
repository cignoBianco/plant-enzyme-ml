from pathlib import Path
import json

import joblib
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, EsmModel
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


with open(MODEL_DIR / "config.json", encoding="utf-8") as f:
    CONFIG = json.load(f)


CLASSES = CONFIG["classes"]
ESM_MODEL_NAME = CONFIG["esm_model"]


classifier = joblib.load(
    MODEL_DIR / "esm2_logistic.joblib"
)

tokenizer = AutoTokenizer.from_pretrained(
    ESM_MODEL_NAME
)

esm_model = EsmModel.from_pretrained(
    ESM_MODEL_NAME
)

esm_model.eval()


app = FastAPI(
    title="Plant Enzyme ML API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    sequence: str = Field(
        min_length=20,
        max_length=5000,
    )


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]


def clean_sequence(sequence: str) -> str:
    sequence = "".join(sequence.split()).upper()

    allowed = set("ACDEFGHIKLMNPQRSTVWY")

    if not sequence:
        raise ValueError("Sequence is empty.")

    invalid = sorted(set(sequence) - allowed)

    if invalid:
        raise ValueError(
            f"Invalid amino-acid characters: {invalid}"
        )

    return sequence


def get_embedding(sequence: str):
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
    )

    with torch.no_grad():
        outputs = esm_model(**inputs)

    hidden = outputs.last_hidden_state

    # Убираем BOS/EOS
    residue_embeddings = hidden[:, 1:-1, :]

    embedding = residue_embeddings.max(dim=1).values

    return embedding.cpu().numpy()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": ESM_MODEL_NAME,
        "embedding_dim": CONFIG["embedding_dim"],
        "classes": CLASSES,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    try:
        sequence = clean_sequence(
            request.sequence
        )

        embedding = get_embedding(
            sequence
        )

        prediction = classifier.predict(
            embedding
        )[0]

        probabilities = classifier.predict_proba(
            embedding
        )[0]

        probability_dict = {
            class_name: float(probability)
            for class_name, probability in zip(
                classifier.classes_,
                probabilities,
            )
        }

        return PredictionResponse(
            prediction=prediction,
            probabilities=probability_dict,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
