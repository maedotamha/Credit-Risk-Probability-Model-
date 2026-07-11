"""FastAPI service entry point.

The deployed prediction API will be completed in Task 6.
"""

from fastapi import FastAPI

app = FastAPI(title="Credit Risk Probability Model")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
