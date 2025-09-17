from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Сервис для пространственно-временных рядов работает!"}


class SelectRequest(BaseModel):
    dataset_description: str
    task: str
    candidates: list[str] = ["pysteps", "sktime", "tslearn", "torch_geometric"]


@app.post("/select_model")
async def select_model_endpoint(req: SelectRequest):
    try:
        from app.llm import select_model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM module import failed: {e}")

    result = select_model(req.dataset_description, req.task, req.candidates)
    return result


class SelectAndPredictRequest(BaseModel):
    dataset_csv: str
    dataset_description: str
    task: str


@app.post("/select_and_predict")
def select_and_predict(req: SelectAndPredictRequest):
    try:
        from app.llm import select_model
        from app.model_selector import find_best_model
        from app.model_runner import run_model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal import failed: {e}")

    # ask LLM for suggestion
    suggestion = select_model(req.dataset_description, req.task, ["pysteps", "sktime", "tslearn", "torch_geometric"])

    # choose nearest model in local vector db based on suggestion text
    suggestion_text = suggestion.get("library", "") + ": " + suggestion.get("model_choice", "")
    sel = find_best_model(suggestion_text)
    selected = sel.get("selected")

    # run model
    code_py, predictions_csv = run_model(selected.get("library"), req.dataset_csv)

    return {
        "selected": selected,
        "selection_score": sel.get("score"),
        "suggestion": suggestion,
        "code_py": code_py,
        "predictions_csv": predictions_csv,
    }
