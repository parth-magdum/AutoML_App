from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pycaret.classification import setup, compare_models, predict_model, save_model, pull
import shutil
import os
import uuid

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/automl", response_class=HTMLResponse)
async def automl(request: Request, file: UploadFile = File(...), target_column: str = Form(...)):
    temp_id = str(uuid.uuid4())
    temp_path = f"temp_{temp_id}.csv"

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    df = pd.read_csv(temp_path)
    os.remove(temp_path)

    # Setup and run PyCaret (classification only)
    s = setup(data=df, target=target_column, session_id=123, verbose=False)
    best_model = compare_models()
    preds = predict_model(best_model)
    leaderboard = pull()

    # Save predictions to CSV
    pred_path = f"app/model/predictions_{temp_id}.csv"
    preds.to_csv(pred_path, index=False)

    # Prepare preview and full tables
    table_preview = preds.head(5).to_html(classes='table table-striped', index=False)
    table_full = preds.to_html(classes='table table-striped', index=False)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "table_preview": table_preview,
        "table_full": table_full,
        "leaderboard": leaderboard.to_html(classes='table table-bordered', index=False),
        "download_link": f"/download/{temp_id}"
    })

@app.get("/download/{file_id}")
async def download_predictions(file_id: str):
    path = f"app/model/predictions_{file_id}.csv"
    return FileResponse(path, filename="predictions.csv", media_type='text/csv')
