from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import yagmail
import re

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# -------------------------
# Email Validation
# -------------------------
def is_valid_email(email):
    regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(regex, email)

# -------------------------
# TOPSIS Implementation
# -------------------------
def topsis(df, weights, impacts):
    data = df.iloc[:, 1:].values.astype(float)

    # Step 1: Normalize
    norm = np.sqrt((data ** 2).sum(axis=0))
    normalized = data / norm

    # Step 2: Apply weights
    weighted = normalized * weights

    # Step 3: Ideal best & worst
    ideal_best, ideal_worst = [], []

    for i in range(len(impacts)):
        if impacts[i] == '+':
            ideal_best.append(weighted[:, i].max())
            ideal_worst.append(weighted[:, i].min())
        else:
            ideal_best.append(weighted[:, i].min())
            ideal_worst.append(weighted[:, i].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    # Step 4: Distance calculation
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    # Step 5: TOPSIS Score
    score = dist_worst / (dist_best + dist_worst)

    df["Topsis Score"] = score
    df["Rank"] = df["Topsis Score"].rank(ascending=False)

    return df

# -------------------------
# Home Page
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -------------------------
# Handle Form Submission
# -------------------------
@app.post("/submit/")
async def submit(
    file: UploadFile,
    weights: str = Form(...),
    impacts: str = Form(...),
    email: str = Form(...)
):
    # Email validation
    if not is_valid_email(email):
        return {"error": "Invalid email format"}

    weights = list(map(float, weights.split(",")))
    impacts = impacts.split(",")

    if len(weights) != len(impacts):
        return {"error": "Weights and impacts count must be same"}

    for i in impacts:
        if i not in ['+', '-']:
            return {"error": "Impacts must be + or - only"}

    df = pd.read_csv(file.file)

    if df.shape[1] < 3:
        return {"error": "CSV must have at least 3 columns"}

    result = topsis(df, weights, impacts)

    output_file = "result.csv"
    result.to_csv(output_file, index=False)

    # -------------------------
    # Send Email
    # -------------------------
    sender_email = "yourgmail@gmail.com"
    app_password = "your_gmail_app_password"

    yag = yagmail.SMTP(sender_email, app_password)
    yag.send(
        to=email,
        subject="TOPSIS Result",
        contents="Please find the attached TOPSIS result file.",
        attachments=output_file
    )

    return {"message": "✅ Result sent to your email successfully!"}
