from fastapi import FastAPI, Path, File, UploadFile, Form, Request
from typing import Optional
from pydantic import BaseModel
import os
from models.tea_disease_model import process_image_disease
from models.tea_disease_model import process_image_stem_cancer
from models.tea_disease_percentage import prediction_model
from models.insect_detection import preprocess
import requests
import shutil

app = FastAPI()


@app.post("/predict/tea-disease-blister-blight")
async def predict_image_disease(req: Request):
    json_data = await req.json()
    url = json_data["url"]
    response = requests.get(url, stream=True)
    response.raise_for_status()
    file_name = url.split("/")[-1]
    file_path = rf"D:\Python\python_project\tea_api\temp_images\{file_name}"
    with open(file_path, "wb") as f:
        response.raw.decode_content = True
        shutil.copyfileobj(response.raw, f)
        response = process_image_disease(rf"{file_path}")
        f.close()
        os.remove(file_path)
        return response


@app.post("/predict/tea-disease-stem-canncer")
async def predict_image_disease(req: Request):
    json_data = await req.json()
    url = json_data["url"]
    response = requests.get(url, stream=True)
    response.raise_for_status()
    file_name = url.split("/")[-1]
    file_path = rf"D:\Python\python_project\tea_api\temp_images\{file_name}"
    with open(file_path, "wb") as f:
        response.raw.decode_content = True
        shutil.copyfileobj(response.raw, f)
        response = process_image_stem_cancer(rf"{file_path}")
        f.close()
        os.remove(file_path)
        return response


@app.post("/predict/tea-disease-damage-percentage")
async def predict_image_disease(req: Request):
    json_data = await req.json()
    url = json_data["url"]
    response = requests.get(url, stream=True)
    response.raise_for_status()
    file_name = url.split("/")[-1]
    file_path = rf"D:\Python\python_project\tea_api\temp_images\{file_name}"
    with open(file_path, "wb") as f:
        response.raw.decode_content = True
        shutil.copyfileobj(response.raw, f)
        response = prediction_model(rf"{file_path}")
        f.close()
        os.remove(file_path)
        return {"Disease Damage Percentage ": f"{response}%"}


@app.post("/predict/tea-insect-detection")
async def predict_image_disease(req: Request):
    json_data = await req.json()
    url = json_data["url"]
    print(url)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    file_name = url.split("/")[-1]
    file_path = rf"D:\Python\python_project\tea_api\temp_images\{file_name}"
    with open(file_path, "wb") as f:
        response.raw.decode_content = True
        shutil.copyfileobj(response.raw, f)
        response = preprocess(rf"{file_path}")
        f.close()
        os.remove(file_path)
        response = int(response)
        if response > 0:
            return {"Insect detected": True, "Insect count": response}
        else:
            return {"Insect detected": False, "Insect count": 0}
