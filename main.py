from fastapi import FastAPI, File
from segmentation import get_yolov5, get_image_from_bytes
from pydantic import BaseModel, conlist
from typing import List,Optional
import pandas as pd
from recmodel import recommend,output_recommended_recipes
from starlette.responses import Response
import io
from PIL import Image
import json
import base64
from fastapi.middleware.cors import CORSMiddleware

dataset=pd.read_csv('Data/sample.csv')

print(dataset)
model = get_yolov5()

app = FastAPI()

origins = [
    "192.168.43.47:8000",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class params(BaseModel):
    n_neighbors:int=5
    return_distance:bool=False

class PredictionIn(BaseModel):
    nutrition_input:conlist(item_type=float, min_length=9, max_length=9)
    ingredients:list[str]=[]
    params:Optional[params]


class Recipe(BaseModel):
    Name:str
    CookTime:Optional[str]
    PrepTime:Optional[str]
    TotalTime:Optional[str]
    RecipeIngredientParts:list[str]
    Calories:float
    FatContent:float
    SaturatedFatContent:float
    CholesterolContent:float
    SodiumContent:float
    CarbohydrateContent:float
    FiberContent:float
    SugarContent:float
    ProteinContent:float
    RecipeInstructions:list[str]

class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None

class UploadimageSchema(BaseModel):
    image:bytes 

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get('/notify/v1/health')
def get_health():
    """
    Usage on K8S
    readinessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    livenessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    :return:
        dict(msg='OK')
    """
    return dict(msg='OK')

@app.post("/object-to-json")
async def detect_ingre_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    return {"result": detect_res}

@app.post("/binary-object-to-json")
async def binary_detect_ingre_return_json_result(payload:UploadimageSchema):
    data_split = payload.image.split('base64,')
    encoded_data = data_split[1]
    input_image = base64.b64decode(encoded_data)
    with open("uploaded_image.png", "wb") as writer:
        writer.write(input_image)

    return {"detail": "Profile update successful"}


@app.post("/object-to-img")
async def detect_food_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels
    for img in results.ims:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")

@app.post("/predict/",response_model=PredictionOut)
def update_item(prediction_input:PredictionIn):
    recommendation_dataframe=recommend(dataset, prediction_input.nutrition_input, prediction_input.ingredients, prediction_input.params.dict())
    output=output_recommended_recipes(recommendation_dataframe)
    if output is None:
        return {"output":None}
    else:
        return {"output":output}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)
