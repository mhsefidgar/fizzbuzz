from pydantic import BaseModel
from fastapi import FastAPI
from transformers import T5Tokenizer, T5ForConditionalGeneration
import warnings
warnings.filterwarnings("ignore")

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")



#print(tokenizer.decode(outputs[0], skip_special_tokens=True))
app = FastAPI()


class TranslationObject(BaseModel):
    WordQuery: str 


@app.post("/add_data")
async def add_data(Payload: TranslationObject):
    TextTranslate = Payload.WordQuery
    input_ids = tokenizer(TextTranslate, return_tensors="pt").input_ids
    TranslatedText = model.generate(input_ids)
    WordTranslated=tokenizer.decode(TranslatedText[0], skip_special_tokens=True)
   
    return TextTranslate + " means 👉 " + WordTranslated + " 👈 in German"
    