import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI(title= "Sentiment_Analyzer API")

class TextInput(BaseModel):
    text:str
    
@app.get("/")
def home():
    return {"Message": "Welcome to the Sentiment_Analyzer_API"}

@app.post("/sentiment")
async def get_sentiment(input: TextInput):
    analyzer = SentimentIntensityAnalyzer()
    result = analyzer.polarity_scores(input.text)
    sentiment = None
    if result["compound"] >= 0.05:
        sentiment = "Positive"
    elif result["compound"] == 0:
        sentiment = "Neutral"
    elif result["compound"] <= 0.05:
        sentiment = "Negative"
    else:
        sentiment = "No comment"
        
    return {"result": f"The sentiment of the text is {sentiment}!"}

# To run the API
if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port=8000)


