import pandas as pd
from fastapi import FastAPI
from classification_model.predict import make_prediction


app = FastAPI(title="Consumer Complaint Classification API")


@app.get("/predict")
async def predict(complaint: str):
    # complaint = "I took out a {$5000.00} loan with XXXX XXXX back in XX/XX/XXXX or XX/XX/XXXX for my attendance at" \
    #             " University XXXX XXXX. Once I started repayment on them my loan has increased to almost {$10.00}." \
    #             " So I accumulated double in interest. How is this possible. I went back to school and deferred my" \
    #             " loans which was until XX/XX/XXXX. By this time navient has now taken over my student loans. I was" \
    #             " n't advised of the change. I 've to one person in regards to my account and they say I " \
    #             "owe {$300.00}. which I was only pay {$100.00}. They have been calling me nonstop all day." \
    #             " I do not want to speak to them unless someone is willing to help. No one is willing they only" \
    #             " want there money. Why is my interest so high? Why are n't you trying to help me lower my monthly " \
    #             "payment. why is the interest that much on a college student."
    return {"result": make_prediction(input_data=pd.DataFrame({"Consumer complaint narrative": [complaint]}))}
