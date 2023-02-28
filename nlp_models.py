import os
try:
    from transformers import AutoTokenizer, AutoModel, TFAutoModel, pipeline
    from transformers import AutoModelForSequenceClassification
    from transformers import AutoModelForTokenClassification
    from transformers import TFAutoModelForSequenceClassification
except Exception as e:
    print("\ninstalling transformers")
    print(os.system("pip install transformers"))
    print(os.system("pip install sentencepiece"))
    print(os.system("pip install accelerate"))
try:
    from PIL import Image, ImageDraw
except Exception as e:
    print("\ninstalling pillow")
    print(os.system("pip install --upgrade protobuf==3.20.0"))
    print(os.system("pip install pillow"))

class NLPModel():
    def __init__(self, modelId, modelName, modelFunction):
        self.modelFunction = modelFunction
        self.modelId = modelId
        self.modelName = modelName
    async def RunModel(self, text_to_process):
        model_results = await self.modelFunction(text_to_process)
        return model_results 
    async def modelInfo(self):
        return {"name": self.modelName, "id":self.modelId}
    
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

async def distilRoberta_Emotion_ModelFunction(text):
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True, use_fast=True)
    analysis = classifier(text)
    if(len(analysis) == 1):
        analysis = analysis[0]
    print("\nanalysis",analysis)
    return analysis

async def tweetEval_Emotion_ModelFunction(text):
    classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion", return_all_scores=True, use_fast=True)
    analysis = classifier(text)
    if(len(analysis) == 1):
        analysis = analysis[0]
    print("\nanalysis",analysis)
    return analysis

async def tweetEval_Sentiment_ModelFunction(text):
    classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment", return_all_scores=True, use_fast=True)
    analysis = classifier(text)
    
    if(len(analysis) == 1):
        analysis = analysis[0]
    for index in range(0, len(analysis)):
        result = analysis[index]
        if(result["label"] == "LABEL_0"):
            analysis[index]["label"] = "positive"
            analysis[index]["score"] = float(analysis[index]["score"])
        elif(result["label"] == "LABEL_1"):
            analysis[index]["label"] = "neutral"
            analysis[index]["score"] = float(analysis[index]["score"])
        elif(result["label"] == "LABEL_2"):
            analysis[index]["label"] = "negative"
            analysis[index]["score"] = float(analysis[index]["score"])
    print("\nanalysis",analysis)
    return analysis

async def tweetEval_Irony_ModelFunction(text):
    classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-irony", return_all_scores=True, use_fast=True)
    analysis = classifier(text)
    if(len(analysis) == 1):
        analysis = analysis[0]

    for index in range(0, len(analysis)):
        result = analysis[index]
        if(result["label"] == "LABEL_0"):
            analysis[index]["label"] = "irony"
            analysis[index]["score"] = float(analysis[index]["score"])
        elif(result["label"] == "LABEL_1"):
            analysis[index]["label"] = "non_irony"
            analysis[index]["score"] = float(analysis[index]["score"])
    print("\nanalysis",analysis)
    return analysis

async def tweetEval_Climate_ModelFunction(text):
    classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-stance-climate", return_all_scores=True, use_fast=True)
    analysis = classifier(text)
    if(len(analysis) == 1):
        analysis = analysis[0]
    for index in range(0, len(analysis)):
        result = analysis[index]
        if(result["label"] == "LABEL_0"):
            analysis[index]["label"] = "climate_favor"
            analysis[index]["score"] = float(analysis[index]["score"])
        elif(result["label"] == "LABEL_1"):
            analysis[index]["label"] = "climate_neutral"
            analysis[index]["score"] = float(analysis[index]["score"])
        elif(result["label"] == "LABEL_2"):
            analysis[index]["label"] = "climate_against"
            analysis[index]["score"] = float(analysis[index]["score"])
    print("\nanalysis",analysis)
    return analysis
    
async def tweetEval_Offensive_Language_ModelFunction(text):
    classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-offensive", return_all_scores=True, use_fast=True)
    analysis = classifier(text)
    if(len(analysis) == 1):
        analysis = analysis[0]
    for index in range(0, len(analysis)):
        result = analysis[index]
        if(result["label"] == "LABEL_0"):
            analysis[index]["label"] = "offensive"
            analysis[index]["score"] = float(analysis[index]["score"])
        elif(result["label"] == "LABEL_1"):
            analysis[index]["label"] = "not_offensive"
            analysis[index]["score"] = float(analysis[index]["score"])
    print("\nanalysis",analysis)
    return analysis

async def distilbert_Uncased_Emotion_ModelFunction(text):
    classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True, use_fast=True)
    prediction = classifier(text)
    print("\nanalysis",prediction)
    return prediction
    
async def distilbert_jhartman_emotion_ModelFunction(text):
    classifier = pipeline("text-classification",model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True, use_fast=True)
    prediction = classifier(text)
    print("\nanalysis",prediction)
    return prediction

async def distilbert_crossencoder_zero_shot_classification_Categories_ModelFunction(text):
    classifier = pipeline("zero-shot-classification", model='cross-encoder/nli-distilroberta-base')
    candidate_labels = ['space & cosmos', 'scientific discovery', 'microbiology', 'robots', 'archeology', 'politics', 'sports','finance', 'computer science', 'technology', 'programming', 'software', 'drone', 'aviation', 'hacking', 'war']
    res = classifier(text, candidate_labels, multi_label=True)
    finalModelResultArray = []
    for iterator1 in range(0, len(res['scores'])):
        scoreObject = res['scores'][iterator1]
        scoreObject = float(scoreObject)
        label = candidate_labels[iterator1]
        resObj = {"label": label, "score": scoreObject}
        finalModelResultArray.append(resObj)
    print("\nanalysis",finalModelResultArray)
    return finalModelResultArray

async def distilbert_crossencoder_zero_shot_classification_Emotion_ModelFunction(text):
    classifier = pipeline("zero-shot-classification", model='cross-encoder/nli-distilroberta-base')
    candidate_labels =  ['contradiction', 'entailment', 'sarcasm','neutral']
    res = classifier(text, candidate_labels, multi_label=True)
    finalModelResultArray = []
    for iterator1 in range(0, len(res['scores'])):
        scoreObject = res['scores'][iterator1]
        scoreObject = float(scoreObject)

        label = candidate_labels[iterator1]
        resObj = {"label": label, "score": scoreObject}
        finalModelResultArray.append(resObj)
    print("\nanalysis",finalModelResultArray)
    return finalModelResultArray

async def distilbert_crossencoder_zero_shot_classification_Political_Affiliation_ModelFunction(text):
    classifier = pipeline("zero-shot-classification", model='cross-encoder/nli-distilroberta-base')
    candidate_labels_politics = ['republican', 'democrat', 'capitalism', 'communism', 'socialism', 'anarchism','marxism']
    result1 = classifier(text, candidate_labels_politics, multi_label=True)
    finalModelResultArray = []
    for iterator1 in range(0, len(result1['scores'])):
        scoreObject = result1['scores'][iterator1]
        scoreObject = float(scoreObject)
        label = candidate_labels_politics[iterator1]
        resObj = {"label": label, "score": scoreObject}
        finalModelResultArray.append(resObj)
    print("\nanalysis",finalModelResultArray)
    return finalModelResultArray

async def multi_language_mDeberta_v3_zero_shot_classification_Political_Affiliation_ModelFunction(text):
    classifier = pipeline("zero-shot-classification", model='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli')
    candidate_labels_politics = ['republican', 'democrat', 'capitalism', 'communism', 'socialism', 'anarchism','marxism']
    result1 = classifier(text, candidate_labels_politics, multi_label=True)
    finalModelResultArray = []
    for iterator1 in range(0, len(result1['scores'])):
        scoreObject = result1['scores'][iterator1]
        scoreObject = float(scoreObject)
        label = candidate_labels_politics[iterator1]
        resObj = {"label": label, "score": scoreObject}
        finalModelResultArray.append(resObj)
        
    print("\nanalysis",finalModelResultArray)
    return finalModelResultArray
async def multi_language_mDeberta_v3_zero_shot_classification_Positive_Negative_ModelFunction(text):
    classifier = pipeline("zero-shot-classification", model='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli')
    candidate_labels_approval = ['positive', 'approval','agreement', 'negative', 'disaproval', 'disagreement', 'neutral']
    result2 = classifier(text, candidate_labels_approval, multi_label=True)
    finalModelResultArray = []
    for iterator2 in range(0, len(result2['scores'])):
        scoreObject = result2['scores'][iterator2]
        scoreObject = float(scoreObject)
        label = candidate_labels_approval[iterator2]
        resObj = {"label": label, "score": scoreObject}
        finalModelResultArray.append(resObj)
    print("\nanalysis",finalModelResultArray)
    return finalModelResultArray

async def multi_language_mDeberta_v3_zero_shot_classification_Country_ModelFunction(text):
    classifier = pipeline("zero-shot-classification", model='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli')
    candidate_labels_politics = ['usa','russia','china','argentina','israel', 'taiwan', 'ukraine']
    result1 = classifier(text, candidate_labels_politics, multi_label=True)
    finalModelResultArray = []
    for iterator1 in range(0, len(result1['scores'])):
        scoreObject = result1['scores'][iterator1]
        scoreObject = float(scoreObject)
        label = candidate_labels_politics[iterator1]
        resObj = {"label": label, "score": scoreObject}
        finalModelResultArray.append(resObj)
    print("\nanalysis",finalModelResultArray)
    return finalModelResultArray

async def facebook_bert_large_mnli_ModelFunction(text):
    classifier = pipeline("zero-shot-classification", model='facebook/bart-large-mnli')
    candidate_labels_politics = ['positive','negative','neutral', 'hate', 'love', 'suprise','anger','confusion', 'sadness','disappointment', 'disgust', 'curiosity', 'pride', 'annoyance', 'grief','disapproval', 'optimism', 'contradiction', 'entailment', 'sarcasm', 'fear', 'embarrassment', 'nervousness', 'desire', 'gratitude', 'amusement', 'excitement', 'caring', 'joy', 'approval', 'admiration']
    result1 = classifier(text, candidate_labels_politics, multi_label=True)
    finalModelResultArray = []
    for iterator1 in range(0, len(result1['scores'])):
        scoreObject = result1['scores'][iterator1]
        scoreObject = float(scoreObject)
        label = candidate_labels_politics[iterator1]
        resObj = {"label": label, "score": scoreObject}
        finalModelResultArray.append(resObj)
    print("\nanalysis",finalModelResultArray)
    return finalModelResultArray

async def unitaryai_detoxify_ModelFunction(text):
    from detoxify import Detoxify
    model_results = Detoxify('original').predict(text)
    finalModelResultArray = []
    for result in model_results:
        resObj = {"label": result, "score": float(model_results[result])}
        finalModelResultArray.append(resObj)
    print("\nanalysis",finalModelResultArray)
    return finalModelResultArray 

# MODEL_distilRoberta_Emotion = NLPModel(1, "distil_Roberta", distilRoberta_Emotion_ModelFunction)
MODEL_tweetEval_Sentiment = NLPModel(2, "tweet_Eval_Sentiment", tweetEval_Sentiment_ModelFunction)
MODEL_tweetEval_Irony = NLPModel(3, "tweet_Eval_Irony", tweetEval_Irony_ModelFunction)
MODEL_tweetEval_Climate = NLPModel(4, "tweet_Eval_Climate", tweetEval_Climate_ModelFunction)
MODEL_distilbert_Uncased_Emotion = NLPModel(5, "distilbert_uncased_emotion", distilbert_Uncased_Emotion_ModelFunction)
MODEL_distilRoberta_Emotion = NLPModel(6, "distilbert_jhartman_emotion", distilRoberta_Emotion_ModelFunction)
MODEL_distilbert_crossencoder_zero_shot_classification_Political_Affiliation = NLPModel(7, "distilbert_crossencoder_zero_shot_classification_Political_Affiliation", distilbert_crossencoder_zero_shot_classification_Political_Affiliation_ModelFunction)
MODEL_distilbert_crossencoder_zero_shot_classification_Emotion = NLPModel(8, "distilbert_crossencoder_zero_shot_classification_Emotion", distilbert_crossencoder_zero_shot_classification_Emotion_ModelFunction)
MODEL_mDeberta_multi_language_zero_shot_classification_Political_Affiliation = NLPModel(9, "distilbert_crossencoder_zero_shot_classification_Emotion", multi_language_mDeberta_v3_zero_shot_classification_Political_Affiliation_ModelFunction)
MODEL_mDeberta_multi_language_zero_shot_classification_Positive_Negative = NLPModel(10, "distilbert_crossencoder_zero_shot_classification_Positive_Negative", multi_language_mDeberta_v3_zero_shot_classification_Positive_Negative_ModelFunction)
MODEL_mDeberta_multi_language_zero_shot_classification_Country = NLPModel(11, "distilbert_crossencoder_zero_shot_classification_Country", multi_language_mDeberta_v3_zero_shot_classification_Country_ModelFunction)
MODEL_tweetEval_Offensive_Language = NLPModel(12, "tweet_Eval_Offensive_Language", tweetEval_Offensive_Language_ModelFunction)
MODEL_facebook_bert_large_mnli = NLPModel(13, "facebook_bert_large_mnli", facebook_bert_large_mnli_ModelFunction)
MODEL_unitaryai_detoxify = NLPModel(14, "unitaryai_detoxify", unitaryai_detoxify_ModelFunction)
MODEL_distilbert_crossencoder_zero_shot_classification_Categories = NLPModel(15, "distilbert_crossencoder_zero_shot_classification_Categories", distilbert_crossencoder_zero_shot_classification_Categories_ModelFunction)



# ===============================================================================================================
# Load as many Models as you'd like! 
# Model functions must take a single text parameter and 
#   return an array with the following format:
# {
#     "label": "happy_example",
#     "score": .2342 # Float32
# }
# ===============================================================================================================
def getModels():
    models = [
        # MODEL_tweetEval_Sentiment,
        # MODEL_distilbert_crossencoder_zero_shot_classification_Political_Affiliation,
        MODEL_tweetEval_Irony,
        MODEL_tweetEval_Climate,
        MODEL_distilbert_Uncased_Emotion,
        MODEL_distilRoberta_Emotion,
        MODEL_distilbert_crossencoder_zero_shot_classification_Emotion,
        MODEL_mDeberta_multi_language_zero_shot_classification_Political_Affiliation,
        MODEL_mDeberta_multi_language_zero_shot_classification_Positive_Negative,
        MODEL_mDeberta_multi_language_zero_shot_classification_Country,
        MODEL_tweetEval_Offensive_Language,
        MODEL_distilbert_crossencoder_zero_shot_classification_Categories,
        MODEL_facebook_bert_large_mnli,
        MODEL_unitaryai_detoxify,
    ]
    return models

async def getModelIds():
    models = getModels()
    model_id_list = []
    for model in models:
        model_id = model.modelId
        model_id_list.append(model_id)
    return model_id_list

async def pullEntitiesFromText(text):
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    entities_extracted = nlp(text)
    response_object = []
    for entity in entities_extracted:
        group = entity["entity"]
        word = entity["word"]
        score = str(entity["score"])
        response_object.append({"word": word, "group": group, "score": score})
    return response_object
