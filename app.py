from flask import Flask, request
from flask_cors import CORS
from flask_session import Session
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
from pymongo import MongoClient
import json
from oauth2client.service_account import ServiceAccountCredentials
import gspread

model = SentenceTransformer("all-MiniLM-L6-v2")
app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
# add options=[<allowed domains>]
CORS(app, allow_headers="Content-Type", supports_credentials=True)

# MongoDB
MONGO_URI = "mongodb+srv://Souvikb:5UDNztJW24RIcprN@faqsystem.arqlwlk.mongodb.net/?retryWrites=true&w=majority&appName=FAQSystem"
client = MongoClient(MONGO_URI)
db = client["LawAndAI2025"]
answeredQuestions = db["AnsweredQuestions"]

# Google API config
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/16eP2Q7e_CdHSxoOsK4UjqL-CNnnLAPtHGXLhMrD6c6I"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
gc = gspread.authorize(creds)

# Input: two normalized vectors
def COSIM(vector1, vector2):
    return dot(vector1, vector2)

# figure this out with testing
MINIMUM_COS_SIM = 0.4

# Find the closest vector to the input query
def matchResponse(query):
    queryEmbeding = model.encode(query, normalize=True)
    entries = answeredQuestions.find()
    result = {}
    bestSim = 0
    for entry in entries:
        cosim = COSIM(entry["embedding"], queryEmbeding)
        if cosim > bestSim:
            result = entry
            bestSim = cosim
    del result["_id"]
    del result["embedding"]
    result["matchFound"] = True
    if bestSim < MINIMUM_COS_SIM:
        result["matchFound"] = False
    return result

# Add an unanswered question to the database
# Have a minimum cosim to prevent garbage questions clogging the db
# Flask limiter?
# Might use mongo instead of sheets
def addNewQuestion(query):
    try:
        sh = gc.open_by_url(SPREADSHEET_URL)
        worksheet = sh.get_worksheet(1)
        list_of_values = worksheet.col_values(1)
        col = len(list_of_values) + 1
        worksheet.update_cell(col, 1, query)
        return True
    except Exception as e:
        print(e)
        return False

# Handle input
@app.route("/inputQuery", methods=["POST", "GET", "OPTIONS"])
def handleQuery():
    if request.method == "POST":
        query = request.get_json()["query"]
        return json.dumps(matchResponse(query))
    else:
        return "Flask server is running"

@app.route("/newQuery", methods=["POST", "GET", "OPTIONS"])
def submitNewQuestion():
    if request.method == "POST":
        query = request.get_json()["query"]
        return json.dumps(addNewQuestion(query))
    else:
        return ""

# if __name__ == "__main__":
#     app.run(debug=True, port=8000)
