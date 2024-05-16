import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("C:/Users/Admin/ML_MINIPROJECT/serviceAccountKey.json")

try:
    app = firebase_admin.get_app()
except ValueError:
    app = firebase_admin.initialize_app(cred, {
        'databaseURL': "https://ml-miniproject-6969-default-rtdb.firebaseio.com/"
    })

ref = db.reference('emp')
data = {
    "159753":
        {
            "name": "Vedant Shejwal",
            "department": "Doctor",
            "safety_violation":1,
            "last_violation": "2023-04-22 00:54:34"
        },
    "357951":
        {
            "name": "Harsh Pande",
            "department": "Construction",
            "safety_violation":0,
            "last_violation": "2023-04-24 00:54:34"
        },
    "486248":
        {
            "name": "Utkarsh Gupta",
            "department": "Welding",
            "safety_violation":2,
            "last_violation": "2023-04-25 00:54:34"
        }
}

for key, value in data.items():
    ref.child(key).set(value)
