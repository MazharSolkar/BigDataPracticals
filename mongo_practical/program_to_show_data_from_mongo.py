from pymongo import MongoClient
client = MongoClient('localhost:27017')
db = client.get_database('practice')
records = db.employee
print(records.count_documents({}))
print(list(records.find()))

