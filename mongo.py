from pymongo import MongoClient
client = MongoClient('localhost:27017')
db = client.sample
db = client.get_database('sample')
records = db.employee
print(records.count_documents({}))
print(list(records.find()))
