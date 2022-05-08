from pymongo import MongoClient
client = MongoClient('localhost:27017')
db = client.get_database('sample')
records = db.employee

print("\n############### Count of Records ###############")
print(records.count_documents({}))

print("\n############### list of records ###############")
print(list(records.find()))

print("\n############### one record updated ###############")
myquery = { "eno" : 4}
newvalues = { "$set":{"ename":"abhijeet"}}
records.update_one(myquery,newvalues)
for v in records.find():
    print(v)

print("\n############### one record inserted ###############")
myq1={"eno":6,"name":"Raj","location":"India"}
x=records.insert_one(myq1)
for v in records.find():
    print(v)

print("\n############### one record deleted ###############")
y=records.delete_one({"name":"Raj"})
for v in records.find():
    print(v)
print("\n__By Abhijeet Maity")



