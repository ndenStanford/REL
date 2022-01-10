import requests

print("here")

IP_ADDRESS = "http://localhost"
PORT = "1235"
text_doc = "If you're going to try, go all the way - Charles Bukowski"

document = {
    "text": text_doc,
    "spans": [],  # in case of ED only, this can also be left out when using the API
}

API_result = requests.post("{}:{}".format(IP_ADDRESS, PORT), json=document)  # .json()
import pdb

pdb.set_trace()
print(API_result)
