
import requests

#for local
#url = 'http://localhost:9696/predict'

# Deployed
url='https://cardclass-2avfrxfgrq-lm.a.run.app/predict'


img_url1 = 'https://cdn.britannica.com/56/129056-050-318DAD51/joker-jokes-playing-card.jpg'
img_url2='https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Playing_card_club_A.svg/614px-Playing_card_club_A.svg.png'

response1 = requests.post(url, json={'url':img_url1}).json()
print(dict(sorted(response1.items(), key=lambda x: x[1], reverse=True)))
response2 = requests.post(url, json={'url':img_url2}).json()
print(dict(sorted(response2.items(), key=lambda x: x[1], reverse=True)))
response2 = requests