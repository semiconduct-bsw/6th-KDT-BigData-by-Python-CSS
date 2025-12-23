import requests

url = "https://jsonplaceholder.typicode.com/todos"

response = requests.get(url)

data_list = response.json()
for lists in data_list:
    print(lists['title'])
