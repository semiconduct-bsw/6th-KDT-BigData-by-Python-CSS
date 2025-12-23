import requests

url = "https://jsonplaceholder.typicode.com/todos"
response = requests.get(url)
data_list = response.json()

for i in range(10):
    print(data_list[i]['title'])
