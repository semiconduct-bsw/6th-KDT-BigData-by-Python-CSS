from flask import Flask, render_template, request

app = Flask(__name__)

#1. path paramter
#2. query paramter
#3. request body

# path parameter
@app.route("/find-user/<id>")
def find_user(id):
    print(f"User ID: {id}")
    return f"User ID: {id}"

# query parameter
@app.route("/find-item")
def find_item():
    name = request.args.get("name")
    price = request.args.get("price")
    seller = request.args.get("seller")
    print(f"Item Name: {name}")
    print(f"Item Price: {price}")
    print(f"Item Seller: {seller}")
    return f"Item Name: {name}, Item Price: {price}, Item Seller: {seller}"

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
