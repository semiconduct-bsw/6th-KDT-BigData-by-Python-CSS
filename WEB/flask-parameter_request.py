from flask import Flask, render_template, request

app = Flask(__name__)

#1. path paramter
#2. query paramter
#3. request body

# path parameter /find-user/1234567890
@app.route("/find-user/<id>")
def find_user(id):
    print(f"User ID: {id}")
    return f"User ID: {id}"

# query parameter /find-item?name=lionkoreaofficial&price=1234567890
@app.route("/find-item", methods=["GET"])
def find_item():
    name = request.args.get("name")
    price = request.args.get("price")
    seller = request.args.get("seller")
    print(f"Item Name: {name}")
    print(f"Item Price: {price}")
    print(f"Item Seller: {seller}")
    return f"Item Name: {name}, Item Price: {price}, Item Seller: {seller}"

# request body
@app.route("/save-item", methods=["POST"])
def save_item():
    name = request.json.get("name")
    price = request.json.get("price")
    seller = request.json.get("seller")
    print(f"Item Name: {name}")
    print(f"Item Price: {price}")
    print(f"Item Seller: {seller}")
    return f"Item Name: {name}, Item Price: {price}, Item Seller: {seller}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/save-user")
def save_user():
    print("회원가입 완료 로직 수행")
    return "회원가입 완료"

if __name__ == '__main__':
    app.run(debug=True)

