from flask import Blueprint, request

# Blueprint 생성
bp = Blueprint('test', __name__, url_prefix='/test')

# (연습용 코드들 이동)
@bp.route("/find-user/<id>")
def find_user(id):
    print(f"User ID: {id}")
    return f"User ID: {id}"

@bp.route("/find-item", methods=["GET"])
def find_item():
    name = request.args.get("name")
    price = request.args.get("price")
    seller = request.args.get("seller")
    print(f"Item Name: {name}")
    print(f"Item Price: {price}")
    print(f"Item Seller: {seller}")
    return f"Item Name: {name}, Item Price: {price}, Item Seller: {seller}"

@bp.route("/save-item", methods=["POST"])
def save_item():
    name = request.json.get("name")
    price = request.json.get("price")
    seller = request.json.get("seller")
    print(f"Item Name: {name}")
    print(f"Item Price: {price}")
    print(f"Item Seller: {seller}")
    return f"Item Name: {name}, Item Price: {price}, Item Seller: {seller}"