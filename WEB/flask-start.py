from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/start')
def start():
    return render_template('tips.html')

if __name__ == '__main__':
    app.run(debug=True)
