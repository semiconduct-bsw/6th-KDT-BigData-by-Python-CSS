from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/basic')
def basic():
    return render_template('basic.html')

@app.route('/basik')
def basik():
    return render_template('basik.html')


@app.route('/start')
def user_set():
    return render_template('user-setting.html')

if __name__ == '__main__':
    app.run(debug=True) 
