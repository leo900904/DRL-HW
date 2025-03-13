from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_grid', methods=['POST'])
def update_grid():
    data = request.json
    return jsonify({"message": "Grid updated successfully!", "data": data})

if __name__ == '__main__':
    app.run(debug=True)
