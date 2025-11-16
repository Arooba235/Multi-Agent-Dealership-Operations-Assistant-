from flask import Flask, render_template, request, jsonify
from helper import graph
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get data from frontend
    user_query = request.form.get('query')

    try:
        response = graph.invoke({"messages":user_query})
        response = response["messages"][-1].content
    except Exception as e:
        response = f"Error: {str(e)}"
    
    # Return the response as JSON
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
