from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def main():
    return 

if __name__ == "__main__":
    app.run(debug=True, port=5000)



# about the model and how it does its predictions
# data visualization -- charts and graphs of sample data vs cleaned up output
#   plotly for interactive charts — much nicer than matplotlib for a presentation
#   seaborn or plotly for the heatmaps
# recommendations: what to do if you have alzheimers, suspect you do, to prevent, who you can contact

# upload files/folders
# select model in sidebar (maybe)