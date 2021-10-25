from flask import Flask, render_template, request
app = Flask(__name__)
import pickle

file = open('model.pkl', 'rb')

clf = pickle.load(file)
file.close()
@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        runnyNose = int(myDict['runnyNose'])
        breathingDifficulty = int(myDict['breathingDifficulty'])

        # code for inference
        input_features = [fever, pain, age, runnyNose, breathingDifficulty]
        infection_probability = clf.predict_proba([input_features])[0][1]
        print(infection_probability, "%")
        return render_template('show.html', inf=round(infection_probability*100))
    return render_template('index.html')
    # return 'Hello, World!' + str(infection_probability)


if __name__ == '__main__':
    app.run(debug=True)