from flask import Flask, request, render_template
from flask_app.predict import *
import joblib
from flask_app.live import * 
import pandas as pd
from pymongo import MongoClient


app = Flask(__name__)

@app.route('/', methods=['GET'])
def home(): 
    return ''' <p> clickityclick 
                <a href="/score">score</a> </p> '''

@app.route('/hello', methods=['GET'])
def hello_world():
    return ''' <h1> Hello, World!</h1> '''

@app.route('/form_example', methods=['GET'])
def form_display():
    return ''' <form action="/string_reverse" method="POST">
                <input type="text" name="some_string" />
                <input type="submit" />
            </form>
            '''

# HTML Code
# <input type="text" name="chat_in" maxlength="500" ><!-- Submit button -->
# <input type="submit" value="Submit" method="get" >


@app.route('/string_reverse', methods=['POST'])
def reverse_string():
    text = str(request.form['some_string'])
    reversed_string = text[-1::-1]
    return ''' output: {}  '''.format(reversed_string)

@app.route('/score', methods=['GET']) #,'POST'
def return_predictions():
    raw_data = client.get_data()
    raw_df = pd.DataFrame(raw_data)
    # threshold = request.args['gimme_threshold']
    new_df = predictions(raw_df, model,threshold=.5)
    table=new_df.to_html()
    json_entry = new_df.to_dict('records')
    database_table.insert_one(json_entry[0])
    Html_file= open("flask_app/templates/table.html","w")
    Html_file.write(table)
    Html_file.close()

    return render_template('table.html') #data=table)

@app.route('/database', methods=['GET'])
def return_database():
    # raw_data = client.get_data()
    # raw_df = pd.DataFrame(raw_data)
    # new_df = predictions(raw_df, model)
    # table=new_df.to_html()
    mongo_entries = []
    for row in database_table.find():
        mongo_entries.append(row)
    db = pd.DataFrame(mongo_entries)
    db_table = db.to_html()
    Html_file= open("flask_app/templates/database.html","w")
    Html_file.write(db_table)
    Html_file.close()

    return render_template('database.html')

if __name__ == '__main__':

    client = EventAPIClient()
    M_client = MongoClient('localhost', 27017)
    db = M_client['flask']
    database_table = db['transaction2']
    #Clean data
    model = joblib.load('flask_app/gb.pkl')
    #Predict with the model
    # app.run(host='0.0.0.0', port=8080, debug=True)
    #connect  to mongo db

    app.run(debug=False)

    