from joblib import load
import json


clf = load('my_model.joblib') 


def hello_world(request):


    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    request_json = request.get_json()

    input_data = [request_json['sepal length'], 
                  request_json['sepal width' ],
                  request_json['petal length' ],
                  request_json['petal width' ]]

    print(input_data)

    return ( json.dumps({'prediction': clf.predict([input_data])[0]}), 200, headers)
