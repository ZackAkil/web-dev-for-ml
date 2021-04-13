from joblib import load


clf = load('my_model.joblib') 


def hello_world(request):

    print(clf)
    request_json = request.get_json()

    input_data = [request_json['sepal length'], 
                  request_json['sepal width' ],
                  request_json['petal length' ],
                  request_json['petal width' ]]

    return clf.predict([input_data])[0]
