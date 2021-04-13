from joblib import load


clf = load('my_model.joblib') 


def hello_world(request):

    print(clf)
    request_json = request.get_json()

    request.args.get('sepal length')

    input_data = [ request.args.get('sepal length'), 
                   request.args.get('sepal width'),
                  request.args.get('petal length'),
                  request.args.get('petal width')]

    print(input_data)

    return clf.predict([input_data])[0]
