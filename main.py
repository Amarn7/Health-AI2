from flask import Flask,render_template,request,url_for,Response

#EDA Packages
import pandas as pd
import numpy as np

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

@app.route("/")
def chech():
    return render_template('chech.html')

@app.route("/predict", methods=['GET'])
def predict():
	# return "inside predict.."
	url = "https://raw.githubusercontent.com/Amarn7/Data-Science/master/newdata.csv"
	#url = "https://raw.githubusercontent.com/Jcharis/Machine-Learning-Web-Apps/master/Youtube-Spam-Detector-ML-Flask-App/YoutubeSpamMergedData.csv"
	df_data = pd.read_csv(url)
	print(df_data)
	df_x = df_data.drop('CLASS',axis = 1)
	df_y = df_data['CLASS']

    # Extract Feature With CountVectorizer

	# corpus = df_x
	# cv = CountVectorizer(lowercase=False)
	# X = cv.fit_transform(corpus)

	# Fit the Data

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33)

	# Bayes Classifier

	'''from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(X_train,y_train)'''
	from sklearn.svm import SVC
	s = SVC()
	s.fit(X_train,y_train)
	#ypred = s.predict(X_test)
	#clf.score(X_test,y_test)
	#my_prediction= clf.predict('comment')
	#x = np.reshape(my_prediction, (len(my_prediction),-1))
	#data = my_prediction.reshape(1,-1)

	if request.method == 'GET':
		comment = request.GET('comment')
		#print(comment)
		test = pd.DataFrame(data = pd.Series(191),columns = ['CONTENT'])
		#print(comment)
		my_prediction = s.predict(test)

	return render_template('result.html' , prediction = my_prediction)
	#return render_template("result.html")

if __name__ == '__main__':
	app.run(debug=True)
	
 
