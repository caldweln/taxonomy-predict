import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import  cross_validation, preprocessing, metrics
import decimal
from classes import taxonomy_predict as tp

data_path = 'data/'
model_file = 'fitted_model.p'
model_file_path = os.path.join(data_path, model_file)
vectorizer_file = 'fitted_vectorizer.p'
vectorizer_file_path = os.path.join(data_path, vectorizer_file)
tbc_feature_file = 'feature_data-tbc.p'
tbc_feature_file_path = os.path.join(data_path, tbc_feature_file)
pred_data_file = 'data_predicted.p'
pred_data_file_path = os.path.join(data_path, pred_data_file)

#
# Load data, vectorizer & model
#
features_df = pd.DataFrame(pd.DataFrame(pd.read_pickle(tbc_feature_file_path))['feature_bag']).reset_index()
vectorizer = pickle.load( open( vectorizer_file_path, "rb" ) )
t = pickle.load( open( model_file_path, "rb" ) )

#
# Strip out non-ASCII
#
ascii_filter = lambda x : x.encode('ascii','ignore')
features_df['feature_bag'] = features_df['feature_bag'].apply(ascii_filter)

#
# Vectorize features
#
feature_vectors = pd.DataFrame(vectorizer.transform(features_df['feature_bag']).toarray())

#
# Predict & save data
#
preds = t.predict(feature_vectors)

pickle.dump(preds, open(pred_data_file_path, 'wb'))
