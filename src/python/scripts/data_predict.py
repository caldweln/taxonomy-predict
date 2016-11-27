import os
import pandas as pd
import pickle
from etc import config_openfoodfacts as config

fitted_model_path = os.path.join(config.fs['data_path'], config.fs['fitted_model'])
fitted_vectorizer_path = os.path.join(config.fs['data_path'], config.fs['fitted_vectorizer'])
uncategorized_docs_path = os.path.join(config.fs['data_path'], config.fs['uncategorized_docs'])
uncategorized_features_path = os.path.join(config.fs['data_path'], config.fs['uncategorized_features'])
prediction_results_path = os.path.join(config.fs['data_path'], config.fs['prediction_results'])

#
# Load data, vectorizer & model
#
features_df = pd.DataFrame(pd.DataFrame(pd.read_pickle(uncategorized_features_path))['feature_bag']).reset_index()
vectorizer = pickle.load( open( fitted_vectorizer_path, "rb" ) )
t = pickle.load( open( fitted_model_path, "rb" ) )

#
# Vectorize features
#
feature_vectors = pd.DataFrame(vectorizer.transform(features_df['feature_bag']).toarray())

#
# Predict & save data
#
preds = t.predict(feature_vectors)

docs_df = pd.DataFrame(pd.read_pickle(uncategorized_docs_path))
docs_df[config.db['update_update_field']] = pd.Series(preds).values
dict_results = docs_df[[config.db['update_filter_field'],config.db['update_update_field']]].to_dict('records')
pickle.dump(dict_results, open(prediction_results_path, 'wb'))
