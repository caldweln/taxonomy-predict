db = dict(
    db_conn_str = "mongodb://localhost:27017/",
    db_database = "off",
    db_table = "products",
    find_where_categorized = { \
        'categories_hierarchy':{'$exists':'true', '$ne':[]}, \
        'product_name':{'$exists':'true','$ne':''}, \
        'brands':{'$exists':'true','$ne':''}, \
        'quantity':{'$exists':'true','$ne':''}, \
        'ingredients_text':{'$exists':'true','$ne':''} \
        },
    find_where_uncategorized = { \
        'categories_hierarchy':{'$exists':'true', '$eq':[]}, \
        'product_name':{'$exists':'true','$ne':''}, \
        'brands':{'$exists':'true','$ne':''}, \
        'quantity':{'$exists':'true','$ne':''}, \
        'ingredients_text':{'$exists':'true','$ne':''} \
        },
    find_select_fields = {"_id":1,"product_name":1,"brands":1,"quantity":1,"ingredients_text":1,"categories_hierarchy":1},
    update_filter_field = "_id",
    update_update_field = "test_categories_hierarchy"

    )

fs = dict(
    data_path = "data/",
    categorized_docs = "categorized_docs.p",
    categorized_features = "categorized_features.p",
    uncategorized_docs = "uncategorized_docs.p",
    uncategorized_features = "uncategorized_features.p",
    fitted_model = "fitted_model.p",
    fitted_vectorizer = "fitted_vectorizer.p",
    prediction_results = "prediction_results.p"

)

op = dict(
    classifier_module='sklearn.linear_model',
    classifier_name='LogisticRegression',
    classifier_params={'C':1,'class_weight':'balanced'}
)
