
from data_processing.build_model import build_model
from data_processing.run_model import run_model


def get_recommendations(request, df, threshold_movie_list):
    username = request.args.get('username')
    # num_items = int(request.args.get('num_items'))
    num_items = 30
    
    training_data_rows = 200000
    model_df = df.head(training_data_rows)

    algo, user_watched_list = build_model(model_df, username)
    recs = run_model(username, algo, user_watched_list, threshold_movie_list, num_items)

    return recs