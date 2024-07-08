# Get "raw" datasets
echo -e "--- AMAZON ---"
python data/amazon/process_amazon.py
python data/preprocess_data.py --filepath data/amazon/amazon_movies-tv_test_orig.joblib
cp data/amazon/amazon_movies-tv_test_orig_preprocessed.joblib data/sa_data/amazon_movies-tv.joblib

echo -e "\n--- IMDB ---"
python data/imdb/process_imdb.py
python data/preprocess_data.py --filepath data/imdb/imdb_test_orig.joblib
cp data/imdb/imdb_test_orig_preprocessed.joblib data/sa_data/imdb_test.joblib

echo -e "\n--- YELP ---"
python data/yelp/process_yelp.py
python data/preprocess_data.py --filepath data/yelp/yelp_test_orig.joblib
cp data/yelp/yelp_test_orig_preprocessed.joblib data/sa_data/yelp_test.joblib

echo -e "\n--- MOVIES ---"
python data/movies/process_movie_rationales.py
python data/preprocess_data.py --filepath data/movies/movies_dev-test_orig.joblib
cp data/movies/movies_dev-test_orig_preprocessed.joblib data/sa_data/movies_dev-test.joblib

echo -e "\n--- SST ---"
python data/sst/download_and_process_sst.py
python data/sst/process_sst.py
python data/preprocess_data.py --filepath data/sst/sst_train_orig.joblib
python data/preprocess_data.py --filepath data/sst/sst_dev_orig.joblib
python data/preprocess_data.py --filepath data/sst/sst_test_orig.joblib
cp data/sst/sst_train_orig_preprocessed.joblib data/sa_data/sst_train.joblib
cp data/sst/sst_dev_orig_preprocessed.joblib data/sa_data/sst_dev.joblib
cp data/sst/sst_test_orig_preprocessed.joblib data/sa_data/sst_test.joblib

