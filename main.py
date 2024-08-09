import os
from src.image_retrieval import *
from src.plot_results import *
import src.config

def query_images(query_path, get_similarity, method_name, reverse=False):
    query_path = os.path.join(config.TEST_DIR, query_path)
    query, ls_path_score = get_similarity(config.TRAIN_DIR, query_path, config.IMAGE_SIZE)
    plot_results(query_path, ls_path_score, reverse=reverse, save=True, filename=method_name)

def run_retrieval():
    query_path_1 = 'Orange_easy/0_100.jpg'
    query_images(query_path_1, get_l1_score, 'l1')
    query_images(query_path_1, get_l2_score, 'l2')
    query_images(query_path_1, get_cosine_similarity_score, 'cosine_similarity', True)
    query_images(query_path_1, get_correlation_coefficient_score, 'correlation_coefficient', True)
    
    query_path_2 = 'African_crocodile/n01697457_18534.JPEG'
    query_images(query_path_2, get_l1_score, 'l1')
    query_images(query_path_2, get_l2_score, 'l2')
    query_images(query_path_2, get_cosine_similarity_score, 'cosine_similarity', True)
    query_images(query_path_2, get_correlation_coefficient_score, 'correlation_coefficient', True)
    

if __name__ == "__main__":
    run_retrieval()
