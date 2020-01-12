import os
import argparse as ap
import fileUtil
from datetime import datetime
import tensorflow as tf

model_path = ''

# Trains the specified model
# if no arguments are given: trains the latest model
if __name__ == '__main__':
    cparse = ap.ArgumentParser(prog='train_model', description='Trains an existing model')
    cparse.add_argument('--model', help='Path to the model', default='')

    args = cparse.parse_args()
    if args.model == '':
        models = os.listdir(fileUtil.model_directory)
        if not models:
            print("No models found at " + fileUtil.model_directory)
            exit(-1)
        model_dates = []
        for m in models:
            model_dates.append(datetime.strptime(m, '%d-%m-%Y_%H-%M-%S'))
        model_dates.sort(reverse=True)
        model_path = fileUtil.model_directory + datetime.strftime(model_dates[0], '%d-%m-%Y_%H-%M-%S')
        print(model_path)
    else:
        model_path = args.model

    if not model_path.endswith('/'):
        model_path += '/'

    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        saver = fileUtil.load_model(model_path=model_path, sess=sess)
        graph = tf.get_default_graph()
        graph_vars = tf.global_variables()
        for var in graph_vars:
            print(var)
        fileUtil.save_model(saver=saver, sess=sess)
