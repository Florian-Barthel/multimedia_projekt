from tqdm import tqdm
import numpy as np
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import dataSet
import dataUtil
import graph
import anchorgrid
from tensorboard import program

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir=logs'])
url = tb.launch()

f_map_rows = 10
f_map_cols = 10
scale_factor = 32.0
scales = [50, 80, 100, 150]
aspect_ratios = [1.0, 1.5, 2.0]
batch_size = 30
iou = 0.5
learning_rate = 0.001
iterations = 20
negative_percentage = 15

logs_directory = './logs/run6'

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)

anchor_grid = anchorgrid.anchor_grid(f_map_rows=f_map_rows,
                                     f_map_cols=f_map_cols,
                                     scale_factor=scale_factor,
                                     scales=scales,
                                     aspect_ratios=aspect_ratios)

training_dataset = dataSet.create("./dataset_mmp/train", anchor_grid, iou).batch(batch_size)
test_dataset = dataSet.create("./dataset_mmp/test", anchor_grid, iou, include_annotated_gts=True).batch(batch_size)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)

training_iterator = training_dataset.make_one_shot_iterator()
test_iterator = test_dataset.make_one_shot_iterator()

next_element = iterator.get_next()

with tf.Session(config=config) as sess:
    training_handle = sess.run(training_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())

    images, labels = next_element

    calculate_output = graph.output(images=images,
                                    num_scales=len(scales),
                                    num_aspect_ratios=len(aspect_ratios),
                                    f_rows=f_map_rows,
                                    f_cols=f_map_cols)

    calculate_loss, num_labels, num_random, num_weights, num_predicted = graph.loss(input_tensor=calculate_output,
                                                                                    labels=labels,
                                                                                    negative_percentage=negative_percentage)
    
    def optimize(my_loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        objective = optimizer.minimize(loss=my_loss)
        return objective


    optimize = optimize(calculate_loss)

    tf.summary.scalar('loss', calculate_loss)
    tf.summary.scalar('num_predicted', num_predicted)
    merged_summary = tf.summary.merge_all()

    graph_vars = tf.global_variables()
    for var in tqdm(graph_vars):
        try:
            sess.run(var)
        except:
            print('found uninitialized variable {}'.format(var.name))
            sess.run(tf.initialize_variables([var]))

    # TensorBoard graph summary
    log_writer = tf.summary.FileWriter(logs_directory, sess.graph)
    progress_bar = tqdm(range(iterations))
    for i in progress_bar:

        loss, labels, random, weights, predicted, _, summary = sess.run([calculate_loss, num_labels, num_random, num_weights, num_predicted, optimize, merged_summary], feed_dict={handle: training_handle})

        description = ' loss:' + str(np.around(loss, decimals=5)) + ' num_labels: ' + str(
            labels) + ' num_random: ' + str(random) + ' num_weights: ' + str(weights) + ' num_predicted: ' + str(
            predicted)
        progress_bar.set_description(description, refresh=True)
        # TensorBoard scalar summary
        log_writer.add_summary(summary, i)


    images_result, output_result = sess.run([images, calculate_output], feed_dict={handle: test_handle})

    for i in range(np.shape(images_result)[0]):
        image = Image.fromarray(((images_result[i] + 1) * 127.5).astype(np.uint8), 'RGB')
        image.resize((320*4, 320*4), Image.ANTIALIAS).save('test_images/{}_gts.jpg'.format(i))

        # dataUtil.draw_bounding_boxes(image=image,
        #                      annotation_rects=dataUtil.convert_to_annotation_rects_label(anchor_grid, labels_result[i]),
        #                      color=(0, 255, 255))

        # image.resize((320*4, 320*4), Image.ANTIALIAS).save('test_images/{}_labels.jpg'.format(i))

        dataUtil.draw_bounding_boxes(image=image,
                                     annotation_rects=dataUtil.convert_to_annotation_rects_output(anchor_grid, output_result[i]),
                                     color=(0, 0, 255))
        
        image.resize((320*4, 320*4), Image.ANTIALIAS).save('test_images/{}_estimates.jpg'.format(i))