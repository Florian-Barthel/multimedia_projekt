import tensorflow as tf

adjustment = tf.constant([[[[1, 2, 3, 4], [4, 5, 5, 4]], [[1, 2, 3, 3], [4, 5, 5, 4]]],
                          [[[6, 2, 3, 4], [4, 5, 5, 4]], [[10, 3, 3, 12], [1, 5, 5, 4]]],
                          [[[6, 2, 3, 4], [4, 5, 5, 4]], [[10, 3, 3, 12], [1, 5, 5, 4]]]])
# (2, 2, 2, 4)
num_batch_size = tf.shape(adjustment)[0]
num_anchors = tf.shape(adjustment)[1]

labels = tf.ones([num_batch_size, num_anchors])

adjustment_sum = tf.reduce_sum(tf.abs(adjustment), axis=-1)

adjustment_sum_min = tf.math.argmin(adjustment_sum, axis=-1)
adjustment_sum_min_expand = tf.expand_dims(adjustment_sum_min, -1)

# arrange = tf.range(2)
# arrange_expand = tf.cast(tf.expand_dims(arrange, -1), tf.int64)

batch_range = tf.range(num_batch_size)
anchor_range = tf.range(num_anchors)
grid = tf.meshgrid(anchor_range, batch_range)
grid_expand_x = tf.cast(tf.expand_dims(grid[0], -1), tf.int64)
grid_expand_y = tf.cast(tf.expand_dims(grid[1], -1), tf.int64)
grid_concat = tf.concat([grid_expand_y, grid_expand_x], axis=-1)

concat = tf.concat([grid_concat, adjustment_sum_min_expand], axis=-1)

min_adjustments = tf.gather_nd(adjustment, concat)


mask = tf.cast(labels, tf.bool)
filtered_min_adjustments = tf.boolean_mask(min_adjustments, mask)

with tf.Session() as sess:
    debug_adjustment, debug_adjustment_sum, debug_adjustment_sum_min, debug_adjustment_sum_min_expand, debug_concat, debug_min_adjustments, debug_grid_expand_x, debug_grid_expand_y = sess.run(
        [adjustment, adjustment_sum, adjustment_sum_min, adjustment_sum_min_expand, concat, min_adjustments, grid_expand_x, grid_expand_y])
    print('a')


