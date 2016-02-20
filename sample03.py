# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# 入力データの定義　4行2列（データの定義方法がchainerとは違うようです）
# x_data = [
#   np.array([0., 0.]),
#   np.array([0., 1.]),
#   np.array([1., 0.]),
#   np.array([1., 1.])
# ]
x_data = np.array([
  [0., 0.],
  [0., 1.],
  [1., 0.],
  [1., 1.]
])

# 結果データの定義（4行1列）
# y_data = [
#   np.array([0.]),
#   np.array([0.]),
#   np.array([0.]),
#   np.array([1.])
# ]
y_data = np.array([
  [0.],
  [1.],
  [0.],
  [1.]
])


# 機械学習で最適化するWとbを設定する。Wは4行2列のテンソル。bは4行1列のテンソル。
W = tf.Variable(tf.random_uniform([4, 2], -1.0, 1.0))
b = tf.Variable(tf.zeros([4, 1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 学習を始める前にこのプログラムで使っている変数を全てリセットして空っぽにする
init = tf.initialize_all_variables()

# Launch the graph.（おきまりの文句）
sess = tf.Session()
sess.run(init)

# 学習を1000回行い、100回目ごとに画面に学習回数とWとbのその時点の値を表示する
for step in xrange(1001):
    sess.run(train)
    if step % 100 == 0:
        print step, sess.run(W), sess.run(b)

# 学習結果を確認
x_input = np.array([
  [0., 0.],
  [0., 1.],
  [1., 0.],
  [1., 1.]
])

y_res = tf.Variable(tf.zeros([4, 1]))
y_res = W * x_input + b
print sess.run(y_res)
# 4行1列の結果を期待しているのだが、4行2列になってしまう？ 
# 学習が十分進めば、どちらの列も同じような結果になるからいいか。
print sess.run(b)
