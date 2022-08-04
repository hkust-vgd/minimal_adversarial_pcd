import sys
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import importlib.machinery
from pyntcloud import PyntCloud
import tf_nndistance
import pandas as pd


class StatsValue(object):
    def __init__(self):
        self.original_pred_success = 0
        self.attack_success = 0
        self.max_chamfers = []
        self.max_hausdorff = []
        self.modified_number = []

    def print(self):
        print("mean max chamfer: ", np.mean(self.max_chamfers))
        print("mean max hausdorff: ", np.mean(self.max_hausdorff))
        print("mean mean_modified_number: ", np.mean(self.modified_number))

        print("original prediction success: ", self.original_pred_success)
        print("attack success: ", self.attack_success)
        print(
            "attack success %: ",
            (self.attack_success / self.original_pred_success * 100),
        )


class Attack:
    def __init__(self, args):
        self.args = args
        self.modulename = importlib.machinery.SourceFileLoader(
            self.args.model_name, self.args.model_code_path
        ).load_module()
        self.model = importlib.import_module(self.args.model_name)

        print(" ----- Model loading ----- ")
        with tf.device("/gpu: " + str(self.args.gpu)):
            self.pointclouds_pl, self.labels_pl = self.model.placeholder_inputs(
                1, self.args.num_point
            )
            self.is_training_pl = tf.placeholder(tf.bool, shape=())

            self.ori_pred, self.ori_loss, self.ori_end_points = self.model_loss_fn(
                self.pointclouds_pl, self.labels_pl, self.is_training_pl
            )

            self.bound = tf.placeholder(tf.float32, shape=[2, 3])
            self.class_loss_weight = tf.placeholder(shape=[1], dtype=tf.float32)
            self.count_weight = tf.placeholder(shape=[1], dtype=tf.float32)
            self.dist_weight = tf.placeholder(shape=[1], dtype=tf.float32)
            self.lr_attack = tf.placeholder(dtype=tf.float32)

            self.count_vector = tf.get_variable(
                name="count",
                shape=[self.args.num_point, 1],
                initializer=tf.ones_initializer(),
            )
            self.pert = tf.get_variable(
                name="pert",
                shape=[1, self.args.num_point, 3],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
            )
            self.clip_count_vector = tf.clip_by_value(self.count_vector, 0, 1)

            # add clipping
            self.clip_pert = tf.clip_by_value(self.pert, self.bound[1], self.bound[0])

            if self.args.attack_type == "perturbation":
                self.final_pert = tf.multiply(
                    tf.tile(self.clip_count_vector, [1, 3]), self.clip_pert
                )
                self.adversarial_pl = self.pointclouds_pl + self.final_pert

            elif self.args.attack_type == "addition":
                self.final_pert = tf.multiply(
                    tf.tile(self.clip_count_vector, [1, 3]), self.clip_pert
                )
                self.mask = tf.greater(self.clip_count_vector, 0)
                self.candidate_points = (
                    tf.identity(self.pointclouds_pl) + self.final_pert
                )
                self.added_points = [
                    tf.boolean_mask(
                        self.candidate_points, tf.reshape(self.mask, [1, 1024])
                    )
                ]
                self.adversarial_pl = tf.concat(
                    [self.pointclouds_pl, self.added_points], axis=1
                )

            dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(
                self.adversarial_pl, self.pointclouds_pl
            )

            ch_dists_forward = tf.reduce_mean(dists_forward, axis=1)
            ch_dists_backward = tf.reduce_mean(dists_backward, axis=1)
            self.ch_max = tf.maximum(ch_dists_forward, ch_dists_backward)

            h_dists_forward = tf.reduce_max(dists_forward, axis=1)
            h_dists_backward = tf.reduce_max(dists_backward, axis=1)
            self.h_max = tf.maximum(h_dists_forward, h_dists_backward)
            h_dists = self.h_max

            self.pred, loss, end_points = self.model_loss_fn(
                self.adversarial_pl, self.labels_pl, self.is_training_pl
            )
            self.misclassification_loss = self.model.get_misclass_loss(
                self.pred, self.labels_pl
            )
            self.minimum_loss = tf.multiply(
                self.count_weight, tf.norm(self.clip_count_vector, ord=1)
            )
            self.distance_loss = tf.multiply(self.dist_weight, h_dists)

            attack_optimizer = tf.train.AdamOptimizer(self.lr_attack)

            adv_loss = (
                tf.multiply(self.class_loss_weight, self.misclassification_loss)
                + self.minimum_loss
                + self.distance_loss
            )

            self.attack_op = attack_optimizer.minimize(
                adv_loss, var_list=[self.pert, self.count_vector]
            )

            vl=tf.global_variables()
            vl=[x for x in vl if ('pert' or 'count') not in x.name]
            v2=[]
            for x in vl:
                if ('pert' not in x.name) and ('count' not in x.name):
                    v2.append(x)
            self.saver = tf.train.Saver(v2)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, self.args.ckpt_path)
        print(" ----- Model loaded ----- ")


    def model_loss_fn(self, x, t, is_training):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            y, end_points = self.model.get_model(
                x, is_training, num_class=self.args.num_class
            )
            if t == None:
                loss = None
            else:
                loss = self.model.get_loss(y, t, end_points)
        return y, loss, end_points


    def get_bound(self, points, clip_bound):
        max_bound = np.amax(points, axis=0) + clip_bound
        min_bound = np.amin(points, axis=0) - clip_bound
        return [max_bound, min_bound]


    def save_ply(self, index, ori_pc, adv_pc, pred_label, adv_label):
        base_path = os.path.join(self.args.save_dir)
        original_pc_path = os.path.join(base_path, str(index)+'_ori_' +  str(pred_label) + '_' + str(adv_label) + '.ply')
        adversarial_pc_path = os.path.join(base_path, str(index)+'_attack_' +  str(pred_label) + '_' + str(adv_label) + '.ply')

        ori_cloud = PyntCloud(pd.DataFrame(ori_pc, columns=["x", "y", "z"]))
        ori_cloud.to_file(original_pc_path)

        adv_cloud = PyntCloud(pd.DataFrame(adv_pc, columns=["x", "y", "z"]))
        adv_cloud.to_file(adversarial_pc_path)


    def save_stats(self, stats):
        with open(os.path.join(self.args.save_dir, "stats.csv"), "w") as f:
            f.write("attack type: {}\n".format(self.args.attack_type))
            f.write("iter: {}\n".format(self.args.iter))
            f.write("lr_attack: {}\n".format(self.args.lr_attack))
            f.write("clip: {}\n\n".format(self.args.clip))

            f.write("count_lambda: {}\n".format(self.args.count_weight))
            f.write("h_dist_lambda: {}\n".format(self.args.h_dist_weight))
            f.write("class_loss_lambda: {}\n\n".format(self.args.class_loss_weight))

            f.write("original prediction success: {}\n".format(stats.original_pred_success))
            f.write("attack success: {}\n".format(stats.attack_success))
            f.write("attack success %: {}\n".format(stats.attack_success/stats.original_pred_success*100))
            
            f.write('mean_chamfer: %10.9f\n\n'%(np.mean(np.array(np.mean(stats.max_chamfers)))))
            f.write('mean_hausdorff: %10.9f\n\n'%(np.mean(np.array(np.mean(stats.max_hausdorff)))))
            f.write('mean_number: %10.2f\n\n'%(np.mean(np.array(np.mean(stats.modified_number)))))


    def test(self, data, labels):
        stats = StatsValue()

        for i, ori_pc in tqdm(enumerate(data)):
            ori_label = labels[i]

            feed_dict = {
                self.pointclouds_pl: [ori_pc],
                self.is_training_pl: False,
                self.labels_pl: [ori_label],
            }

            # ---- check prediction ----
            pred_val = self.sess.run(self.ori_pred, feed_dict=feed_dict)
            original_pred_label = np.squeeze(np.argmax(pred_val, axis=-1))

            if original_pred_label == ori_label:
                # ----- matched ---- 
                stats.original_pred_success += 1

                # ----- initialization vector A -----
                zeros = np.random.uniform(0, 0.00001, size=(self.args.num_point, 1))
                ones = np.random.uniform(0.9999, 1, size=(self.args.num_point, 1))
                init_cv = np.concatenate((zeros, ones), axis=0)
                np.random.shuffle(init_cv)
                init_cv = init_cv[
                    np.random.choice(self.args.num_point * 2, self.args.num_point)
                ]

                _ = self.sess.run(tf.assign(self.count_vector, init_cv))
                _ = self.sess.run(
                    tf.assign(
                        self.pert,
                        tf.truncated_normal(
                            [1, self.args.num_point, 3], mean=0, stddev=0.0000001
                        ),
                    )
                )

                
                # ----- attack ---- 
                feed_dict = {
                    self.pointclouds_pl: [ori_pc],
                    self.is_training_pl: False,
                    self.labels_pl: [ori_label],
                    self.lr_attack: self.args.lr_attack,
                    self.class_loss_weight: [self.args.class_loss_weight],
                    self.count_weight: [self.args.count_weight],
                    self.dist_weight: [self.args.h_dist_weight],
                    self.bound: self.get_bound(ori_pc, self.args.clip)
                }
                for j in range(self.args.iter):
                    _ = self.sess.run([self.attack_op], feed_dict=feed_dict)

                    pred_val, adv_pc, h_max, ch_max = self.sess.run(
                        [self.pred, self.adversarial_pl, self.h_max, self.ch_max],
                        feed_dict=feed_dict,
                    )
                    adv_pred_label = np.squeeze(np.argmax(pred_val, axis=-1))
                    adv_pc = np.squeeze(adv_pc)

                stats.max_chamfers.append(ch_max[0])
                stats.max_hausdorff.append(h_max[0])

                if self.args.attack_type == 'perturbation':
                    difference = np.linalg.norm(adv_pc - ori_pc, axis=1, ord=2)
                    difference_pc = difference > 0
                    diff_num = np.sum(difference_pc[:])
                    stats.modified_number.append(diff_num)
                else:
                    stats.modified_number.append(adv_pc.shape[0] - ori_pc.shape[0])


                # ---- if attack success ----
                if adv_pred_label != ori_label:
                    stats.attack_success += 1

                    if self.args.save_ply:
                        self.save_ply(i, ori_pc, adv_pc, original_pred_label, adv_pred_label)

                stats.print()

        stats.print()
        self.save_stats(stats)
        print(" ----- finish ----- ")