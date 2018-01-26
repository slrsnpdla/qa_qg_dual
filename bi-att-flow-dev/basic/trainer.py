import tensorflow as tf

from basic.model import Model
from my.tensorflow import average_gradients

class MultiGPUTrainer(object):
    def __init__(self, config, models):
        model = models[0]
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        self.opt = tf.train.AdamOptimizer(config.init_lr)
        #self.opt = tf.train.GradientDescentOptimizer(config.init_lr)
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.summary = model.summary
        self.models = models
        losses = []
        grads_list = []
        for gpu_idx, model in enumerate(models):
            with tf.name_scope("grads_{}".format(gpu_idx)), tf.device("/{}:{}".format(config.device_type, gpu_idx)):
                loss = model.get_loss()
                grads = self.opt.compute_gradients(loss)
                losses.append(loss)
                grads_list.append(grads)

        # QA weight update
        self.loss = tf.add_n(losses)/len(losses)
        self.seq2seq_loss = model.get_qg_loss()
        self.gen_q = model.get_qg_sample()

        alpha = 0.5
        total_loss = alpha * self.loss + (1-alpha) * self.seq2seq_loss
        self.grads = self.opt.compute_gradients(total_loss)

        """
        self.capped_gvs_qa = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in self.grads
                              if "q_gen" not in var.name]
        
        self.capped_gvs_qg = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in self.grads
                              if "q_gen" in var.name]
        capped_gvs = self.capped_gvs_qa
        """
        #for grad, var in self.grads:
        #    print(grad, var)
        capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in self.grads if grad is not None]
        self.train_op = self.opt.apply_gradients(capped_gvs, global_step=self.global_step)

    def step(self, sess, batches, get_summary=False, is_gen=False):
        assert isinstance(sess, tf.Session)
        feed_dict = {}

        if is_gen:
            feed_dict.update(self.models[0].get_feed_dict(batches, True))
        else:
            for batch, model in zip(batches, self.models):
                _, ds = batch
                feed_dict.update(model.get_feed_dict(ds, True))

        if get_summary:
            gen_q_sample, loss, seq2seq_loss, summary, train_op = \
                sess.run([self.gen_q, self.loss, self.seq2seq_loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            gen_q_sample, loss, seq2seq_loss, train_op = \
                sess.run([self.gen_q, self.loss, self.seq2seq_loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, seq2seq_loss, summary, train_op, gen_q_sample
