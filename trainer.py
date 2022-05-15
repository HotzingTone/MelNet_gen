import tensorflow as tf
import random


class Trainer(object):
    """
    Trains and evaluates the model
    """
    def __init__(self, work_dir, data_source, model):
        self.work_dir = work_dir
        self.model = model
        self.data_source = data_source
        self.train_writer = tf.summary.create_file_writer(f'{work_dir}/train')
        self.eval_writer = tf.summary.create_file_writer(f'{work_dir}/eval')
        self.train_epoch = data_source.create_dataset()
        self.eval_epoch = data_source.create_dataset(eval_mode=True)

    def run(self, n_epochs=100):
        global_step = tf.constant(1, dtype=tf.int64)
        optimizer = tf.keras.optimizers.Adam()

        for epoch_counter in range(n_epochs):
            print(f'\nTraining epoch {epoch_counter}')
            with self.train_writer.as_default():
                global_step = self.epoch_training(global_step, optimizer, self.model, self.train_epoch)
            with self.eval_writer.as_default():
                self.epoch_eval(global_step, self.eval_epoch, self.model)

    def epoch_training(self, global_step: tf.Tensor, optimizer, model, epoch) -> int:
        for i, data in enumerate(epoch):  # (..., 8, 128, 5) one epoch
            do_summary = random.random() < 0.1
            with tf.GradientTape() as tape:
                if do_summary:
                    loss = model.compute_loss_with_summaries(data['inputs'], data['targets'],
                                                             global_step)  # (8, 128, 5)
                    tf.summary.scalar('loss', loss, step=global_step)
                else:
                    loss = model.compute_loss(data['inputs'], data['targets'])  # (8, 128, 5)
                print(f'Step {global_step}, Loss {loss}')
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            global_step = global_step + 1
        return global_step

    def epoch_eval(self, global_step: int, epoch, model):
        print('Evaluating')
        loss_acc = 0.0
        n = 0
        for i, data in enumerate(epoch):
            loss_acc += model.compute_loss(data['inputs'], data['targets'])
            n += 1
        loss = loss_acc / n
        tf.summary.scalar('loss', loss, step=global_step)
        print(f'Eval :: loss={loss}')
