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
        self.dataset = data_source.create_dataset()
        self.eval_epoch = data_source.create_dataset(eval_mode=True)

    def run(self, n_epochs=100):
        step = tf.constant(1, dtype=tf.int64)
        optimizer = tf.keras.optimizers.Adam()

        for epoch_counter in range(n_epochs):
            print(f'\nTraining epoch {epoch_counter}')
            with self.train_writer.as_default():
                step = self.epoch_training(step, optimizer, self.model, self.dataset)
            with self.eval_writer.as_default():
                self.epoch_eval(step, self.eval_epoch, self.model)

    def epoch_training(self, step: tf.Tensor, optimizer, model, dataset):

        for i, X in enumerate(dataset):  # (..., 8, 128, 5) one epoch
            summary = random.random() < 0.1
            with tf.GradientTape() as tape:
                loss = model.compute_loss(X, step, summary)
                print(f'Step {step}, Loss {loss}')
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            step += 1
        return step

    def epoch_eval(self, step, dataset, model):
        print('Evaluating')
        loss_acc = 0.0
        n = 0
        for i, X in enumerate(dataset):
            loss_acc += model.compute_loss(X, step)
            n += 1
        loss = loss_acc / n
        tf.summary.scalar('loss', loss, step=step)
        print(f'Eval :: loss={loss}')
