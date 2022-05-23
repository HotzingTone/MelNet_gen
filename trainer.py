import tensorflow as tf
import random
import math


class Trainer(object):

    def __init__(self, work_dir, source, model):
        self.work_dir = work_dir
        self.model = model
        self.writer_train = tf.summary.create_file_writer(f'{work_dir}/train')
        self.writer_eval = tf.summary.create_file_writer(f'{work_dir}/eval')
        self.data_train = source.get_data()
        self.data_eval = source.get_data(eval_mode=True)

    def run(self, n_epochs=10,
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.0002, global_clipnorm=0.25)
            ):
        step = 1
        for epoch_counter in range(n_epochs):
            print(f'\nTraining epoch {epoch_counter}')
            with self.writer_train.as_default():
                step = self.epoch_train(step, optimizer, self.model, self.data_train)
            with self.writer_eval.as_default():
                self.epoch_eval(step, self.model, self.data_eval)

    def epoch_train(self, step: tf.Tensor, optimizer, model, data_train):
        for i, X in enumerate(data_train):
            summary = random.random() < 0.1
            with tf.GradientTape() as tape:
                loss = model.compute_loss(X, step, summary)
                while math.isnan(loss):
                    print(f'Got NaN loss, retraining...')
                    model.load_weights('./checkpoints/params.tf')
                    loss = model.compute_loss(X, step, summary)
                print(f'Step {step}, Loss {loss}')
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            step += 1
            model.save_weights('./checkpoints/params.tf', overwrite=True, save_format='tf')
        return step

    def epoch_eval(self, step, model, data_eval):
        print('Evaluating')
        loss_acc = 0.0
        n = 0
        for i, X in enumerate(data_eval):
            loss_acc += model.compute_loss(X, step)
            n += 1
        loss = loss_acc / n
        tf.summary.scalar('loss', loss, step=step)
        print(f'Eval :: loss={loss}')
