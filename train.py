import sys
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data_loader import data_loader


def train(gen_feed, dis_feed, gen_output, dis_out_real, dis_out_fake, dis_labels_real, dis_labels_fake, parameters):
    eps = 1e-12
    dis_loss_real = -(dis_labels_real * tf.log(dis_out_real + eps) + (1.0 - dis_labels_real) * tf.log(1.0 - dis_out_real + eps))
    dis_loss_fake = -(dis_labels_fake * tf.log(dis_out_fake + eps) + (1.0 - dis_labels_fake) * tf.log(1.0 - dis_out_fake + eps))
    dis_loss = tf.reduce_mean(dis_loss_real + dis_loss_fake, name='dis_loss')
    gen_loss = -tf.reduce_mean(dis_labels_fake * tf.log(dis_out_fake + eps) + (1.0 - dis_labels_fake) * tf.log(1.0 - dis_out_fake + eps), name='gen_loss')

    dis_optimizer = tf.train.AdamOptimizer(learning_rate=parameters['dis_learning_rate'], beta1=parameters['dis_beta1'])
    gen_optimizer = tf.train.AdamOptimizer(learning_rate=parameters['gen_learning_rate'], beta1=parameters['gen_beta1'])

    dis_var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    print("\nNo of trainable discriminator variables: {}".format(len(dis_var_list)))
    print(dis_var_list)
    gen_var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    print("No of trainable generator variables: {}\n".format(len(gen_var_list)))
    print(gen_var_list)

    dis_train_op = dis_optimizer.minimize(loss=dis_loss, var_list=dis_var_list, name='dis_train_step')
    gen_train_op = gen_optimizer.minimize(loss=gen_loss, var_list=gen_var_list, name='gen_train_step' )

    saver = tf.train.Saver(max_to_keep=2)
    data = data_loader(parameters)
    data.load_data()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        '''saver.restore(sess, parameters['checkpoint_path']+ "ckpt-8")'''
        for i in range(1, parameters['epochs']+1):
            data.shuffle_data()
            iterations = int(len(data.images)/(parameters['batch_size'] * parameters['dis_train_steps']))
            d_loss_avg = 0
            g_loss_avg = 0
            for j in range(1, iterations+1):
                d_loss_k = 0
                g_loss_l = 0
                for k in range(1, parameters['dis_train_steps']+1):

                    real_images, dis_lab = data.next_batch()
                    real_images = np.asarray(real_images, dtype=np.float32) 
                    real_images = (real_images / 127.5) - 1.0
                    gen_noise, _ = data.noise()
                    '''index = np.arange(2 * parameters['batch_size'])
                    np.random.shuffle(index)'''

                    d_loss, _ = sess.run([dis_loss, dis_train_op], feed_dict={gen_feed:gen_noise,
                                                                              dis_feed:real_images,
                                                                              dis_labels_real:dis_lab[:parameters['batch_size']],
                                                                              dis_labels_fake:dis_lab[parameters['batch_size']:]})                    
                    d_loss_k = ((d_loss_k * (k-1)) + d_loss) / k
                d_loss_avg = ((d_loss_avg * (j-1)) + d_loss_k) / j

                for l in range(1, parameters['gen_train_steps']+1):

                    gen_noise, fake_lab = data.noise(2) 
                    g_loss, _ = sess.run([gen_loss, gen_train_op], feed_dict={gen_feed: gen_noise,
                                                                              dis_labels_fake: fake_lab})
                    g_loss_l = ((g_loss_l * (l-1)) + g_loss) / l
                g_loss_avg = ((g_loss_avg * (j-1)) + g_loss_l) / j

                #plot_loss(i, j, d_loss_avg, g_loss_avg)
                sys.stdout.write('\rEpoch={0} iteration={1}/{2} gen_loss={3:.6f}, dis_loss={4:.6f}'.format(i, j, iterations, g_loss_avg, d_loss_avg))
                sys.stdout.flush()
            
            saver.save(sess, parameters['checkpoint_path']+ "ckpt-" + str(i))
            print("  Model saved for epoch:{}".format(i))
            gen_images, = sess.run([gen_output], feed_dict={gen_feed: gen_noise})            
            save_generated_images(parameters['img_save_dir'], gen_images, i)

def save_generated_images(save_dir, generated_images, epoch):

    plt.figure(figsize=(8, 8), num=2)
    gs1 = gridspec.GridSpec(8, 8)
    gs1.update(wspace=0, hspace=0)

    for i in range(64):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        image = (generated_images[i, :, :, :] + 1) * 127.5
        fig = plt.imshow(image.astype(np.uint8))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    img_name = save_dir + 'generatedSamples_epoch' + str(epoch) + '.png'

    plt.savefig(img_name, bbox_inches='tight', pad_inches=0)

def plot_loss(epoch, iterations, dis_loss, gen_loss):
    plt.figure(1)
    plt.plot(iterations, gen_loss, color='green', label='Generator Loss')
    plt.plot(iterations, dis_loss, color='blue', label='Discriminator Loss')
    plt.title("DCGAN Training Loss Plot")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    if epoch == 1:
        plt.legend()
    plt.pause(0.0000000001)
    plt.show()
    plt.savefig('TrainingLossPlot.jpg')

def infer(gen_output, ged_feed, parameters):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.restore(sess, tf.train.latest_checkpoint(parameters['checkpoint_path']))
        data = data_loader(parameters)
        gen_noise = data. noise()
        gen_images, = sess.run([gen_output], feed_dict={gen_feed: gen_noise})

        for i in range(gen_images.shape[0]):
            write_img = gen_images[i]
            cv2.imwrite(str("/content/drive/My Drive/artgan/gen_images/gen_img_in-"+str(i)+".jpg"), write_img)

