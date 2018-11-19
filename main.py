import tensorflow as tf
import json
import argparse
import architecture
from train import train, infer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', type=str, default='/home/mdo2/Documents/artgan/parameters.json', help='model parameters file')
    parser.add_argument('--mode', type=bool, default=True, help=' True for training or False for inference')
    args = parser.parse_args()

    with open(args.parameters, 'r') as f:
        parameters = json.load(f)
    parameters['mode'] = args.mode

    gen_feed = tf.placeholder(dtype=tf.float32, shape=(None, parameters['noise_length']), name='gen_feed')
    dis_feed = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3), name='dis_feed')
    dis_labels_real = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='dis_labels_real')
    dis_labels_fake = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='dis_labels_fake')
    dis_feed_cond = tf.placeholder(dtype=tf.bool, name='dis_feed_cond')

    model = architecture.Model(parameters)
    gen_output = model.generator(gen_feed)
    dis_out_real = model.discriminator(dis_feed)
    dis_out_fake = model.discriminator(gen_output, reuse=True)     

    if args.mode == True: #Training
        train(gen_feed, dis_feed, gen_output, dis_out_real, dis_out_fake, dis_labels_real, dis_labels_fake, parameters)
    elif args.mode == False: #Inference 
        infer(model, gen_feed)


if __name__ == "__main__":
    main()
