import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, ReLU, Add,\
    Embedding, MaxPooling1D, Multiply, Dropout, Softmax, Lambda
from keras.initializers import RandomUniform, Constant
from keras.regularizers import l2
from keras.models import Model
import numpy as np


def reshape_tensor(x, shape):
    return tf.reshape(x, shape)

def CNN_text_processer(input_u, input_i, user_vocab_size, item_vocab_size, embed_word_dim, random_seed,
                       num_filters, filter_size, review_num_u, review_len_u, review_num_i, review_len_i,
                       initW_u, initW_i):

    x_u = Embedding(user_vocab_size, embed_word_dim, embeddings_initializer=Constant(
        initW_u), name='user_text_embed')(input_u)
    x_i = Embedding(item_vocab_size, embed_word_dim, embeddings_initializer=Constant(
        initW_i), name='item_text_embed')(input_i)

    x_u = Lambda(reshape_tensor, arguments={'shape': [-1, embed_word_dim, review_len_u, 1]})(x_u)
    x_i = Lambda(reshape_tensor, arguments={'shape': [-1, embed_word_dim, review_len_i, 1]})(x_i)
    # x_u = Lambda(tf.reshape)(x_u, [-1, embed_word_dim, review_len_u, 1])
    # x_i = Lambda(tf.reshape)(x_i, [-1, embed_word_dim, review_len_i, 1])

    x_u = Conv2D(num_filters, kernel_size=(embed_word_dim, filter_size), padding='valid', activation='relu',
                 use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='user_text_conv')(x_u)
    x_i = Conv2D(num_filters, kernel_size=(embed_word_dim, filter_size), padding='valid', activation='relu',
                 use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='item_text_conv')(x_i)

    x_u = Lambda(reshape_tensor, arguments={'shape': [-1, x_u.shape[-2], x_u.shape[-1]]})(x_u)
    x_i = Lambda(reshape_tensor, arguments={'shape': [-1, x_i.shape[-2], x_i.shape[-1]]})(x_i)
    # x_u = tf.reshape(x_u, [-1, x_u.shape[-2], x_u.shape[-1]])
    # x_i = tf.reshape(x_i, [-1, x_i.shape[-2], x_i.shape[-1]])

    x_u = MaxPooling1D(pool_size=review_len_u - 2, padding='valid', name='user_text_pool')(x_u)
    x_i = MaxPooling1D(pool_size=review_len_i - 2, padding='valid', name='item_text_pool')(x_i)

    x_u = Lambda(reshape_tensor, arguments={'shape': [-1, review_num_u, num_filters]})(x_u)
    x_i = Lambda(reshape_tensor, arguments={'shape': [-1, review_num_i, num_filters]})(x_i)
    # x_u = tf.reshape(x_u, [-1, review_num_u, num_filters])
    # x_i = tf.reshape(x_i, [-1, review_num_i, num_filters])

    return x_u, x_i


def attention_weights(input_uid, input_iid, x_u, x_i,
                      user_num, item_num, embed_id_dim, random_seed,
                      attention_size, l2_reg_lambda):

    vec_uid = Embedding(user_num + 2, embed_id_dim, embeddings_initializer=RandomUniform(minval=-
                                                                                         0.1, maxval=0.1, seed=random_seed), name='user_id_embed')(input_uid)
    vec_iid = Embedding(item_num + 2, embed_id_dim, embeddings_initializer=RandomUniform(minval=-
                                                                                         0.1, maxval=0.1, seed=random_seed), name='item_id_embed')(input_iid)

    # Mapping user/item ID vectors and semantics of user/item's reviews to the attention space
    vec_uid = Dense(attention_size, activation=None, use_bias=False, kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(l2_reg_lambda), name='user_id_attention')(vec_uid)
    vec_iid = Dense(attention_size, activation=None, use_bias=False, kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(l2_reg_lambda), name='item_id_attention')(vec_iid)
    vec_textu = Dense(attention_size, activation=None, use_bias=False, kernel_initializer='glorot_uniform',
                      kernel_regularizer=l2(l2_reg_lambda), name='user_text_attention')(x_u)
    vec_texti = Dense(attention_size, activation=None, use_bias=False, kernel_initializer='glorot_uniform',
                      kernel_regularizer=l2(l2_reg_lambda), name='item_text_attention')(x_i)

    # Interaction between the user and each item review to learn personalized review-usefulness
    out_u = Multiply(name='usertext_itemid_interaction')([vec_textu, vec_iid])
    out_i = Multiply(name='itemtext_userid_interaction')([vec_texti, vec_uid])

    # b_u = np.random.uniform(low=-0.1, high=0.1, size=[attention_size])
    # b_i = np.random.uniform(low=-0.1, high=0.1, size=[attention_size])
    # b_u = b_u.astype(np.float32)
    # b_i = b_i.astype(np.float32)
    # # b_u = np.ndarray(b_u, 'float32')
    # # b_i = np.ndarray(b_i, 'float32')

    def biasadd_layer(x):
        b = tf.keras.backend.random_uniform_variable(
            [attention_size], low=-0.1, high=0.1, seed=random_seed)
        return tf.keras.backend.bias_add(x, b)
    # out_u = tf.keras.backend.bias_add(out_u, b_u)
    # out_i = tf.keras.backend.bias_add(out_i, b_i)
    out_u = Lambda(biasadd_layer)(out_u)
    out_i = Lambda(biasadd_layer)(out_i)
    # out_u = tf.keras.backend.bias_add(out_u, b_u)
    # out_i = tf.keras.backend.bias_add(out_i, b_i)

    # out_u = Dense(1, activation=None, use_bias=True,
    #               kernel_initializer='ones', bias_initializer='random_uniform')(out_u)
    # out_i = Dense(1, activation=None, use_bias=True,
    #               kernel_initializer='ones', bias_initializer='random_uniform')(out_i)

    out_u = ReLU()(out_u)
    out_i = ReLU()(out_i)
    out_u = Dense(1, activation=None, use_bias=True,
                  kernel_initializer='glorot_uniform', bias_initializer='zeros')(out_u)
    out_i = Dense(1, activation=None, use_bias=True,
                  kernel_initializer='glorot_uniform', bias_initializer='zeros')(out_i)

    # Output the weight (usefulness) for each review
    out_u = Softmax(axis=1, name='user_rev_weights')(out_u)
    out_i = Softmax(axis=1, name='item_rev_weights')(out_i)

    return out_u, out_i


def weighted_sum(out_u, x_u, out_i, x_i, dropout_keep_prob, random_seed):


    feas_u = Multiply()([out_u, x_u])
    feas_u = Lambda(tf.reduce_sum, arguments={'axis': 1})(feas_u)
    feas_i = Multiply()([out_i, x_i])
    feas_i = Lambda(tf.reduce_sum, arguments={'axis': 1})(feas_i)
    # feas_u = tf.reduce_sum(Multiply()([out_u, x_u]), axis=1)
    # feas_i = tf.reduce_sum(Multiply()([out_i, x_i]), axis=1)

    # Dropout layers here to reduce overfitting
    feas_u = Dropout(1 - dropout_keep_prob, seed=random_seed)(feas_u)
    feas_i = Dropout(1 - dropout_keep_prob, seed=random_seed)(feas_i)

    return feas_u, feas_i


def combine_features(input_uid, input_iid, feas_u, feas_i, user_num, item_num, n_latent, random_seed):


    vec_uid = Embedding(user_num + 2, n_latent, embeddings_initializer=RandomUniform(minval=-
                                                                                     0.1, maxval=0.1, seed=random_seed), name='user_id_latent')(input_uid)
    vec_iid = Embedding(item_num + 2, n_latent, embeddings_initializer=RandomUniform(minval=-
                                                                                     0.1, maxval=0.1, seed=random_seed), name='item_id_latent')(input_iid)
    vec_uid = Lambda(reshape_tensor, arguments={'shape': [-1, vec_uid.shape[-1]]})(vec_uid)
    vec_iid = Lambda(reshape_tensor, arguments={'shape': [-1, vec_iid.shape[-1]]})(vec_iid)

    # vec_uid = tf.reshape(vec_uid, [-1, vec_uid.shape[-1]])
    # vec_iid = tf.reshape(vec_iid, [-1, vec_iid.shape[-1]])

    t_u = Dense(n_latent, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                bias_initializer='zeros', name='user_rev_latent')(feas_u)
    t_i = Dense(n_latent, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                bias_initializer='zeros', name='item_rev_latent')(feas_i)

    # Merge the latent vectors for reviews and IDs
    f_u = Add(name='user_latent')([vec_uid, t_u])
    f_i = Add(name='item_latent')([vec_iid, t_i])
    return f_u, f_i
    # return t_u, t_i


def predict_rating(f_u, f_i, input_uid, input_iid, dropout_keep_prob, random_seed, user_num, item_num):


    # Merge the latent representation from user modeling and the latent representation from item modeling
    rating = Multiply()([f_u, f_i])
    rating = ReLU()(rating)
    rating = Dropout(1 - dropout_keep_prob, seed=random_seed)(rating)
    rating = Dense(1, activation=None, use_bias=True,
                   kernel_initializer='glorot_uniform', bias_initializer='zeros')(rating)

    bias_u = Embedding(user_num + 2, 1, embeddings_initializer='zeros')(input_uid)
    bias_i = Embedding(item_num + 2, 1, embeddings_initializer='zeros')(input_iid)
    bias_u = Lambda(reshape_tensor, arguments={'shape': [-1, 1]})(bias_u)
    bias_i = Lambda(reshape_tensor, arguments={'shape': [-1, 1]})(bias_i)
    # bias_u = tf.reshape(bias_u, [-1, 1])
    # bias_i = tf.reshape(bias_i, [-1, 1])

    rating = Add(name='final_rating')([rating, bias_u, bias_i])

    return rating


def DeepRecSys(l2_reg_lambda, random_seed, dropout_keep_prob, embed_word_dim, embed_id_dim,
               filter_size, num_filters, attention_size, n_latent,
               user_num, item_num, user_vocab_size, item_vocab_size,
               review_num_u, review_len_u, review_num_i, review_len_i,
               initW_u, initW_i, is_output_weights=False):

    # input_u: user's review history; input_i: item's reviews
    input_u = Input(shape=(review_num_u, review_len_u), dtype='int32', name='texts_u')
    input_i = Input(shape=(review_num_i, review_len_i), dtype='int32', name='texts_i')
    # input_uid: User ID; input_iid: Item ID
    input_uid = Input(shape=(1,), dtype='int32', name='uid')
    input_iid = Input(shape=(1,), dtype='int32', name='iid')

    x_u, x_i = CNN_text_processer(input_u, input_i, user_vocab_size, item_vocab_size, embed_word_dim, random_seed,
                                  num_filters, filter_size, review_num_u, review_len_u, review_num_i, review_len_i,
                                  initW_u, initW_i)
    # out_u, out_i = attention_weights(input_uid, input_iid, x_u, x_i,
    #                                  user_num, item_num, embed_id_dim, random_seed,
    #                                  attention_size, l2_reg_lambda)
    # feas_u, feas_i = weighted_sum(out_u, x_u, out_i, x_i, dropout_keep_prob, random_seed)
    f_u, f_i = combine_features(input_uid, input_iid, x_u, x_i,
                                user_num, item_num, n_latent, random_seed)
    rating = predict_rating(f_u, f_i, input_uid, input_iid,
                            dropout_keep_prob, random_seed, user_num, item_num)

    if is_output_weights:  # also output the weights (usefulness) of the reviews
        return Model(inputs=[input_u, input_i, input_uid, input_iid], outputs=[rating, out_u, out_i])
    else:
        return Model(inputs=[input_u, input_i, input_uid, input_iid], outputs=rating)
