import tensorflow as tf
from deepctr.input_embedding import preprocess_input_embedding
from deepctr.layers.core import MLP, PredictionLayer
from deepctr.layers.interaction import CIN
from deepctr.layers.utils import concat_fun
from deepctr.utils import check_feature_config_dict
from tensorflow.keras.utils import plot_model

def xDeepFM_MTL(feature_dim_dict, embedding_size=1, hidden_size=(256, 256), cin_layer_size=(256, 256,),
                cin_split_half=True,
                task_net_size=(128,), l2_reg_linear=0.00001, l2_reg_embedding=0.00001,
                seed=1024, ):
    check_feature_config_dict(feature_dim_dict)
    if len(task_net_size) < 1:
        raise ValueError('task_net_size must be at least one layer')

    deep_emb_list, linear_logit, inputs_list = preprocess_input_embedding(
        feature_dim_dict, embedding_size, l2_reg_embedding, l2_reg_linear, 0.0001, seed)

    # Add fuzzy feature inputs
    #watch_membership_input = tf.keras.layers.Input(shape=(1,), name='watch_membership_input')
    #interest_membership_input = tf.keras.layers.Input(shape=(1,), name='interest_membership_input')
    #inputs_list.extend([watch_membership_input, interest_membership_input])

    fm_input = concat_fun(deep_emb_list, axis=1)

    if len(cin_layer_size) > 0:
        exFM_out = CIN(cin_layer_size, 'relu',
                       cin_split_half, seed)(fm_input)
        exFM_logit = tf.keras.layers.Dense(1, activation=None, )(exFM_out)

    deep_input = tf.keras.layers.Flatten()(fm_input)
    deep_out = MLP(hidden_size)(deep_input)

    #combined_input = concat_fun([deep_out, watch_membership_input, interest_membership_input], axis=1)

    finish_out = MLP(task_net_size)(deep_out)
    finish_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(finish_out)

    like_out = MLP(task_net_size)(deep_out)
    like_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(like_out)

    finish_logit = tf.keras.layers.add(
        [linear_logit, finish_logit, exFM_logit])
    like_logit = tf.keras.layers.add(
        [linear_logit, like_logit, exFM_logit])

    output_finish = PredictionLayer('sigmoid', name='finish')(finish_logit)
    output_like = PredictionLayer('sigmoid', name='like')(like_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=[
                                  output_finish, output_like])
    #plot_model(model, to_file='model.png')
    return model



