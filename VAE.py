import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop, Adam, sgd
from keras import objectives

def create_lstm_vae(input_dim, 
    timesteps, 
    batch_size, 
    intermediate_dim, 
    latent_dim,
    layers =1,
    epsilon_std=1.):

    """
    File for making VAE with variable number of layers.
    Creates an RNN Variational Autoencoder (VAE). Returns VAE, Encoder, Generator. 
    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        batch_size: int.
        intermediate_dim: int, output shape of LSTM. 
        latent_dim: int, latent z-layer shape. 
        epsilon_std: float, z-layer sigma.
    """
    x = Input(shape=(timesteps, input_dim,))

    h = x
    # LSTM encoding
    for i in range(layers-1):
      h = LSTM(intermediate_dim, return_sequences=True)(h)

    h = LSTM(intermediate_dim)(h)

    # VAE Z layer
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)
    
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(latent_dim,),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    
    # decoded RNN layer
    decoder_h, decoder_mean = [], []
    for i in range(layers):
      decoder_h.append(LSTM(intermediate_dim, return_sequences=True))
      decoder_mean.append(LSTM(input_dim, return_sequences=True, activation=None))

    h_decoded = RepeatVector(timesteps)(z)
    for i in range(layers):
      h_decoded = decoder_h[i](h_decoded)

    x_decoded_mean = h_decoded
    # decoded layer
    for i in range(layers):
      x_decoded_mean = decoder_mean[i](x_decoded_mean)
    
    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))

    _h_decoded = RepeatVector(timesteps)(decoder_input)
    
    for i in range(layers):
      _h_decoded = decoder_h[i](_h_decoded)

    _x_decoded_mean = _h_decoded
    for i in range(layers):
      _x_decoded_mean = decoder_mean[i](_x_decoded_mean)
   
    generator = Model(decoder_input, _x_decoded_mean)
    
    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    vae.compile(optimizer='rmsprop', loss=vae_loss)
    
    return vae, encoder, generator