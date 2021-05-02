#This work is inspired from https://medium.com/mlearning-ai/image-super-resolution-using-edsr-and-wdsr-f4de0b00e039
########################################## EDSR MODEL #####################################
def EDSR(scale, num_filters=256, res_blocks=8, res_block_scaling=None):
    x_input = Input(shape=(256, 256, 3))
   
    # assign value of x to x_res block for further operations
    x = x_res_block = Conv2D(num_filters, 3, padding='same')(x_input)

    # Goes in number of res block
    for i in range(res_blocks):
        x_res_block = ResBlock(x_res_block, num_filters, res_block_scaling)
    # convolution
    x_res_block = Conv2D(num_filters, 3, padding='same',kernel_initializer='he_normal')(x_res_block)
    x_res_block=LeakyReLU(alpha=0.1)(x_res_block)

    # add res_block output and original normalizwd input
    x = Add()([x, x_res_block])

    # upsampling
    x = Upsampling(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)
    x=AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)

    x = Conv2D(3, 3, padding='same')(x)
    return Model(x_input, x, name="EDSR")
  
################################## ResBlock Architecture ################################
def ResBlock(x_input, num_filters):
    '''This function Implementes Proposed ResBlock Architecture as per EDSR paper'''
    # proposed ResBlock ==> Conv --> Relu --> Conv --> Scaling(mul) --> Add
    x = Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')(x_input)
    x=LeakyReLU(alpha=0.1)(x)
    x = Conv2D(num_filters, 3, padding='same',kernel_initializer='he_normal')(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=AveragePooling2D(pool_size=(2,2),strides=(1,1),padding='same')(x)

    return x
######################################### Upsampling #######################################
def Upsampling(x, scale, num_filters):
    '''This function upsampling as mentioned in EDSR paper'''
    def upsample(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(shuffle_pixels(scale=factor))(x)

    if scale == 2:
        x = upsample(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample(x, 2, name='conv2d_1_scale_2')
        x = upsample(x, 2, name='conv2d_2_scale_2')

    return x

model=EDSR(2, num_filters=128, res_blocks=8, res_block_scaling=None)