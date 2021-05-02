#https://arxiv.org/pdf/1907.03128.pdf
#https://github.com/AureliePeng/Keras-WaveletTransform/blob/master/models/DWT.py
#https://www.tutorialspoint.com/keras/keras_customized_layer.htm
class dwt(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
      config = super().get_config().copy()
      
      return config
        
    def call(self, x):
       
        x1 = x[:, 0::2, 0::2, :] #x(2i−1, 2j−1)
        x2 = x[:, 1::2, 0::2, :] #x(2i, 2j-1)
        x3 = x[:, 0::2, 1::2, :] #x(2i−1, 2j)
        x4 = x[:, 1::2, 1::2, :] #x(2i, 2j)
        print(x1)   

        x_LL = x1 + x2 + x3 + x4
        x_LH = -x1 - x3 + x2 + x4
        x_HL = -x1 + x3 - x2 + x4
        x_HH = x1 - x3 - x2 + x4

        return Concatenate(axis=-1)([x_LL, x_LH, x_HL, x_HH])
    
#https://github.com/AureliePeng/Keras-WaveletTransform/blob/master/models/DWT.py
#https://www.tutorialspoint.com/keras/keras_customized_layer.htm

class iwt(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_config(self):
      config = super().get_config().copy()
      
      return config
        
    def call(self, x):
        x_LL = x[:, :, :, 0:x.shape[3]//4]
        x_LH = x[:, :, :, x.shape[3]//4:x.shape[3]//4*2]
        x_HL = x[:, :, :, x.shape[3]//4*2:x.shape[3]//4*3]
        x_HH = x[:, :, :, x.shape[3]//4*3:]

        x1 = (x_LL - x_LH - x_HL + x_HH)/4
        x2 = (x_LL - x_LH + x_HL - x_HH)/4
        x3 = (x_LL + x_LH - x_HL - x_HH)/4
        x4 = (x_LL + x_LH + x_HL + x_HH)/4 

        y1 = K.stack([x1,x3], axis=2)
        y2 = K.stack([x2,x4], axis=2)
        shape = K.shape(x)
        return K.reshape(K.concatenate([y1,y2], axis=-1), K.stack([shape[0],\
               shape[1]*2, shape[2]*2, shape[3]//4]))



def channel_attention(input_feature,channel,ratio):
  x=GlobalAveragePooling2D()(input_feature)
  x=Reshape((1,1,channel))(x)
  assert x.shape[1:] == (1,1,channel)
  x=Conv2D(channel // ratio,1,activation='relu',kernel_initializer='he_normal',\
           use_bias=True,bias_initializer='zeros')(x)
  assert x.shape[1:] == (1,1,channel//ratio)
  x = Conv2D(channel,1,activation='sigmoid',kernel_initializer='he_normal',\
             use_bias=True,bias_initializer='zeros')(x)
  x = multiply([input_feature, x])
  return x
#channel_attention(first_input,64,4)


def RCAB(prev_input,filters,kernal_size,blocks):
  for i in range(blocks):
    if (i==0):
      x=Conv2D(filters,kernal_size,padding='same')(prev_input)
    else:
      x=Conv2D(filters,kernal_size,padding='same')(lip)
    x= PReLU(alpha_initializer='he_normal')(x)
    x=Conv2D(filters,1,padding='same')(x)
    x=channel_attention(x,filters,4)
    if (i==0):
      lip=Add()([prev_input,x])
    else:
      lip=Add()([lip,x])
  x=Conv2D(filters,kernal_size,padding='same')(x)
  x=Add()([prev_input,x])
  return x
  #return Model(inputs=prev_input,outputs=x)

def Model_Creation():
  first_input=Input(shape=(256,256,3))

  #encoder3
  first=dwt()(first_input)
  inp=Conv2D(64,3,padding='same')(first)
  inp=PReLU(alpha_initializer='he_normal')(inp)
  second=RCAB(inp,64,3,3)


  #encoder2
  out_dwt_second = dwt()(second)
  inp=Conv2D(256,3,padding='same')(out_dwt_second)
  inp=PReLU(alpha_initializer='he_normal')(inp)
  third=RCAB(inp,256,3,3)

  #encoder1
  out_dwt_third=dwt()(third)
  inp=Conv2D(512,3,padding='same')(out_dwt_third)
  inp=PReLU(alpha_initializer='he_normal')(inp)
  inp=RCAB(inp,512,3,3)


  #decoder1
  inp=RCAB(inp,512,3,3)
  inp=Conv2D(1024,3,padding='same')(inp)
  inp=PReLU(alpha_initializer='he_normal')(inp)
  inp=iwt()(inp)
  inp=Add()([third,inp])

  #decoder2
  inp=RCAB(inp,256,3,3)
  inp=Conv2D(256,3,padding='same')(inp)
  inp=PReLU(alpha_initializer='he_normal')(inp)
  inp=iwt()(inp)
  inp=Add()([second,inp])


  #decoder3
  inp=RCAB(inp,64,3,3)
  inp=Conv2D(12,3,padding='same')(inp)
  inp=PReLU(alpha_initializer='he_normal')(inp)
  inp=iwt()(inp)

  out=Add()([first_input,inp])

  return Model(inputs=first_input,outputs=out)



model=Model_Creation()