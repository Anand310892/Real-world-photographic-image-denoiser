#Denseblock
def denseBlock(previous_output,ks,depth):
  op_x1=Conv2D(depth,(ks,ks),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.03),\
               bias_regularizer=l2(0.03))(previous_output)
  op_x2=Activation('relu')(op_x1)
  conc1=concatenate([previous_output,op_x2],axis=-1)  

  op_x3=Conv2D(depth,(ks,ks),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.03), \
               bias_regularizer=l2(0.03))(conc1)
  op_x4=Activation('relu')(op_x3)
  conc2=concatenate([previous_output,conc1,op_x4],axis=-1)

  op_x5=Conv2D(depth,(ks,ks),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.03), \
               bias_regularizer=l2(0.03))(conc2)
  op_x6=Activation('relu')(op_x5)
  conc3=concatenate([previous_output,conc1,conc2,op_x6],axis=-1)

  op_x7=Conv2D(depth,(ks,ks),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.03), \
               bias_regularizer=l2(0.03))(conc3)
  op_x8=Activation('relu')(op_x7)
  out_aspp=ASPP(previous_output,depth)
  conc3=concatenate([previous_output,conc1,conc2,conc3,op_x8,out_aspp],axis=-1)

  mdr_out=Conv2D(128, (1,1), padding='same',kernel_regularizer=l2(0.03), bias_regularizer=l2(0.03))(conc3)

  final_mdr_out=Add()([mdr_out,previous_output])

  return final_mdr_out

#ASPP block
def ASPP(previous_output,depth):
  op_x1=Conv2D(depth,(3,3),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.03), \
               bias_regularizer=l2(0.03))(previous_output)
  op_x2=Activation('relu')(op_x1)

  
  op_x3 = Conv2D(depth, (1,1), padding='same',kernel_regularizer=l2(0.03), bias_regularizer=l2(0.03))(op_x2)
  op_x3    =  Dropout(0.3)(op_x3)
  op_x4 = Conv2D(depth, (3,3), padding='same',dilation_rate=6,kernel_regularizer=l2(0.03), \
                 bias_regularizer=l2(0.03))(op_x2)
  op_x4    =  Dropout(0.3)(op_x4)
  op_x5 = Conv2D(depth, (3,3), padding='same',dilation_rate=12,kernel_regularizer=l2(0.03), \
                 bias_regularizer=l2(0.03))(op_x2)
  op_x5    =  Dropout(0.3)(op_x5)
  op_x6 = MaxPooling2D((3,3), strides=(1,1), padding='same')(op_x2)
  
  conc4    = concatenate([op_x3,op_x4,op_x5,op_x6],axis=-1)
  op_x7    = Conv2D(depth, (1,1), padding='same',kernel_regularizer=l2(0.03), bias_regularizer=l2(0.03))(conc4)
  return op_x7

#Sequential model starts from here.
depth=128
first_input=Input(shape=(256,256,3))
inp1 = Conv2D(depth, (3,3), padding='same',kernel_regularizer=l2(0.03), bias_regularizer=l2(0.03))(first_input)
inp2 = Conv2D(depth, (3,3), padding='same',kernel_regularizer=l2(0.03), bias_regularizer=l2(0.03))(inp1)

inp3 = denseBlock(inp2,3,128)
inp3 =  Dropout(0.3)(inp3)
inp4 = denseBlock(inp3,3,128)
inp4 =  Dropout(0.3)(inp4)


conc = concatenate([inp2,inp3,inp4],axis=-1)

conv3    = Conv2D(depth, (1,1), padding='same',kernel_regularizer=l2(0.03), bias_regularizer=l2(0.03))(conc)
conv4    = Conv2D(depth, (3,3), padding='same',kernel_regularizer=l2(0.03), bias_regularizer=l2(0.03))(conv3)
add      = Add()([inp1,conv4])
conv5    = Conv2D(depth, (3,3), padding='same',kernel_regularizer=l2(0.03), bias_regularizer=l2(0.03))(add)
outfinal    = Conv2D(3, (3,3), padding='same',kernel_regularizer=l2(0.03), bias_regularizer=l2(0.03))(conv5)
#create model
model=Model(inputs=first_input,outputs = outfinal)