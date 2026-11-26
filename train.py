import argparse,json
import numpy as np,pandas as pd,tensorflow as tf,optuna
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from config import *
p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=3); p.add_argument('--trials',type=int,default=3); p.add_argument('--cv',action='store_true'); p.add_argument('--advanced-augmentation',action='store_true'); a=p.parse_args()
keras.utils.set_random_seed(SEED)
df=pd.read_csv(DATA/'manifest.csv'); names=json.loads((DATA/'classes.json').read_text()); n=len(names)
tr,va,te=[df[df.split==s] for s in ['train','valid','test']]
def dataset(frame,training=False,advanced=False):
 d=tf.data.Dataset.from_tensor_slices((frame.path.tolist(),frame.label.astype('int32').tolist()))
 if training: d=d.shuffle(len(frame),seed=SEED)
 def load(path,y):
  x=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False); x=tf.image.resize(x,SIZE)
  return x,tf.one_hot(y,n)
 d=d.map(load,num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH)
 if training and advanced:
  def mix(x,y):
   lam=tf.random.uniform([],0.3,0.7); other=tf.reverse(x,[0]); yy=tf.reverse(y,[0])
   def cut():
    side=tf.cast(224*tf.sqrt(1-lam),tf.int32); top=tf.random.uniform([],0,225-side,dtype=tf.int32); left=tf.random.uniform([],0,225-side,dtype=tf.int32)
    mask=tf.pad(tf.ones([side,side,1]),[[top,224-top-side],[left,224-left-side],[0,0]])
    w=tf.reduce_mean(mask); return x*(1-mask)+other*mask,y*(1-w)+yy*w
   return tf.cond(tf.random.uniform([])<0.5,lambda:(lam*x+(1-lam)*other,lam*y+(1-lam)*yy),cut)
  d=d.map(mix,num_parallel_calls=tf.data.AUTOTUNE)
 return d.prefetch(tf.data.AUTOTUNE)
def build(kind,lr=.001,drop=.2,width=128,optimizer='Adam',unfreeze=0):
 inp=keras.Input((*SIZE,3)); x=inp
 if kind=='baseline':
  x=layers.Rescaling(1/255)(x)
  for filters in [16,32,64]: x=layers.Conv2D(filters,3,activation='relu',padding='same')(x); x=layers.MaxPooling2D()(x)
  x=layers.Activation('linear',name='cam_features')(x)
 else:
  x=keras.Sequential([layers.RandomFlip('horizontal'),layers.RandomRotation(.1),layers.RandomContrast(.15)],name='augmentation')(x)
  base=keras.applications.EfficientNetB0(include_top=False,weights='imagenet',input_shape=(*SIZE,3)); base.trainable=bool(unfreeze)
  if unfreeze:
   for l in base.layers: l.trainable=False
   for l in base.layers[-unfreeze:]: l.trainable=not isinstance(l,layers.BatchNormalization)
  x=base(x,training=False); x=layers.Activation('linear',name='cam_features')(x)
 x=layers.GlobalAveragePooling2D()(x); x=layers.Dense(width,activation='relu')(x)
 if kind!='baseline': x=layers.Dropout(drop)(x)
 out=layers.Dense(n,activation='softmax')(x); m=keras.Model(inp,out)
 opt=(keras.optimizers.AdamW if optimizer=='AdamW' else keras.optimizers.Adam)(learning_rate=lr)
 m.compile(opt,loss='categorical_crossentropy',metrics=['accuracy']); return m
def fit(m,train=tr,valid=va,advanced=False,epochs=None):
 weights=compute_class_weight('balanced',classes=np.arange(n),y=train.label.to_numpy())
 d=dataset(train,True,advanced)
 # Soft-label weighted sample weights, compatible with MixUp/CutMix.
 if advanced: d=d.map(lambda x,y:(x,y,tf.reduce_sum(y*tf.constant(weights,dtype=tf.float32),axis=1)))
 return m.fit(d,validation_data=dataset(valid),epochs=epochs or a.epochs,class_weight=None if advanced else dict(enumerate(weights)),callbacks=[keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)],verbose=2)
def save(m,h,name):
 m.save(MODELS/f'{name}.keras'); pd.DataFrame(h.history).to_csv(RESULTS/f'{name}_history.csv',index=False)
for kind in ['baseline','efficientnet']:
 m=build(kind); h=fit(m); save(m,h,kind); del m; keras.backend.clear_session()
def objective(t):
 keras.backend.clear_session(); params=dict(lr=t.suggest_float('lr',1e-5,1e-3,log=True),drop=t.suggest_float('drop',.1,.5),width=t.suggest_categorical('width',[64,128,256]),optimizer=t.suggest_categorical('optimizer',['Adam','AdamW']),unfreeze=t.suggest_categorical('unfreeze',[0,10,20])); m=build('efficientnet',**params); h=fit(m,advanced=a.advanced_augmentation,epochs=min(a.epochs,2)); return max(h.history['val_accuracy'])
study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED)); study.optimize(objective,n_trials=a.trials); study.trials_dataframe().to_csv(RESULTS/'optuna.csv',index=False); (RESULTS/'best_params.json').write_text(json.dumps(study.best_params)); keras.backend.clear_session()
m=build('efficientnet',**study.best_params); h=fit(m,advanced=a.advanced_augmentation); save(m,h,'optimized'); del m
rows=[]
for name in ['baseline','efficientnet','optimized']:
 m=keras.models.load_model(MODELS/f'{name}.keras'); pred=m.predict(dataset(te),verbose=0).argmax(1); report=classification_report(te.label,pred,labels=list(range(n)),target_names=names,output_dict=True,zero_division=0); pd.DataFrame(report).T.to_csv(RESULTS/f'{name}_report.csv'); rows.append({'model':name,'accuracy':accuracy_score(te.label,pred),**{f'macro_{k}':report['macro avg'][k] for k in ['precision','recall','f1-score']},'weighted_f1':report['weighted avg']['f1-score']}); cm=confusion_matrix(te.label,pred,labels=list(range(n))); fig,ax=plt.subplots(figsize=(10,8)); ax.imshow(cm); ax.set_xticks(range(n),names,rotation=90); ax.set_yticks(range(n),names); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); fig.tight_layout(); fig.savefig(RESULTS/f'{name}_confusion.png'); plt.close(fig); del m
pd.DataFrame(rows).to_csv(RESULTS/'comparison.csv',index=False); print(pd.DataFrame(rows))
if a.cv:
 pool=pd.concat([tr,va],ignore_index=True); scores=[]
 for i,(ix,jx) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(pool,pool.label)):
  keras.backend.clear_session(); m=build('efficientnet',**study.best_params); fit(m,pool.iloc[ix],pool.iloc[jx],a.advanced_augmentation); scores.append({'fold':i+1,'accuracy':m.evaluate(dataset(pool.iloc[jx]),verbose=0)[1]}); pd.DataFrame(scores).to_csv(RESULTS/'cv.csv',index=False)
