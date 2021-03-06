#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 18:50:15 2022
B榜代码相比A榜写的要优雅一些，特征构造、评分、筛选单独写了类和函数，没有A榜代码那么多的特征构建代码……
B榜采用了对抗验证来缓解训练集与测试集分布不同的影响（但是没有把特征构建筛选放在对抗验证后，导致筛选的特征实际上依旧是针对原训练集的，所以特征验证显得效果不明显，没能起到清除毒特征的作用，pity）
B榜过程中，自己想明白了为什么特征增多时ANN的性能会剧烈下降，并对ANN的模块代码做了调整。而且通过本次竞赛发现，ANN在小样本下的性能比SVM稳定，更关键的是训练速度快、方便特征筛选时的迭代。
@author: lieu
"""

import sys
sys.path.append('/Users/lieu/Documents/Code/MachineLearning')
import pandas as pd
import numpy as np
import mlMethod.preProcess as prep
from mlMethod import annClass as ann
from mlMethod import featureConstructor as fc
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
import random
from sklearn import svm

'''import data '''
trainset = pd.read_csv('/Users/lieu/Documents/Code/MachineLearning/Kaggle/cmbFintech/2022cmbData/train.csv',
                       header=0)
testset_a = pd.read_csv('/Users/lieu/Documents/Code/MachineLearning/Kaggle/cmbFintech/2022cmbData/test_A.csv',
                       header=0)
testset_b = pd.read_csv('/Users/lieu/Documents/Code/MachineLearning/Kaggle/cmbFintech/2022cmbData/test_B.csv',
                       header=0)
trainset['sign'] = 'train'
testset_a['sign'] = 'testa'
testset_b['sign'] = 'testb'
dataset = pd.concat([trainset,testset_a,testset_b], axis=0, ignore_index=True)
dataset['vLabel'] = dataset['sign'].map(lambda x:0 if x in ['train','testa'] else 1)

'''data preprocess'''
preset = dataset.copy()
#unify the symbols of NaN
preset.replace('?', np.nan, inplace=True)
#fill nans
def minus2nfillna(data,column):
    for cl in column:
        data[cl] = pd.to_numeric(data[cl])
        data[cl] = data[cl].map(lambda x:(x-2) if x!=np.nan else x)
        #np.nan cant be filled by lambda function
        data[cl].fillna(0,inplace=True)
    return data
def fillna(data,column,a):
    for cl in column:
        data[cl].fillna(a,inplace=True)
    return data
m2f0 = ['AGN_CNT_RCT_12_MON','ICO_CUR_MON_ACM_TRX_TM','NB_RCT_3_MON_LGN_TMS_AGV',
        'AGN_CUR_YEAR_AMT','AGN_CUR_YEAR_WAG_AMT','AGN_AGR_LATEST_AGN_AMT','ICO_CUR_MON_ACM_TRX_AMT',
        'COUNTER_CUR_YEAR_CNT_AMT','PUB_TO_PRV_TRX_AMT_CUR_YEAR','CUR_YEAR_PUB_TO_PRV_TRX_PTY_CNT',
        'MON_12_EXT_SAM_TRSF_IN_AMT','MON_12_EXT_SAM_TRSF_OUT_AMT','MON_12_EXT_SAM_NM_TRSF_OUT_CNT','MON_12_EXT_SAM_AMT',
        'CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT','CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT','MON_12_TRX_AMT_MAX_AMT_PCTT',
        'MON_12_ACM_ENTR_ACT_CNT','MON_12_ACM_LVE_ACT_CNT','MON_6_50_UP_ENTR_ACT_CNT','MON_6_50_UP_LVE_ACT_CNT','CUR_YEAR_COUNTER_ENCASH_CNT',
        'MON_12_ACT_OUT_50_UP_CNT_PTY_QTY','MON_12_ACT_IN_50_UP_CNT_PTY_QTY','LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL',
        'LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV','LAST_12_MON_MON_AVG_TRX_AMT_NAV','COR_KEY_PROD_HLD_NBR','CUR_YEAR_MID_BUS_INC',
        'EMP_NBR','REG_CPT','SHH_BCK','HLD_DMS_CCY_ACT_NBR','HLD_FGN_CCY_ACT_NBR']
fn = ['MON_12_CUST_CNT_PTY_ID']
fb = ['WTHR_OPN_ONL_ICO']#why not a
fg = ['LGP_HLD_CARD_LVL','NB_CTC_HLD_IDV_AIO_CARD_SITU']
f0 = ['CUR_YEAR_MON_AGV_TRX_CNT','MON_12_AGV_TRX_CNT','MON_12_AGV_ENTR_ACT_CNT','MON_12_AGV_LVE_ACT_CNT',
      'LAST_12_MON_COR_DPS_DAY_AVG_BAL','CUR_MON_COR_DPS_MON_DAY_AVG_BAL','CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL','CUR_YEAR_COR_DPS_YEAR_DAY_AVG_INCR',
      'AI_STAR_SCO','REG_DT','OPN_TM']
preset = minus2nfillna(preset,m2f0)
preset = fillna(preset,fn,'N')
preset = fillna(preset,fb,'A')#I am not sure
preset = fillna(preset,fg,'G')
preset = fillna(preset,f0,0)
#dtype = object, need to be converted into np.number
for cl in f0:
    preset[cl] = pd.to_numeric(preset[cl])
#number code to char code
preset['AI_STAR_SCO'].map({0:'f',1:'a',2:'b',3:'c',4:'d',5:'e'})
dis_column = ['MON_12_CUST_CNT_PTY_ID','WTHR_OPN_ONL_ICO','LGP_HLD_CARD_LVL','NB_CTC_HLD_IDV_AIO_CARD_SITU','AI_STAR_SCO']
for cl in dis_column:
    counts = preset.value_counts(subset=cl)
    preset[cl] = preset[cl].map(lambda x:counts[x]/counts.sum())
#numeric processing
numberFeature = m2f0+f0 
numberFeature.remove('AI_STAR_SCO')    
delist = ['AGN_CUR_YEAR_WAG_AMT','MON_12_EXT_SAM_TRSF_IN_AMT','MON_12_EXT_SAM_TRSF_OUT_AMT','CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT',
          'MON_12_ACM_ENTR_ACT_CNT','MON_6_50_UP_ENTR_ACT_CNT','LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV',
          'LAST_12_MON_MON_AVG_TRX_AMT_NAV','CUR_YEAR_MID_BUS_INC','EMP_NBR','MON_12_AGV_TRX_CNT','MON_12_AGV_ENTR_ACT_CNT','MON_12_ACM_LVE_ACT_CNT',
       'MON_12_AGV_LVE_ACT_CNT','LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL','CUR_MON_COR_DPS_MON_DAY_AVG_BAL'] 
log_pre_list = ['AGN_CNT_RCT_12_MON','ICO_CUR_MON_ACM_TRX_TM','NB_RCT_3_MON_LGN_TMS_AGV',
                'AGN_CUR_YEAR_AMT','AGN_AGR_LATEST_AGN_AMT','ICO_CUR_MON_ACM_TRX_AMT','COUNTER_CUR_YEAR_CNT_AMT',
                'PUB_TO_PRV_TRX_AMT_CUR_YEAR','MON_12_EXT_SAM_NM_TRSF_OUT_CNT','CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT',
                'MON_12_TRX_AMT_MAX_AMT_PCTT','CUR_YEAR_MON_AGV_TRX_CNT','CUR_YEAR_PUB_TO_PRV_TRX_PTY_CNT', 
                'MON_6_50_UP_LVE_ACT_CNT','CUR_YEAR_COUNTER_ENCASH_CNT','MON_12_ACT_OUT_50_UP_CNT_PTY_QTY','MON_12_ACT_IN_50_UP_CNT_PTY_QTY',
                'LAST_12_MON_COR_DPS_DAY_AVG_BAL','CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL','COR_KEY_PROD_HLD_NBR',
                'REG_CPT','HLD_DMS_CCY_ACT_NBR','REG_DT','HLD_FGN_CCY_ACT_NBR','OPN_TM']+delist
log_pre_list.remove('CUR_YEAR_MID_BUS_INC')
for cl in log_pre_list:
    preset[cl] = np.log(preset[cl]+1)
for cl in numberFeature:
    preset[cl] = prep.dataStandardizing(preset[cl])

'''feature construction'''
fcm1 = fc.featureConstructor(preset,strip_column=['LABEL','CUST_UID','sign','vLabel'],label_name='vLabel')
fcm1.oddPower([1/3,1/5],score=0.1)
multi_set1 = pd.concat([fcm1.dataset,preset[['LABEL','CUST_UID','sign','vLabel']]],axis=1)
fcm1.featureScore(multi_set1)
low_set = multi_set1[fcm1.low_feature+fcm1.medium_feature+fcm1.high_feature+fcm1.very_high_feature]
after_score_set1 = pd.concat([low_set,preset[['LABEL','CUST_UID','sign','vLabel']]],axis=1)
after_score_set1.info()
control_set1 = after_score_set1.copy()

'''split train and validation'''
finaldata = control_set1.copy()
finaldata0 = finaldata[finaldata['vLabel']==0]
finaldata1 = finaldata[finaldata['vLabel']==1]
validation0 = finaldata0.sample(frac = 0.2, random_state=2)
validation1 = finaldata1.sample(frac = 0.2, random_state=1)
validation_before = pd.concat([validation0,validation1],axis=0)
train_before = pd.concat([finaldata0,finaldata1],axis=0).drop(validation_before.index,axis=0,inplace=False)
origin_index = np.array(validation_before.index)
random.shuffle(origin_index)
validation = validation_before.loc[origin_index].reset_index(drop=True,inplace=False)
origin_index2 = np.array(train_before.index)
random.shuffle(origin_index2)
train = train_before.loc[origin_index2].reset_index(drop=True,inplace=False)
#trainset and validation set
y_train = train['vLabel']
x_train = train.drop(['LABEL','CUST_UID','sign','vLabel'],axis=1,inplace=False)#UID cant be one of features
y_valid = validation['vLabel']
x_valid = validation.drop(['LABEL','CUST_UID','sign','vLabel'],axis=1,inplace=False)#UID cant be one of features

'''construct classifier'''
#label onehot
label_enc = OneHotEncoder()
label_enc.fit(train[['vLabel']])
label_train = pd.DataFrame(label_enc.transform(train[['vLabel']]).toarray(),columns = ['y_0','y_1'])

#ann
layers = [5,4,2]
alpha = [0.005,np.exp(-20),np.exp(-30)]
epoch = 50
num_batch = 5000
estimator2 = ann.annClass(x_train,label_train,layers,random_state=11)
for a in alpha:   
    estimator2.train(a,epoch,num_batch)
probability = estimator2.predict(x_valid)[:,1]
print(roc_auc_score(y_valid,probability))
RocCurveDisplay.from_predictions(y_valid,probability)
plt.show()

#predict
finaldata_x = finaldata[finaldata['sign']=='train'].drop(['LABEL','CUST_UID','sign','vLabel'],axis=1,inplace=False)#UID cant be one of features
estimation = estimator2.predict(finaldata_x)[:,0]
order = np.argsort(estimation)#返回为训练集数据的概率从小到大排序，从而为测试集概率从大到小
np.save('/Users/lieu/Documents/Code/MachineLearning/Kaggle/cmbFintech/2022cmbData/adversialVorder',order)

'''re-train the original model'''
#加载对抗验证后顺序
order1 = np.load('/Users/lieu/Documents/Code/MachineLearning/Kaggle/cmbFintech/2022cmbData/adversialVorder.npy')
#feature construction
fcm = fc.featureConstructor(preset,strip_column=['LABEL','CUST_UID','sign','vLabel'],label_name='LABEL')
fcm.oddPower([1/3,1/5],score=0.3)
multi_set = fcm.multiply(layers=[2,3,4],score=0.3)
criterion = fcm.distributionTest(sign_name='sign',sign_value_list=['train','testa','testb'])
multi_set2 = multi_set.drop(criterion['feature'][((criterion['C3testb']>1.1)|(criterion['C3testb']<0.9))|(criterion['C1testb']==False)],axis=1,inplace=False)
fcm.featureScore(multi_set2)
after_score_set = pd.concat([multi_set2[fcm.very_high_feature],multi_set2[['LABEL','CUST_UID','sign']]],axis=1)
control_set = after_score_set.copy()
#split train,validation and test set
final_dataset = control_set.copy()
traindata_before = final_dataset[final_dataset['sign']=='train'].loc[order[:5000]]
origin_index = np.array(traindata_before.index)
random.shuffle(origin_index)
train_dataset = traindata_before.loc[origin_index].reset_index(drop=True,inplace=False)
#testadata = finaldata[finaldata['sign']=='testa']
testbdata = final_dataset[final_dataset['sign']=='testb']
validation_set = train_dataset.sample(frac = 0.2, random_state=9)
train_set = train_dataset.drop(validation_set.index,axis=0,inplace=False).reset_index(drop=True,inplace=False)
validation_set.reset_index(drop=True,inplace=True)
#trainset and validation set
y_train_set = train_set['LABEL']
x_train_set = train_set.drop(['LABEL','CUST_UID','sign'],axis=1,inplace=False)#UID cant be one of features
y_valid_set = validation_set['LABEL']
x_valid_set = validation_set.drop(['LABEL','CUST_UID','sign'],axis=1,inplace=False)#UID cant be one of features
#test set
#x_test_a = testadata.drop(['LABEL','CUST_UID','sign'],axis=1,inplace=False)
x_test_b = testbdata.drop(['LABEL','CUST_UID','sign'],axis=1,inplace=False)

#ann
#label onehot
label_enc = OneHotEncoder()
label_enc.fit(train_dataset[['LABEL']])
label_trainset = pd.DataFrame(label_enc.transform(train_set[['LABEL']]).toarray(),columns = ['y_0','y_1'])

layers1 = [10,10,2]
alpha1 = [0.005,np.exp(-20),np.exp(-30)]
epoch1 = 50
num_batch1 = 500
random_state=11
estimator = ann.annClass(x_train_set,label_trainset,layers1,random_state)
for a in alpha1:   
    estimator.train(a,epoch1,num_batch1)
probability = estimator.predict(x_valid_set)[:,1]
print(roc_auc_score(y_valid_set,probability))
RocCurveDisplay.from_predictions(y_valid_set,probability)
plt.show()

traindata_x = train_dataset.drop(['LABEL','CUST_UID','sign'],axis=1,inplace=False)

traindata_y_label = pd.DataFrame(label_enc.transform(train_dataset[['LABEL']]).toarray(),columns = ['y_0','y_1'])
estimator = ann.annClass(traindata_x,traindata_y_label,layers1,random_state)
for a in alpha1:   
    estimator.train(a,epoch1,num_batch1)

#svm
estimator1 = svm.SVC(C=100,kernel='rbf',gamma=2**(-15),probability=True)
estimator1.fit(x_train_set,y_train_set)
y_threshold = estimator1.predict_proba(x_valid_set)[:,1]
print(roc_auc_score(y_valid_set,y_threshold))
RocCurveDisplay.from_predictions(y_valid_set,y_threshold)
plt.show()

traindata_y = train_dataset['LABEL']
estimator1.fit(traindata_x,traindata_y)

#predict and export
probability_ann = estimator.predict(x_test_b)[:,1]
prediction_ann = pd.DataFrame({'CUST_UID':testbdata['CUST_UID'],'Probablity':probability_ann})
prediction_ann.to_csv('/Users/lieu/Documents/Code/MachineLearning/Kaggle/cmbFintech/2022cmbData/prediction_b_ann.txt',
             index=False,float_format='%.10f',encoding='utf-8',sep=' ')

probability_svm = estimator1.predict_proba(x_test_b)[:,1]
prediction_svm = pd.DataFrame({'CUST_UID':testbdata['CUST_UID'],'Probablity':probability_svm})
prediction_svm.to_csv('/Users/lieu/Documents/Code/MachineLearning/Kaggle/cmbFintech/2022cmbData/prediction_b_svm.txt',
             index=False,float_format='%.10f',encoding='utf-8',sep=' ')
