#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:58:16 2022
国泰君安·发债企业违约风险预警baseline
@author: lieu
"""
import sys
sys.path.append('/Users/lieu/Documents/Code/MachineLearning')
import numpy as np
import pandas as pd
from mlMethod import preProcess as prep
from mlMethod import annClass as ann
from mlMethod import featureConstructor as fc
from datetime import datetime
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from matplotlib import pyplot as plt
from sklearn import svm
'''
导入原始数据
import raw data
'''
infoset = pd.read_csv('/Users/lieu/Documents/Code/MachineLearning/0dataCompetition/220513国泰君安发债企业违约风险预测/2022S-T3-1st-Training/ent_info.csv',
                      parse_dates=['opfrom','opto','esdate','apprdate'],sep='|')
finset = pd.read_csv('/Users/lieu/Documents/Code/MachineLearning/0dataCompetition/220513国泰君安发债企业违约风险预测/2022S-T3-1st-Training/ent_financial_indicator.csv',
                     parse_dates=['report_period'],sep='|')
defaultset = pd.read_csv('/Users/lieu/Documents/Code/MachineLearning/0dataCompetition/220513国泰君安发债企业违约风险预测/2022S-T3-1st-Training/ent_default.csv',
                     parse_dates=['acu_date'],sep='|')
newset = pd.read_csv('/Users/lieu/Documents/Code/MachineLearning/0dataCompetition/220513国泰君安发债企业违约风险预测/2022S-T3-1st-Training/ent_news.csv',
                       parse_dates=['publishdate'],sep='|',usecols=list(range(9)))

'''
数据预处理
pre-process raw data
'''
###############################
#依据各表主键，删除重复值
#delete duplicates according to the primary key
###############################
infoset.drop_duplicates(['ent_id'],keep='first',inplace=True,ignore_index=True)
finset.drop_duplicates(keep='first',inplace=True,ignore_index=True)
finset.drop_duplicates(['ent_id','report_period'],keep='last',inplace=True,ignore_index=True)#随便删的，样本少，值差距也小
newset.drop_duplicates(['newscode','ent_id','index','indextype','impscore'],keep='first',inplace=True,ignore_index=True)
#删除重复数据中impscore较大的记录,可能写的比较蠢，执行比较慢
de = newset['newscode'][newset.duplicated(['newscode','ent_id','index','indextype'])]
dellist = np.array([])
for a in list(de.unique()):
    b = newset['impscore'][newset['newscode']==a]
    indexlist = np.array(b.index)
    sortlist = np.argsort(np.array(b))
    dellist = np.hstack((dellist,indexlist[sortlist[1:]]))#略微改了以下，把所有索引并到一起，最后再统一drop
newset.drop(dellist,axis=0,inplace=True)
newset.reset_index(drop=True,inplace=True)

'''
特征工程
feature engineering
'''
###############################
#从理论基础出发，抛弃部分字段
#delete some fields based on theory
###############################
infoset.drop(['opto','esdate','apprdate'],axis=1,inplace=True)
newset.drop(['publishtime','newssource','newstitle'],axis=1,inplace=True)#publishtime中含2021，实际为2020年数据
#统一纳入新表preset
preset = infoset[['ent_id','regcap']]
###############################
#分类字段数值化
#convert category fields into numeric
###############################
#基本信息表industryphy按照是否为重资产高周转行业(经营风险)进行二值化，重资产为1，非为0，空值占比极低暂时并入0
#industryco整理起来太麻烦了，暂不纳入特征空间
ind_1_list = ['租赁和商务服务业','交通运输、仓储和邮政业','批发和零售业','房地产业','制造业','建筑业','电力、热力、燃气及水生产和供应业',
              '水利、环境和公共设施管理业','采矿业','住宿和餐饮业','教育']
ind_0_list = ['信息传输、软件和信息技术服务业','金融业','科学研究和技术服务业','农、林、牧、渔业','文化、体育和娱乐业',
              '居民服务、修理和其他服务业','公共管理、社会保障和社会组织','国际组织','卫生和社会工作',np.nan]
preset['industryphy'] = infoset['industryphy'].map(lambda x:1 if x in ind_1_list else 0)
#基本信息表经营时间起opfrom转换为年数(时间久抗风险能力相对高一些，基业长青)；空值填充为同行业平均值
preset['optime'] = infoset['opfrom'].map(lambda x:2022-x.year)#x.values转成时间戳会有负数,这里更严谨的应该使用经营日期与设立日期中早的一个
preset['indus'] = infoset['industryphy']
for val in preset['indus'][preset['indus'].notna()].unique():
    nan_index = preset['optime'][(preset['indus']==val)&(preset['optime'].isnull())].index
    mean = preset['optime'][(preset['indus']==val)&(preset['optime'].notna())].mean()
    preset.loc[nan_index,['optime']]=mean#这里要用loc，用preset[cl][condition]=mean无法赋值
cl_mean = preset['optime'][preset['optime'].notna()].mean()
indus_nan_index = (preset['optime'][preset['optime'].isnull()]).index
preset.loc[indus_nan_index,['optime']]=cl_mean   
#基本信息表企业类型enttype按照是否为国企(外部支持)、是否上市(经营风险)分成两个二值字段
state_1_list = ['有限责任公司（国有独资）','有限责任公司（国有控股）','股份有限公司（非上市、国有控股）','股份有限公司（上市、国有控股）',
                '全民所有制','股份有限公司分公司（国有控股）']
public_1_list = ['其他股份有限公司（上市）','股份有限公司（台港澳合资、上市）','股份有限公司（上市、自然人投资或控股）','股份有限公司（上市、国有控股）',
                 '股份有限公司（台港澳与境内合资、上市）','股份有限公司（中外合资、上市）','股份有限公司（上市）','股份有限公司（上市、外商投资企业投资）','股份有限公司分公司（上市）']
preset['stateowned'] = infoset['enttype'].map(lambda x:1 if x in state_1_list else 0)
preset['public'] = infoset['enttype'].map(lambda x:1 if x in public_1_list else 0)
#基本信息表地区prov改为频率（商业氛围、政策环境等有一定概率体现在企业数量上）
count_region = infoset.value_counts(subset='prov')
preset['region'] = infoset['prov'].map(lambda x:count_region[x]/count_region.sum())
bf_k = preset.copy()#备份用于后续构建测试集
###############################
#财务指标表预处理&挂接
###############################
#财务指标中，盈利能力：息税前利润率、销售毛利率、净利率、净资产收益率
profit_list = ['s_fa_ebittogr','s_fa_grossprofitmargin','s_fa_profittogr','s_fa_roe_yearly']
#财务指标中，偿债能力：财务费用率、流动比率、速动比率、利息保障倍数
solvency_lsit = ['s_fa_finaexpensetogr','s_fa_current','s_fa_quick','s_fa_ebittointerest']
#财务指标中，资本结构：资产负债率、有形资产/负债
capital_list = ['s_fa_debttoassets','s_fa_tangibleassettodebt']
#财务指标中，财务弹性：经营现金流占营业收入比例、经营现金流/营运资金、流动负债占比
elasticity_list = ['s_fa_ocftoor','s_fa_longdebttoworkingcapital','s_fa_currentdebttodebt']
#经营指标：存货周转率，应收帐款周转率、总资产周转率
operation_list = ['s_fa_invturn','s_fa_arturn','s_fa_assetsturn']
#indicator为财务指标表处理过渡dataframe
indicator = finset[profit_list+solvency_lsit+capital_list+operation_list+['ent_id']]
indicator['report_year'] = finset['report_period'].dt.year
#各项财务指标取4季度均值，并挂接(2020年数据后续挂接至测试集)
indicator_grouped = indicator.groupby(['ent_id','report_year'],as_index=False).mean()
financial_2018 = indicator_grouped[indicator_grouped['report_year']==2018]
financial_2019 = indicator_grouped[indicator_grouped['report_year']==2019]
financial_2020 = indicator_grouped[indicator_grouped['report_year']==2020]
preset = preset.merge(financial_2018,how='left',on='ent_id',suffixes=('_y1','_y2'),validate='one_to_one')
preset = preset.merge(financial_2019,how='left',on='ent_id',suffixes=('_y1','_y2'),validate='one_to_one')
preset.drop(['report_year_y1','report_year_y2'],axis=1,inplace=True)
#对无财务数据的空值企业，填充以同行业的平均值;如果行业也是空值，则填充整列平均值
bf = preset.copy()#备份
preset = bf.copy()
for cl in preset.columns:
    if cl not in ['indus','ent_id']:
        for val in preset['indus'][preset['indus'].notna()].unique():
            nan_index = preset[cl][(preset['indus']==val)&(preset[cl].isnull())].index
            mean = preset[cl][(preset['indus']==val)&(preset[cl].notna())].mean()
            preset.loc[nan_index,[cl]]=mean#这里要用loc，用preset[cl][condition]=mean无法赋值
        cl_mean = preset[cl][preset[cl].notna()].mean()
        indus_nan_index = (preset[cl][preset[cl].isnull()]).index
        preset.loc[indus_nan_index,[cl]]=cl_mean           
###############################
#舆情表预处理&挂接
###############################
#按重要性评分计算不同重要性新闻数量
news = newset[['ent_id','newscode','indextype','impscore','publishdate']]
news['publish_year'] = news['publishdate'].dt.year
news_grouped = news[['ent_id']].drop_duplicates()
news_grouped_impor = news[['ent_id','newscode','impscore','publish_year']].groupby(['ent_id','impscore','publish_year'],as_index=False).count()
for val_imp in news['impscore'].unique():
    for year in news['publish_year'].unique():
        if year!=2020:#2020后续加入测试集
            new_cl_name = 'imp_'+str(val_imp)+'_cnt'+str(year)
            news_imp_selected = news_grouped_impor[['ent_id','newscode']][(news_grouped_impor['impscore']==val_imp)&(news_grouped_impor['publish_year']==year)]
            news_imp_selected.rename({'newscode':new_cl_name},axis=1,inplace=True)
            news_grouped = news_grouped.merge(news_imp_selected,how='left',on='ent_id',validate='1:1')
news_grouped_index_temp = news[['ent_id','newscode','indextype','publish_year']].groupby(['ent_id','publish_year','indextype'],as_index=False).count()
for val_index in news['indextype'].unique():
    for year in news['publish_year'].unique():
        if year!=2020:
            new_cl_name = 'index_'+str(val_index)+'_cnt'+str(year)
            news_index_selected = news_grouped_index_temp[['ent_id','newscode']][(news_grouped_index_temp['indextype']==val_index)&(news_grouped_index_temp['publish_year']==year)]
            news_index_selected.rename({'newscode':new_cl_name},axis=1,inplace=True)
            news_grouped = news_grouped.merge(news_index_selected,how='left',on='ent_id',validate='1:1')
#拼接至preset，并填充空值为0   
preset = preset.merge(news_grouped,how='left',on='ent_id',validate='1:1')
for cl in news_grouped.columns:
    if cl!='ent_id':
        preset[cl].fillna(0,inplace=True)#fillna不支持二级切片，preset[cl][condition]不行
################################
#违约表预处理&挂接
################################
defaultset['acu_year'] = defaultset['acu_date'].dt.year
default = defaultset[['ent_id']].drop_duplicates()
default_grouped = defaultset.groupby(['ent_id','acu_year'],as_index=False).count()
for ac_year in  default_grouped['acu_year'].unique():
    if ac_year!=2020:#2020后续加入测试集
        new_cl_name = 'acu_cnt'+str(ac_year)
        default_grouped_selected = default_grouped[['ent_id','acu_date']][default_grouped['acu_year']==ac_year]
        default_grouped_selected.rename({'acu_date':new_cl_name},axis=1,inplace=True)
        default = default.merge(default_grouped_selected,how='left',on='ent_id',validate='1:1')
#拼接至preset，并填充空值为0
preset = preset.merge(default,how='left',on='ent_id',validate='1:1')
for cl in default.columns:
    if cl!='ent_id':
        preset[cl].fillna(0,inplace=True)
################################
#构建训练集label
################################
default_grouped_2020 =  default_grouped[['ent_id','acu_date']][default_grouped['acu_year']==2020]
default_grouped_2020.rename({'acu_date':'LABEL'},axis=1,inplace=True)
default_grouped_2020['LABEL'] = default_grouped_2020['LABEL'].map(lambda x:1 if x>0 else 0)
preset = preset.merge(default_grouped_2020,how='left',on='ent_id',validate='1:1')
preset['LABEL'].fillna(0,inplace=True)
###############################
#基本信息表都是0-1二值或频率数据，只有注册资本、经营时间需要对数化+标准化
#对比率数据进行行业内标准化处理
#对新闻数量进行正态处理和行业内标准化处理
#对违约次数不做处理（数值不大不做行业内标准化。也可以考虑二值化）
###############################
bf_d = preset.copy()#备份
preset=bf_d.copy()
#financial indicator
indicator_list = pd.Series(profit_list+solvency_lsit+capital_list+operation_list)
indicator_list = (indicator_list+'_y1').tolist()+(indicator_list+'_y2').tolist()
for cl in indicator_list:
    for val_ind in preset['indus'][preset['indus'].notna()].unique():
        indus_index = preset[cl][preset['indus']==val_ind].index
        val_mean = preset.loc[indus_index,[cl]].mean()
        if len(indus_index)>1:
            val_std = preset.loc[indus_index,[cl]].std()
            if val_std[0]==0: val_std = 1
        else:val_std=1#如果只有一个元素，std=nan
        preset.loc[indus_index,[cl]] = (preset.loc[indus_index,[cl]]-val_mean)/val_std
    indus_nan_index = preset[cl][preset['indus'].isnull()].index
    indus_nan_mean = preset.loc[indus_nan_index,[cl]].mean()
    if len(indus_nan_index)>1:
        indus_nan_std = preset.loc[indus_nan_index,[cl]].std()
        if indus_nan_std[0]==0: indus_nan_std = 1
    else:indus_nan_std=1
    preset.loc[indus_nan_index,[cl]] = (preset.loc[indus_nan_index,[cl]]-indus_nan_mean)/indus_nan_std
bf_n = preset.copy()#备份
preset=bf_n.copy()  
#news & regcap
for cl in news_grouped.columns.tolist()+['regcap','optime']:#必须转为list，不然是视为series+字符串处理方法
    if cl!='ent_id' :
        preset[cl] = np.log(preset[cl]+1)
for cl in news_grouped.columns.tolist()+['regcap','optime']:
    if cl!='ent_id' :
        for val_ind in preset['indus'][preset['indus'].notna()].unique():
            indus_index = preset[cl][preset['indus']==val_ind].index
            val_mean = preset.loc[indus_index,[cl]].mean()
            if len(indus_index)>1:
                val_std = preset.loc[indus_index,[cl]].std()
                if val_std[0]==0: val_std = 1#如果都为0，则std也为0，改为1
            else:val_std=1
            preset.loc[indus_index,[cl]] = (preset.loc[indus_index,[cl]]-val_mean)/val_std
        indus_nan_index = preset[cl][preset['indus'].isnull()].index
        indus_nan_mean = preset.loc[indus_nan_index,[cl]].mean()
        if len(indus_nan_index)>1:
            indus_nan_std = preset.loc[indus_nan_index,[cl]].std()#返回的是一个series
            if indus_nan_std[0]==0: indus_nan_std = 1
        else:indus_nan_std=1
        preset.loc[indus_nan_index,[cl]] = (preset.loc[indus_nan_index,[cl]]-indus_nan_mean)/indus_nan_std
afterset = preset.copy()#预处理完成
afterset.drop('indus',axis=1,inplace=True)
afterset.info()
'''
特征评价
feature selection
'''
fcm = fc.featureConstructor(afterset,strip_column=['LABEL','ent_id'],label_name='LABEL')
#fcm.oddPower([1/3,1/5],score=4.5)
multi_set = fcm.multiply(layers=[2,3,4],score=2)
#criterion = fcm.distributionTest(sign_name='sign',sign_value_list=['train','testa','testb'])
#multi_set2 = multi_set.drop(criterion['feature'][((criterion['C3testb']>1.1)|(criterion['C3testb']<0.9))|(criterion['C1testb']==False)],axis=1,inplace=False)
fcm.featureScore(multi_set)
len(fcm.very_high_feature)
len(fcm.high_feature)
for cl in fcm.very_high_feature:
    fs = fcm.featureScore_single(multi_set[cl])
    print(cl)
    print(fs)
control_set = pd.concat([multi_set[fcm.very_high_feature[200:]],afterset[['LABEL','ent_id']]],axis=1)
control_set.info()
'''
分割训练集和验证集
split the train-set and validation-set
'''
finaldata = control_set.copy()
finaldata0 = finaldata[finaldata['LABEL']==0]
finaldata1 = finaldata[finaldata['LABEL']==1]
#抽样
random_seeds = [224,7801,11127,334,2]
validation0 = finaldata0.sample(frac = 0.2, random_state=random_seeds[0])#更改随机种子实现多次留出法验证
validation1 = finaldata1.sample(frac = 0.2, random_state=random_seeds[0])#更改随机种子实现多次留出法验证
validation_before = pd.concat([validation0,validation1],axis=0)
train_before = pd.concat([finaldata0,finaldata1],axis=0).drop(validation_before.index,axis=0,inplace=False)
#打乱
origin_index = np.array(validation_before.index)
random.seed(8931)
random.shuffle(origin_index)
validation = validation_before.loc[origin_index].reset_index(drop=True,inplace=False)
origin_index2 = np.array(train_before.index)
random.seed(8931)
random.shuffle(origin_index2)
train = train_before.loc[origin_index2].reset_index(drop=True,inplace=False)
#trainset and validation set
y_train = train['LABEL']
x_train = train.drop(['LABEL','ent_id'],axis=1,inplace=False)#UID cant be one of features
y_valid = validation['LABEL']
x_valid = validation.drop(['LABEL','ent_id'],axis=1,inplace=False)#UID cant be one of features
y_train_all = finaldata['LABEL']
x_train_all = finaldata.drop(['LABEL','ent_id'],axis=1,inplace=False)

'''
构建ANN模型
construct the model
'''
#label onehot
label_enc = OneHotEncoder()
label_enc.fit(train[['LABEL']])
label_trainset = pd.DataFrame(label_enc.transform(train[['LABEL']]).toarray(),columns = ['y_0','y_1'])

layers1 = [10,10,2]
alpha1 = [0.005,np.exp(-30),np.exp(-60)]
epoch1 = 50
num_batch1 = 2000
random_state=11
estimator = ann.annClass(x_train,label_trainset,layers1,random_state)
for a in alpha1:   
    estimator.train(a,epoch1,num_batch1)
probability = estimator.predict(x_valid)[:,1]
print(roc_auc_score(y_valid,probability))
RocCurveDisplay.from_predictions(y_valid,probability)
plt.show()

'''
构建SVM模型
construct the model
'''  
estimator1 = svm.SVC(C=100,kernel='rbf',gamma=2**(-15),probability=True)
estimator1.fit(x_train,y_train)
y_threshold = estimator1.predict_proba(x_valid)[:,1]
print(roc_auc_score(y_valid,y_threshold))
RocCurveDisplay.from_predictions(y_valid,y_threshold)
plt.show()
#利用全部训练集重新训练
estimator1.fit(x_train_all,y_train_all)

'''
构建测试集，并预测
construct the test-set, and predict
'''
#############################
#这边财务指标需要把2018年换为2019年，2019年换为2020年
#新闻数量同上
#违约记录同上,并进行标准化和正态化
#############################
#financial indicator
key_set = bf_k.copy()
key_set = key_set.merge(financial_2019,how='left',on='ent_id',suffixes=('_y1','_y2'),validate='one_to_one')
key_set = key_set.merge(financial_2020,how='left',on='ent_id',suffixes=('_y1','_y2'),validate='one_to_one')
key_set.drop(['report_year_y1','report_year_y2'],axis=1,inplace=True)
for cl in key_set.columns:
    if cl not in ['indus','ent_id']:
        for val in key_set['indus'][key_set['indus'].notna()].unique():
            nan_index = key_set[cl][(key_set['indus']==val)&(key_set[cl].isnull())].index
            mean = key_set[cl][(key_set['indus']==val)&(key_set[cl].notna())].mean()
            key_set.loc[nan_index,[cl]]=mean#这里要用loc，用preset[cl][condition]=mean无法赋值
        cl_mean = key_set[cl][key_set[cl].notna()].mean()
        indus_nan_index = (key_set[cl][key_set[cl].isnull()]).index
        key_set.loc[indus_nan_index,[cl]]=cl_mean
#news
for val_imp in news['impscore'].unique():
    for year in news['publish_year'].unique():
        if year!=2018:
            new_cl_name = 'imp_'+str(val_imp)+'_cnt'+str(year-1)#保持特征名一致
            news_imp_selected = news_grouped_impor[['ent_id','newscode']][(news_grouped_impor['impscore']==val_imp)&(news_grouped_impor['publish_year']==year)]
            news_imp_selected.rename({'newscode':new_cl_name},axis=1,inplace=True)
            key_set = key_set.merge(news_imp_selected,how='left',on='ent_id',validate='1:1')
news_grouped_index_temp = news[['ent_id','newscode','indextype','publish_year']].groupby(['ent_id','publish_year','indextype'],as_index=False).count()
for val_index in news['indextype'].unique():
    for year in news['publish_year'].unique():
        if year!=2018:
            new_cl_name = 'index_'+str(val_index)+'_cnt'+str(year-1)
            news_index_selected = news_grouped_index_temp[['ent_id','newscode']][(news_grouped_index_temp['indextype']==val_index)&(news_grouped_index_temp['publish_year']==year)]
            news_index_selected.rename({'newscode':new_cl_name},axis=1,inplace=True)
            key_set = key_set.merge(news_index_selected,how='left',on='ent_id',validate='1:1')
for cl in news_grouped.columns:
    if cl!='ent_id':
        key_set[cl].fillna(0,inplace=True)#fillna不支持二级切片，preset[cl][condition]不行
#default record
for ac_year in  default_grouped['acu_year'].unique():
    if ac_year==2020:#2020加入测试集
        new_cl_name = 'acu_cnt'+str(ac_year-1)
        default_grouped_selected = default_grouped[['ent_id','acu_date']][default_grouped['acu_year']==ac_year]
        default_grouped_selected.rename({'acu_date':new_cl_name},axis=1,inplace=True)
        key_set = key_set.merge(default_grouped_selected,how='left',on='ent_id',validate='1:1')
for cl in default.columns:
    if cl!='ent_id':
        key_set[cl].fillna(0,inplace=True)
#standardize
indicator_list = pd.Series(profit_list+solvency_lsit+capital_list+operation_list)
indicator_list = (indicator_list+'_y1').tolist()+(indicator_list+'_y2').tolist()
for cl in indicator_list:
    for val_ind in key_set['indus'][key_set['indus'].notna()].unique():
        indus_index = key_set[cl][key_set['indus']==val_ind].index
        val_mean = key_set.loc[indus_index,[cl]].mean()
        if len(indus_index)>1:
            val_std = key_set.loc[indus_index,[cl]].std()
            if val_std[0]==0: val_std = 1
        else:val_std=1#如果只有一个元素，std=nan
        key_set.loc[indus_index,[cl]] = (key_set.loc[indus_index,[cl]]-val_mean)/val_std
    indus_nan_index = key_set[cl][key_set['indus'].isnull()].index
    indus_nan_mean = key_set.loc[indus_nan_index,[cl]].mean()
    if len(indus_nan_index)>1:
        indus_nan_std = key_set.loc[indus_nan_index,[cl]].std()
        if indus_nan_std[0]==0: indus_nan_std = 1
    else:indus_nan_std=1
    key_set.loc[indus_nan_index,[cl]] = (key_set.loc[indus_nan_index,[cl]]-indus_nan_mean)/indus_nan_std
#news & regcap
for cl in news_grouped.columns.tolist()+['regcap','optime']:#必须转为list，不然是视为series+字符串处理方法
    if cl!='ent_id' :
        key_set[cl] = np.log(key_set[cl]+1)
for cl in news_grouped.columns.tolist()+['regcap','optime']:
    if cl!='ent_id' :
        for val_ind in key_set['indus'][key_set['indus'].notna()].unique():
            indus_index = key_set[cl][key_set['indus']==val_ind].index
            val_mean = key_set.loc[indus_index,[cl]].mean()
            if len(indus_index)>1:
                val_std = key_set.loc[indus_index,[cl]].std()
                if val_std[0]==0: val_std = 1#如果都为0，则std也为0，改为1
            else:val_std=1
            key_set.loc[indus_index,[cl]] = (key_set.loc[indus_index,[cl]]-val_mean)/val_std
        indus_nan_index = key_set[cl][key_set['indus'].isnull()].index
        indus_nan_mean = key_set.loc[indus_nan_index,[cl]].mean()
        if len(indus_nan_index)>1:
            indus_nan_std = key_set.loc[indus_nan_index,[cl]].std()#返回的是一个series
            if indus_nan_std[0]==0: indus_nan_std = 1
        else:indus_nan_std=1
        key_set.loc[indus_nan_index,[cl]] = (key_set.loc[indus_nan_index,[cl]]-indus_nan_mean)/indus_nan_std
testdata = key_set.copy()
x_test = testdata[fcm.very_high_feature]
'''x_test.info()
for cl in x_test.columns:
    fs = fcm.featureScore_single(x_test[cl])
    print(cl)
    print(fs)'''
estimation_svm = estimator1.predict_proba(x_test)[:,1]
prediction_svm = pd.DataFrame({'ent_id':testdata['ent_id'],'default_score':estimation_svm})
id_set = pd.read_csv('/Users/lieu/Documents/Code/MachineLearning/0dataCompetition/220513国泰君安发债企业违约风险预测/2022S-T3-1st-Training/answer.csv',
                      sep='|',usecols=['ent_id'])
answer = id_set.merge(prediction_svm,how='left',on='ent_id')#预测企业id和示例文件一致……
answer.to_csv('/Users/lieu/Documents/Code/MachineLearning/0dataCompetition/220513国泰君安发债企业违约风险预测/提交记录/answer.csv',
             index=False,float_format='%.10f',encoding='utf-8',sep='|')
