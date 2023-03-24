#pragma once
#include "./lib/dataprocess/matdataprocess.h"
#include "./lib/dataprocess/matdataprocess_abfc.h"
#include "./lib/dataprocess/matdataprocess_rcs.h"
#include "./lib/dataprocess/matdataprocess_image.h"

#include <vector>
#include <map>

//CustomDataSet.data=F(The M of dataset's mat , dims(model.inputlayer)) 这是一个根据网络类型调整的数据集
class CustomDataset{
public:
    std::vector<std::vector<float>> data;
    std::vector<int> labels;
    std::map<std::string, int> class2label;
    std::vector<int> eachClassQuantity;
    CustomDataset(){}
    CustomDataset(std::string dataSetPath, bool dataProcess, std::string type, std::map<std::string, int> class2label,int inputLen,QString flag="TRA_DL",int modelIdx=1,std::vector<int> dataOrder=std::vector<int>())
        :class2label(class2label){
        QStringList flags = flag.split("_param");
        for(int i=0;i<flags.size();i++){
            qDebug()<<"flags[i]=="<<flags[i];
        }
        if(flags[0]=="RCS_infer"){//这样CustomDataset中单样本长度就是网络输入长度inputlen
            int windowsLength = flags[1].toInt();
            int windowsStep = flags[2].toInt();
            MatDataProcess_rcs matDataPrcs;
            matDataPrcs.loadAllDataFromFolder(dataSetPath, type, dataProcess, data, labels, class2label, inputLen, windowsLength, windowsStep);
        }
        else if(flags[0]=="IMAGE_infer"){
            int windowsLength = flags[1].toInt();
            int windowsStep = flags[2].toInt();
            MatDataProcess_image matDataPrcs;
            matDataPrcs.loadAllDataFromFolder(dataSetPath, type, dataProcess, data, labels, class2label, inputLen, windowsLength, windowsStep);
        }
        else{//根据网络模型输入层的长度inputlen做CustomDataset,把原单个样本复制或裁剪到inputlen长度。inputlen赋-1时等于原单个数据样本长度
            MatDataProcess matDataPrcs;
            matDataPrcs.loadAllDataFromFolder(dataSetPath, type, dataProcess, data, labels, class2label, inputLen);
        }
    }
    int size(){
        return labels.size();
    };
    //specially for FEA_RELE_abfc
    void getDataSpecifically(std::string theClass,int emIndex,float *dataf){
        int theClassIdx=class2label[theClass];
        std::map<std::string, int>::iterator iter;
        iter = class2label.begin();
        while(iter != class2label.end()) {
            qDebug() << QString::fromStdString(iter->first) << " : " << iter->second ;
            iter++;
        }
        int globalIndexOnValidationSet = 0;
        for(int i=0;i<theClassIdx;i++){
            globalIndexOnValidationSet += eachClassQuantity[i];
        }
        //TODO Ҫ��� ֮����ݱ�����
        if(emIndex>=50)emIndex-=50;
        globalIndexOnValidationSet += emIndex;
        for(int i=0;i<data[globalIndexOnValidationSet].size();i++){
            dataf[i]=data[globalIndexOnValidationSet][i];
        }
    }
};
