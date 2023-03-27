#include "matdataprocess_rcs.h"

void MatDataProcess_rcs::oneNormalization(std::vector<float> &list){
    //特征归一化
    float dMaxValue = *max_element(list.begin(),list.end());  //求最大值
    //std::cout<<"maxdata"<<dMaxValue<<'\n';
    float dMinValue = *min_element(list.begin(),list.end());  //求最小值
    //std::cout<<"mindata"<<dMinValue<<'\n';
    for (int f = 0; f < list.size(); ++f) {
        list[f] = (1-0)*(list[f]-dMinValue)/(dMaxValue-dMinValue+1e-8)+0;//极小值限制
    }
}

void MatDataProcess_rcs::getAllDataFromMat(std::string matPath,bool dataProcess,std::vector<std::vector<float>> &data,std::vector<int> &labels,int label,int inputLen){
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    int* matdata;
    pMatFile = matOpen(matPath.c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(MatDataProcess_rcs:getAllDataFromMat)文件指针空!!!";
        return;
    }
    pMxArray = matGetNextVariable(pMatFile, NULL);
    if(!pMxArray){
        qDebug()<<"(MatDataProcess:getAllDataFromMat).mat文件变量没找到!!!("<<QString::fromStdString(matPath);
        return;
    }
    matdata = (int*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //行数 RCS只有一行
    int N = mxGetN(pMxArray);  //列数 
    int allDataNum=(N-windowLen)/windowStep+1;
    int win_start = 0;
    for(int i=0;i<allDataNum;i++){
        std::vector<float> onesmp;//存当前遍历的一个样本
        for(int j=0;j<windowLen;j++){
            onesmp.push_back(matdata[win_start+j]);
        }
        win_start+=windowStep;
        if(dataProcess) oneNormalization(onesmp);//归一化
        std::vector<float> temp;
        int numberOfcopies=inputLen/windowLen; //复制次数=网络的输入长度/一个样本数据的长度
        for(int j=0;j<inputLen;j++){
            //如果inputLen比N还小，不会报错，但显然数据集和模型是不对应的吧，得到的推理结果应会很难看
            temp.push_back(onesmp[j/numberOfcopies]);//64*128,对应训练时(128,64,1)的输入维度
            //temp.push_back(onesmp[j%M]);//128*64,对应训练时(64,128,1)的输入维度
        }
        if(inputLen < 0){       //此时表示上层想要原数据长度
            data.push_back(onesmp);
        }else{
            data.push_back(temp);
        }
        labels.push_back(label);
    }
    qDebug()<<"(MatDataProcess_rcs:getAllDataFromMat)labels.Size（）=="<<labels.size();
}

void MatDataProcess_rcs::loadAllDataFromFolder(std::string datasetPath,std::string type,bool dataProcess,std::vector<std::vector<float>> &data,
                           std::vector<int> &labels,std::map<std::string, int> &class2label,int inputLen,int windowLength,int windowStep){
    this->windowLen = windowLength;
    this->windowStep = windowStep;
    SearchFolder *dirTools = new SearchFolder();
    std::vector<std::string> subDirs;
    dirTools->getDirsplus(subDirs, datasetPath);
    //换序到subDirsQ
    QVector<QString> subDirsQ;
    for (const auto& folderName : subDirs) {
        QString folderNameQ = QString::fromStdString(folderName);
        if (folderNameQ.contains("DT")) {
            subDirsQ.push_front(folderNameQ);
        } else {
            subDirsQ.push_back(folderNameQ);
        }
    }
    for(auto &subDir: subDirsQ){
        std::vector<std::string> fileNames;
        std::string subDirPath = datasetPath+"/"+subDir.toStdString();
        dirTools->getFilesplus(fileNames, type, subDirPath);
        for(auto &fileName: fileNames){
            qDebug()<<QString::fromStdString(subDirPath)<<"/"<<QString::fromStdString(fileName)<<" label:"<<class2label[subDir.toStdString()];
            getAllDataFromMat(subDirPath+"/"+fileName,dataProcess,data,labels,class2label[subDir.toStdString()],inputLen);
        }
    }
    return;
}

void MatDataProcess_rcs::getDataFromMat(std::string targetMatFile,int emIdx,bool dataProcess,float *data,int inputLen,int windowlen,int windowstep){
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    // 读取.mat文件（例：mat文件名为"initUrban.mat"，其中包含"initA"）
    int* matdata;
    pMatFile = matOpen(targetMatFile.c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(MatDataProcess_rcs:getDataFromMat)文件指针空!!!";
        return;
    }
    pMxArray = matGetNextVariable(pMatFile, NULL);
    if(!pMxArray){
        qDebug()<<"(MatDataProcess:getAllDataFromMat).mat文件变量没找到!!!("<<QString::fromStdString(targetMatFile);
        return;
    }
    matdata = (int*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //行数
    int N = mxGetN(pMxArray);  //列数
    if(emIdx>(N-windowlen)/windowstep+1) emIdx=(N-windowlen)/windowstep+1;  
    std::vector<float> onesmp;//存当前样本
    for(int i=0;i<windowlen;i++){
        onesmp.push_back(matdata[(emIdx-1)*windowstep+i]);
    }
    if(dataProcess) oneNormalization(onesmp);
    int numberOfcopies=inputLen/windowlen; //复制次数=网络的输入长度/一个样本数据的长度
    for(int i=0;i<inputLen;i++){
        //data[i]=onesmp[i%M];//matlab按列存储
        data[i]=onesmp[i/numberOfcopies];//网络如果是(128,64,1),应该64+64+64+64+...输入引擎,而不是128+128+...
    }

}
