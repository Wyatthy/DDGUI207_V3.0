#include "matdataprocess_image.h"

void MatDataProcess_image::oneNormalization(std::vector<float> &list){
    //特征归一化
    float dMaxValue = *max_element(list.begin(),list.end());  //求最大值
    //std::cout<<"maxdata"<<dMaxValue<<'\n';
    float dMinValue = *min_element(list.begin(),list.end());  //求最小值
    //std::cout<<"mindata"<<dMinValue<<'\n';
    for (int f = 0; f < list.size(); ++f) {
        list[f] = (1-0)*(list[f]-dMinValue)/(dMaxValue-dMinValue+1e-8)+0;//极小值限制
    }
}

void MatDataProcess_image::getAllDataFromMat(std::string matPath,bool dataProcess,std::vector<std::vector<float>> &data,std::vector<int> &labels,int label,int inputLen){
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    // 读取.mat文件（例：mat文件名为"initUrban.mat"，其中包含"initA"）
    double* matdata;
    pMatFile = matOpen(matPath.c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(MatDataProcess_image:getAllDataFromMat)文件指针空！！！！！！";
        return;
    }
    pMxArray = matGetNextVariable(pMatFile, NULL);
    if(!pMxArray){
        qDebug()<<"(MatDataProcess_image:getAllDataFromMat).mat文件变量没找到!!!("<<QString::fromStdString(matPath);
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //行数
    int N = mxGetN(pMxArray);  //列数
    int allDataNum=(N-windowLen)/windowStep+1;
    for(int k=0;k<allDataNum;k++){
        std::vector<float> onesmp;//存当前遍历的一个样本
        for(int i=0;i<windowLen;i++){
            std::vector<float> onecol;//存当前窗口的一列
            for(int j=0;j<M;j++){
                onecol.push_back(matdata[(k*windowStep+i)*M+j]);
            }
            if(dataProcess) oneNormalization(onecol);
            onesmp.insert(onesmp.end(), onecol.begin(), onecol.end());
        }

        std::vector<float> temp;
        int numberOfcopies=inputLen/onesmp.size(); //复制次数=网络的输入长度/一个样本数据的长度

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
    // qDebug()<<"(MatDataProcess_image:getAllDataFromMat)matVariable=="<<QString::fromStdString(matVariable);
}

void MatDataProcess_image::loadAllDataFromFolder(std::string datasetPath,std::string type,bool dataProcess,std::vector<std::vector<float>> &data,
                           std::vector<int> &labels,std::map<std::string, int> &class2label,int inputLen,int windowLength,int windowStep){
    this->windowLen = windowLength;
    this->windowStep = windowStep;
    SearchFolder *dirTools = new SearchFolder();
    // 寻找子文件夹 
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
        // 寻找每个子文件夹下的样本文件
        std::vector<std::string> fileNames;
        std::string subDirPath = datasetPath+"/"+subDir.toStdString();
        dirTools->getFilesplus(fileNames, type, subDirPath);
        for(auto &fileName: fileNames){
            //qDebug()<<QString::fromStdString(subDirPath)<<"/"<<QString::fromStdString(fileName)<<" label:"<<class2label[subDir.toStdString()];
            getAllDataFromMat(subDirPath + "/" + fileName,dataProcess,data,labels,class2label[subDir.toStdString()],inputLen);
        }
    }
    return;
}

void MatDataProcess_image::getDataFromMat(std::string targetMatFile,int emIdx,bool dataProcess,float *data,int inputLen,int windowlen,int windowstep){    
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    // 读取.mat文件（例：mat文件名为"initUrban.mat"，其中包含"initA"）
    double* matdata;
    pMatFile = matOpen(targetMatFile.c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(MatDataProcess_image:getDataFromMat)文件指针空！！！！！！";
        return;
    }
    pMxArray = matGetNextVariable(pMatFile, NULL);
    if(!pMxArray){
        qDebug()<<"(MatDataProcess:getAllDataFromMat).mat文件变量没找到!!!("<<QString::fromStdString(targetMatFile);
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //行数
    int N = mxGetN(pMxArray);  //列数
    int allDataNum=(N-windowlen)/windowstep+1;
    emIdx = emIdx>allDataNum?allDataNum-1:emIdx;//说明是随机数

    std::vector<float> onesmp;//存当前遍历的一个样本
    for(int i=0;i<windowlen;i++){
        std::vector<float> onecol;//存当前窗口的一列
        for(int j=0;j<M;j++){
            onecol.push_back(matdata[(emIdx*windowlen+i)*M+j]);
        }
        if(dataProcess) oneNormalization(onecol);
        onesmp.insert(onesmp.end(), onecol.begin(), onecol.end());
    }

    int numberOfcopies=inputLen/onesmp.size(); //复制次数=网络的输入长度/一个样本数据的长度
    for(int i=0;i<inputLen;i++){
        //data[i]=onesmp[i%M];//matlab按列存储
        data[i]=onesmp[i/numberOfcopies];//网络如果是(128,64,1),应该64+64+64+64+...输入引擎,而不是128+128+...
    }

}
