#include "matdataprocess_atec.h"

void MatDataProcess_atec::getAllDataFromMat(std::string matPath){
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    double* matdata;
    pMatFile = matOpen(matPath.c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(MatDataProcess_atec::getAllDataFromMat)文件指针空!";
        return;
    }
    pMxArray = matGetNextVariable(pMatFile, NULL);
    if(!pMxArray){
        qDebug()<<"(MatDataProcess_atec::getAllDataFromMat).mat文件变量没找到! matpath="<<QString::fromStdString(matPath);
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //行数
    int N = mxGetN(pMxArray);  //列数
    feaNum = M - classNum - 2;
    for(int i=0;i<N;i++){
        QVector<float> thisFea;
        QVector<float> thisDegree;

        for(int j=0;j<feaNum;j++)
            thisFea.push_back(matdata[i*M+j]);
        for(int j=feaNum;j<M-2;j++)
            thisDegree.push_back(matdata[i*M+j]);
        predLabelIdx.push_back(matdata[i*M+M-2]);
        realLabelIdx.push_back(matdata[i*M+M-1]);
        feaData.push_back(thisFea);
        degreeData.push_back(thisDegree);
    }
}

MatDataProcess_atec::MatDataProcess_atec(std::string mapFeaDatasetlPath){
    //mapFeaDatasetlPath=xxx/test_result
    SearchFolder *dirTools = new SearchFolder();
    // 寻找子文件夹 
    std::vector<std::string> subDirs;
    dirTools->getDirsplus(subDirs, mapFeaDatasetlPath);

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
    classNum = subDirsQ.size();
    // for(int i=0;i<subDirsQ.size();i++)   label2class[i]=subDirsQ[i].toStdString();
    // for(auto &item: label2class)   class2label[item.second] = item.first;

    for(auto &subDir: subDirsQ){
        // 寻找每个子文件夹下的样本文件
        std::vector<std::string> fileNames;
        std::string subDirPath = mapFeaDatasetlPath+"/"+subDir.toStdString();
        dirTools->getFilesplus(fileNames, ".mat", subDirPath);
        for(auto &fileName: fileNames){
            //qDebug()<<QString::fromStdString(subDirPath)<<"/"<<QString::fromStdString(fileName)<<" label:"<<class2label[subDir.toStdString()];
            getAllDataFromMat(subDirPath + "/" + fileName);
        }
    }
    return;
}
