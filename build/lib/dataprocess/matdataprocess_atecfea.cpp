#include "matdataprocess_atecfea.h"


void MatDataProcess_ATECfea::getFeaNFromMat(std::string matPath, std::vector<std::vector<float>> &data){
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    double* matdata;
    pMatFile = matOpen(matPath.c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(MatDataProcess_ATECfea:getFeaNFromMat)文件指针空!";
        return;
    }
    pMxArray = matGetNextVariable(pMatFile, NULL);
    if(!pMxArray){
        qDebug()<<"(MatDataProcess_ATECfea:getFeaNFromMat).mat文件变量没找到! matpath="<<QString::fromStdString(matPath);
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //行数
    int N = mxGetN(pMxArray);  //列数

    for(int i=0;i<N;i++){
        for(int j=0;j<feaNum;j++)
            data[j].push_back(matdata[i*M+j]);
    }
}

void MatDataProcess_ATECfea::loadAllFeaNFromFolder(std::string folder, std::vector<std::vector<float>> &data){
    // 寻找子文件夹 
    std::vector<std::string> subDirs;
    dirTools->getDirsplus(subDirs, folder);

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
        std::string subDirPath = folder+"/"+subDir.toStdString();
        dirTools->getFilesplus(fileNames, "mat", subDirPath);
        for(auto &fileName: fileNames){
            // qDebug()<<QString::fromStdString(subDirPath)<<"/"<<QString::fromStdString(fileName);
            getFeaNFromMat(subDirPath + "/" + fileName, data);
        }
    }
    return;
}

MatDataProcess_ATECfea::MatDataProcess_ATECfea(std::string rootPath){
    std::string mapping_feature_path = rootPath+"/feature_save/mapping_feature";
    // std::string traditional_feature_path = rootPath+"/feature_save/traditional_feature";
    std::string traditional_feature_path = rootPath+"/train_feature";
    if(!dirTools->isExist(traditional_feature_path) || !dirTools->isExist(mapping_feature_path)){
        return;
    }ifSucc=true;

    std::vector<std::string> subDirs;
    dirTools->getDirsplus(subDirs, mapping_feature_path);
    std::string subDirPath = mapping_feature_path +"/"+subDirs[0];

    std::vector<std::string> fileNames;
    dirTools->getFilesplus(fileNames, "mat", subDirPath);
    std::string matfilePath = subDirPath + "/" + fileNames[0];

    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    pMatFile = matOpen(matfilePath.c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(MatDataProcess_ATECfea:init)文件指针空!";
        return;
    }
    pMxArray = matGetNextVariable(pMatFile, NULL);
    if(!pMxArray){
        qDebug()<<"(MatDataProcess_ATECfea:init).mat文件变量没找到!("<<QString::fromStdString(matfilePath);
        return;
    }
    int M = mxGetM(pMxArray);  //行数
    feaNum = M;

    std::vector<std::vector<float>> mapFeaMatrix(feaNum, std::vector<float>(0, 0));
    loadAllFeaNFromFolder(mapping_feature_path, mapFeaMatrix);
    std::vector<std::vector<float>> tradFeaMatrix(feaNum, std::vector<float>(0, 0));
    loadAllFeaNFromFolder(traditional_feature_path, tradFeaMatrix);
    for(int i=0;i<feaNum;i++){
        QVector<QVector<float>> dataFrame;
        QVector<float> mapFeaMatrix_iQ = QVector<float>(mapFeaMatrix[i].begin(), mapFeaMatrix[i].end());
        QVector<float> tradFeaMatrix_iQ = QVector<float>(tradFeaMatrix[i].begin(), tradFeaMatrix[i].end());
        dataFrame.push_back(mapFeaMatrix_iQ);
        dataFrame.push_back(tradFeaMatrix_iQ);
        dataFrames.push_back(dataFrame);
    }
}
