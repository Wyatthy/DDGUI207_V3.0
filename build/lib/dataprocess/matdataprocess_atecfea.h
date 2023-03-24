#pragma once
#include <mat.h>
#include <vector>
#include <map>
#include <QVector>
#include <QDebug>
#include "./lib/guiLogic/tools/searchFolder.h"


class MatDataProcess_ATECfea
{
public:
    MatDataProcess_ATECfea(std::string folder);
    ~MatDataProcess_ATECfea(){};
    void getFeaNFromMat(std::string matPath, std::vector<std::vector<float>> &data);
    void loadAllFeaNFromFolder(std::string folder, std::vector<std::vector<float>> &data);
    void getFeaNumFromFolder(std::string folder);

    int feaNum;
    QVector<QVector<QVector<float>>> dataFrames;
private:
    SearchFolder *dirTools = new SearchFolder();
};


