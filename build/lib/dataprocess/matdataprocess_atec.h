#pragma once
#include <mat.h>
#include <vector>
#include <map>
#include <QVector>
#include <QDebug>
#include "./lib/guiLogic/tools/searchFolder.h"

class MatDataProcess_atec
{
public:
    MatDataProcess_atec(std::string folder);
    ~MatDataProcess_atec(){};
    void getAllDataFromMat(std::string matPath);

    QVector<QVector<float>> feaData;
    QVector<QVector<float>> degreeData;
    QVector<int> realLabelIdx;
    QVector<int> predLabelIdx;
private:
    SearchFolder *dirTools = new SearchFolder();
    int classNum;
    int feaNum;

    std::map<int, std::string> label2class;
    std::map<std::string, int> class2label;
};


