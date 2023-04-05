#ifndef ATECRESULTPARSER_H
#define ATECRESULTPARSER_H

#include <mat.h>
#include <math.h>
#include <stdio.h>
#include <io.h>
#include <QDebug>
#include <QThread>
#include <QVariant>
#include <queue>
#include "./lib/dataprocess/matdataprocess.h"
#include "./lib/dataprocess/customdataset.h"
#include "./lib/dataprocess/MatDataProcess_atec.h"



class ATECResultParser:public QThread
{
    Q_OBJECT   //申明需要信号与槽机制支持
public:
    ATECResultParser();
    ~ATECResultParser(){
        requestInterruption();
        quit();
        wait();
    };
    void run();
    void setHRRPDataset(CustomDataset &dataset);
    void setTradFeaDataset(CustomDataset &dataset);
    void setMapFeaDataset(MatDataProcess_atec *dataset);
    void stopThread();
    bool startOrstop=true;


public
slots:
    void startOrstop_slot(bool);
signals:
    void sigATECResult(QVector<QVector<float>>,QVector<float>,int,int,QVariant);


private:

    CustomDataset* hrrpDataset;
    CustomDataset* tradFeaDataset;
    MatDataProcess_atec *mapFeaDataset;
    bool startorstop_flag = true;


};

#endif // ATECRESULTPARSER_H
