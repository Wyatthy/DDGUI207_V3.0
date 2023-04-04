#include "atecresultparser.h"

ATECResultParser::ATECResultParser()
{
    qRegisterMetaType<QVariant>("QVariant");
}

void ATECResultParser::run(){
    int featureIdx = 2;//暂时先取第一个特征
    int mydataset_size=hrrpDataset->labels.size();
    QVector<float> singleMapFeaFrame;
    QVector<float> singleTradFeaFrame;
    QVector<QVector<float>> featureFrames;
    qDebug()<<"ATECResultParser::run tradFeaDataset.data.size()="<<tradFeaDataset->data.size();
    for(int i=0;i<mydataset_size;i++){
        while(!startorstop_flag){};
        if(isInterruptionRequested()) break;
        while(!startOrstop){};
        QVector<float> singleHRRPFrame(hrrpDataset->data[i].begin(), hrrpDataset->data[i].end());
        singleMapFeaFrame.prepend(mapFeaDataset->feaData[i][featureIdx]);if(singleMapFeaFrame.size()>128) singleMapFeaFrame.removeLast();
        singleTradFeaFrame.prepend(tradFeaDataset->data[i][featureIdx]);if(singleTradFeaFrame.size()>128) singleTradFeaFrame.removeLast();
        featureFrames.clear();
        featureFrames.push_back(singleMapFeaFrame);
        featureFrames.push_back(singleTradFeaFrame);
        int preIdx = mapFeaDataset->predLabelIdx[i];
        int realIdx = mapFeaDataset->realLabelIdx[i];
        // std::vector<float> degrees={0.1,0.2,0.3,0.1,0.2,0.1};
        std::vector<float> degrees;
        // degrees.clear();
        for(int j=0;j<mapFeaDataset->degreeData[i].size();j++){
            degrees.push_back(mapFeaDataset->degreeData[i][j]);
        }
        // qDebug()<<"mapFeaDataset->degreeData[i].size()==="<<mapFeaDataset->degreeData[i].size();
        // if(i==0){
        //     for(int j=0;j<mapFeaDataset->degreeData[i].size();j++){
        //         qDebug()<<"mapFeaDataset->degreeData[i][j]="<<mapFeaDataset->degreeData[i][j];
        //     }
        // }
        // QVariant degreesQV;degreesQV.setValue(mapFeaDataset->degreeData[i]);
        QVariant degreesQV;degreesQV.setValue(degrees);

        emit sigATECResult(featureFrames,singleHRRPFrame,preIdx,realIdx,degreesQV);
        qDebug()<<"ATECResultParser::run  show";
        _sleep(1000);
    }

}

void ATECResultParser::setHRRPDataset(CustomDataset &dataset){
   hrrpDataset = &dataset;
}

void ATECResultParser::setTradFeaDataset(CustomDataset &dataset){
   tradFeaDataset = &dataset;
}

void ATECResultParser::setMapFeaDataset(MatDataProcess_atec *dataset){
   mapFeaDataset = dataset;
}

void ATECResultParser::startOrstop_slot(bool startorstop){
    startOrstop=startorstop;
    qDebug()<<"startOrstop="<<startOrstop;
    qDebug()<<"startOrstop_slot function is in thread:"<<QThread::currentThreadId();
}
void ATECResultParser::stopThread(){
    startorstop_flag = !startorstop_flag;
}
