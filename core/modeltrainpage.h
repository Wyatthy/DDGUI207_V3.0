#ifndef MODELTRAINPAGE_H
#define MODELTRAINPAGE_H


#include <QObject>
#include <QMessageBox>
#include <QFileDialog>
#include <QGraphicsScene>
#include <QGraphicsView>
#include "./lib/guiLogic/customWidget/imagewidget.h"
#include <windows.h>
#include <mat.h>
#include "ui_MainWindow.h"
#include "./lib/guiLogic/bashTerminal.h"
#include "./lib/guiLogic/modelInfo.h"
#include "./lib/guiLogic/datasetInfo.h"
#include "./lib/guiLogic/projectsInfo.h"
#include "./lib/guiLogic/tools/searchFolder.h"

class ModelTrainPage:public QObject
{
    Q_OBJECT
public:
    Ui_MainWindow *ui;
    BashTerminal *terminal;
    DatasetInfo *datasetInfo;
    ModelInfo *modelInfo;
    ProjectsInfo *projectsInfo;
    BashTerminal *train_terminal;

    QString choicedDatasetPATH;
    QString projectPath;
    QProcess *processTrain;
    std::vector<std::string> modelTypes={"HRRP","AFS","FewShot"};
    std::string modelType = "";
    std::string modelName = "";
    std::string dataType = "";

    QString cmd="";
    QString time = "";
    QString batchSize = "";
    QString epoch = "";
    QString saveModelName = "";
    QString old_class_num = "";
    QString reduce_sample = "";
    QString pretrain_epoch = "";
    QString cil_data_dimension = "";
    
    ModelTrainPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo,
                   ModelInfo *globalModelInfo,ProjectsInfo *globalProjectInfo);
    void refreshGlobalInfo();
    void uiInitial();
    void execuCmd(QString cmd);   // 开放在终端运行命令接口
    void showTrianResult();
//    int getDataLen(std::string dataPath);
//    int getDataClassNum(std::string dataPath, std::string specialDir);

    // 为了兼容win与linux双平台
    bool showLog=false;
    #if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    QString bashApi = "powershell";            // "Windows" or "Linux"
    #else
    QString bashApi = "bash";            // "Windows" or "Linux"
    #endif

public slots:
    void startTrain();
    void stopTrain();
    void monitorTrainProcess();
    void changeTrainType();
    void editModelFile();
//    void chooseOldClass();



private:
    // 缩放图像组件
    std::map<QGraphicsView*, ImageWidget*> all_Images;     // 防止内存泄露
    void recvShowPicSignal(QPixmap image, QGraphicsView* graphicsView);

};



#endif // MODELTRAINPAGE_H
