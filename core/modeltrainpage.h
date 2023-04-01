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
#include "qlistwidget.h"
#include "ui_MainWindow.h"
#include "./lib/guiLogic/bashTerminal.h"
#include "./lib/guiLogic/modelInfo.h"
#include "./lib/guiLogic/datasetInfo.h"
#include "./lib/guiLogic/projectsInfo.h"
#include "./lib/dataprocess/MatDataProcess_ATECfea.h"
#include "core/projectsWindow/chart.h"


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

    QListWidget *cliListWidget;
    QLineEdit *cliLineEdit;

    QString choicedDatasetPATH;
    QString projectPath;
    QProcess *processTrain;
    // std::vector<std::string> modelTypes={"HRRP","AFS","FewShot"};

    QString cmd="";
    QString time = "";
    QString batchSize = "";
    QString epoch = "";
    QString saveModelName = "";
    QString old_class_num = "";
    QString reduce_sample = "";
    QString pretrain_epoch = "";
    QString cil_data_dimension = "";
    QString selectedCategories = "";
    ModelTrainPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo,
                   ModelInfo *globalModelInfo,ProjectsInfo *globalProjectInfo);
    void refreshGlobalInfo();
    void refreshTrainResult();
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
    QGraphicsScene *qgraphicsScene = new QGraphicsScene; //要用QGraphicsView就必须要有QGraphicsScene搭配着用

    void showATECfeatrend();
    std::string trainingProjectName,trainingProjectPath,trainingDataType;

    int dataDimension;
    std::string modelTypeOfCurrtProject = "";
    std::string shotModelType = "";
    QString shotModelAlgorithm = "";
    std::string trainningModelType = "";
    std::string modelName = "";
    std::string dataType = "";
    SearchFolder *dirTools = new SearchFolder();
    QString currtPID = "";
};



#endif // MODELTRAINPAGE_H
