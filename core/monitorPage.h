#ifndef MONITORPAGE_H
#define MONITORPAGE_H

#include <QObject>
#include <thread>
#include "./core/projectsWindow/chart.h"
#include "ui_MainWindow.h"
#include "./lib/guiLogic/bashTerminal.h"
#include "./lib/guiLogic/modelInfo.h"
#include "./lib/guiLogic/datasetInfo.h"
#include "./lib/guiLogic/projectsInfo.h"
#include "./lib/guiLogic/tools/socketserver.h"
#include "./lib/guiLogic/tools/socketclient.h"
#include "./lib/guiLogic/tools/atecresultparser.h"
#include "./lib/algorithm/inferthread.h"
#include "./lib/dataprocess/MatDataProcess_atec.h"
#include "./lib/dataprocess/customdataset.h"


class MonitorPage : public QObject
{
    Q_OBJECT
public:
    MonitorPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo,ModelInfo *globalModelInfo,ProjectsInfo *globalProjectInfo);
    void startListen();
    void stopSend();
    void simulateSend();
    SocketServer* server;
    InferThread* inferThread=nullptr;
    SocketClient* client;
    ATECResultParser* atecPerformer;
    void paintLabel();
    std::queue<std::vector<float>> sharedQue;
    TrtInfer* trtInfer;
    void refresh();
    //bool eventFilter(QObject *watched, QEvent *event);
    ~MonitorPage();
    int rightNum =0;
    int inferedNum =0;
    CustomDataset myDataset ;
    // CustomDataset mapFeaDataset;
    MatDataProcess_atec *mapFeaDataset;
    CustomDataset tradFeaDataset;

public
slots:
    void slotEnableSimulateSignal();
    void slotShowInferResult(QVector<float>,int,QVariant);
    void slotShowRealClass(int);
    void slotShowATECResult(QVector<QVector<float>>, QVector<float>, int, int, QVariant);
signals:
    void startOrstop_sig(bool);

private:
    Ui_MainWindow *ui;
    BashTerminal *terminal;
    std::string choicedDatasetPATH="";
    std::string choicedModelPATH="";
    ModelInfo *modelInfo;
    DatasetInfo *datasetInfo;
    ProjectsInfo *projectsInfo;
    void signalVisualize(QVector<float> dataFrameQ);

    std::map<int, std::string> label2class;
    std::map<std::string, int> class2label;

    bool getSigFromClient = false;
    std::vector<int> sigsFromClient;
    int fallBackValue = 0;
    int num_fallBackValueUsed = 0;
    bool getSigFromInfer = false;
    std::string currtDataType;
    std::string currtModelType;
};

#endif // MONITORPAGE_H
