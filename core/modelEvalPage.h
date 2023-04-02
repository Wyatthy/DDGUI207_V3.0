#ifndef MODELEVALPAGE_H
#define MODELEVALPAGE_H

#include <vector>
#include <string>
#include <map>
#include <QDir>
#include <QObject>
#include <QThread>
#include <QGraphicsView>
#include "qlistwidget.h"
#include "ui_MainWindow.h"
#include "./lib/guiLogic/bashTerminal.h"

#include "./lib/guiLogic/modelInfo.h"
#include "./lib/guiLogic/datasetInfo.h"
#include "./lib/guiLogic/projectsInfo.h"
#include "./core/projectsWindow/chart.h"
#include "./lib/guiLogic/tools/searchFolder.h"
#include "./lib/guiLogic/customWidget/imagewidget.h"

#include "lib/algorithm/trtinfer.h"

#undef slots
#include <Python.h>
#include "arrayobject.h"
#define slots Q_SLOTS

class ModelEvalPage:public QObject{
    Q_OBJECT
public:
    ModelEvalPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo, ModelInfo *globalModelInfo, ProjectsInfo *globalProjectInfo);
    ~ModelEvalPage();

    std::map<int, std::string> label2class;
    std::map<std::string, int> class2label;

    void disDegreeChart(QString &classGT, std::vector<float> &degrees, std::map<int, std::string> &classNames);
    void testOneSample_ui();
    friend void testOneSample_ui2(ModelEvalPage dv);
    int emIndex{0};

    Ui_MainWindow *ui;
    BashTerminal *terminal;

public slots:
    void refreshGlobalInfo();

    void on_comboBox_sampleType(QString s);
    void on_comboBox_chosFile(QString s);
    // 针对全部样本
    void testAllSample();
    void execuCmdProcess(QProcess *processInfer, QString cmd);
    void processSampleInferFinished();   // 可视化脚本执行结束事件 
    void processDatasetInferFinished();
    // 针对单样本
    void takeSample();
    void testOneSample();

    void slot_showDegreesChartA();
    void slot_showDegreesChartB();
    void slot_setClassA(QString);
    void slot_setClassB(QString);

    void slot_updateSelectedCategoriesA();
    void slot_updateSelectedCategoriesB();


signals:
    void stating(std::string choicedsamplePATH,std::string choicedmodelPATH,std::vector<float> &degrees);

private:
    DatasetInfo *datasetInfo;
    ModelInfo *modelInfo;
    ProjectsInfo *projectsInfo;

    std::string choicedDatasetPATH="";
    std::string choicedModelPATH="";
    std::string choicedSamplePATH;

    std::string choicedClass;
    std::string choicedFileInClass;
    // 不同平台下文件夹搜索工具
    SearchFolder *dirTools = new SearchFolder();

    // 缩放图像组件
    std::map<QGraphicsView*, ImageWidget*> all_Images;     // 防止内存泄露
    void recvShowPicSignal(QPixmap image, QGraphicsView* graphicsView);
    QGraphicsScene *qgraphicsScene = new QGraphicsScene; //要用QGraphicsView就必须要有QGraphicsScene搭配着用
    
    //推理算法
    TrtInfer *trtInfer;
    QThread *qthread1;
    //优化模型的推理进程
    QProcess *processDatasetInfer;
    QProcess *processSampleInfer;

    //eval页面调用python画混淆矩阵
    PyObject *pModule_drawConfusionMatrix,*pModule_optimizeInfer;
    PyObject *pFunc_drawConfusionMatrix,*pFunc_optimizeInfer;
    PyObject *PyArray,*args_draw,*args_opti;
    PyArrayObject* pRet_draw;
    PyObject * pRet_opti;
    
    // 选择模型结构的xml文件、预览图像路径 // FIXME 后期需要结合系统
    std::string modelStructXmlPath;
    QString modelStructImgPath;
    QString modelCheckpointPath;

    QString camImgsSavePath;
    QString condaPath;
    QString condaEnvName;
    QString pythonApiPath;

    QListWidget *testListWidgetB = new QListWidget;
    QListWidget *testListWidgetA = new QListWidget;
    QLineEdit *cliLineEdit = new QLineEdit;
    QStringList selectedCategoriesA{};
    QStringList selectedCategoriesB{};
    int classA = 0;
    int classB = 0;
    std::vector<std::vector<std::vector<float>>> degrees_matrix_copy;//[c,c,n]



};

#endif // MODELEVALPAGE_H
