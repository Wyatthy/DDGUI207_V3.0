#ifndef MODELCAMPAGE_H
#define MODELCAMPAGE_H

#include <QObject>
#include <QString>
#include <QGraphicsView>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "ui_MainWindow.h"

#include "./lib/guiLogic/bashTerminal.h"
#include "./lib/guiLogic/tools/searchFolder.h"
#include "./lib/guiLogic/projectsInfo.h"
#include "./lib/guiLogic/customWidget/imagewidget.h"


class ModelCAMPage:public QObject{
    Q_OBJECT
public:
    ModelCAMPage(
        Ui_MainWindow *main_ui, 
        BashTerminal *bash_terminal, 
        ProjectsInfo *globalProjectInfo
    );
    ~ModelCAMPage();

    // 从xml加载5级模型结构的暴力方法，不优雅 // TODO
    void loadModelStruct_L1(QStringList &currLayers, std::map<std::string, std::string> &choicedLayers);
    void loadModelStruct_L2(QStringList &currLayers, std::map<std::string, std::string> &choicedLayers);
    void loadModelStruct_L3(QStringList &currLayers, std::map<std::string, std::string> &choicedLayers);
    void loadModelStruct_L4(QStringList &currLayers, std::map<std::string, std::string> &choicedLayers);
    void loadModelStruct_L5(QStringList &currLayers, std::map<std::string, std::string> &choicedLayers);


public slots:
    // 页面切换初始化
    void refreshGlobalInfo();
    void confirmModel();                // 模型选择

    void confirmData();                 // 数据样本选择
    void switchIndex();                 // 切换样本
    void nextIndex();                   // 下一个样本
    void refreshVisInfo();              // 刷新预览图像与可视化目标层
    void clearStructComboBox();         // 清空下拉框

    void confirmVis();                  // 可视化确认按钮事件
    void execuCmdProcess(QString cmd);  // 执行可视化脚本
    void processVisFinished();          // 可视化脚本执行结束事件 

    // 5级下拉框相关槽接口，过于暴力，不优雅 // TODO
    void on_comboBox_L1(QString choicedLayer);
    void on_comboBox_L2(QString choicedLayer);
    void on_comboBox_L3(QString choicedLayer);
    void on_comboBox_L4(QString choicedLayer);
    void on_comboBox_L5(QString choicedLayer);

    // 样本选择下拉框相关槽接口
    void on_comboBox_stage(QString choicedStage);
    void on_comboBox_label(QString choicedLabel);
    void on_comboBox_mat(QString choicedMat);


    /**************** 以下同样的实现代码(为了实现两个可视化对比) ****************/
    void confirmData_2();                 // 数据样本选择
    void switchIndex_2();                 // 切换样本
    void nextIndex_2();                   // 下一个样本
    void refreshVisInfo_2();              // 刷新预览图像与可视化目标层
    void clearStructComboBox_2();         // 清空下拉框

    void confirmVis_2();                  // 可视化确认按钮事件
    void execuCmdProcess_2(QString cmd);  // 执行可视化脚本
    void processVisFinished_2();          // 可视化脚本执行结束事件 


    // 5级下拉框相关槽接口，过于暴力，不优雅 // TODO
    void on_comboBox_L1_2(QString choicedLayer2);
    void on_comboBox_L2_2(QString choicedLayer);
    void on_comboBox_L3_2(QString choicedLayer);
    void on_comboBox_L4_2(QString choicedLayer);
    void on_comboBox_L5_2(QString choicedLayer);

    // 样本选择下拉框相关槽接口
    void on_comboBox_stage_2(QString choicedStage);
    void on_comboBox_label_2(QString choicedLabel);
    void on_comboBox_mat_2(QString choicedMat);
    /************************************************************************/


private:
    Ui_MainWindow *ui;
    BashTerminal *terminal;
    ProjectsInfo *projectsInfo;

    SearchFolder *dirTools = new SearchFolder();

    // 选择的数据集、模型、样本信息
    QString projectPath;

    QString choicedStage;           // 样本阶段, train or test or val
    QString choicedLabel;           // 样本标签
    QString choicedMatName;         // 样本名称
    QString choicedMatPATH;         // 样本路径
    QString choicedCamMethod;       // 可视化方法
    QString targetVisLayer;         // 可视化的目标层
    QString camImgsSavePath;        // 可视化结果保存路径
    
    std::map<std::string, std::string> choicedLayer;    // 选择的层级信息

    int choicedMatIndexBegin = -1;
    int choicedMatIndexEnd = -1;
    int maxMatIndex = -1;
    int currMatIndex = 0;

    QString windowsLength = "0";
    QString windowsStep = "0";

    QString choicedModelName;   // 模型名
    QString choicedModelPATH;   // 模型路径
    QString choicedModelSuffix; // 模型后缀
    QString modelStructXmlPath; // 模型结构文件路径
    QString modelStructImgPath; // 模型结构图片路径

    // 可视化进程
    QProcess *processVis;

    /**************** 以下同样的实现代码(为了实现两个可视化对比) ****************/
    // 选择的数据集、模型、样本信息

    QString choicedStage_2;           // 样本阶段, train or test or val
    QString choicedLabel_2;           // 样本标签
    QString choicedMatName_2;         // 样本名称
    QString choicedMatPATH_2;         // 样本路径
    QString choicedCamMethod_2;       // 可视化方法
    QString targetVisLayer_2;         // 可视化的目标层
    QString camImgsSavePath_2;        // 可视化结果保存路径
    
    std::map<std::string, std::string> choicedLayer_2;    // 选择的层级信息

    int choicedMatIndexBegin_2 = -1;
    int choicedMatIndexEnd_2 = -1;
    int maxMatIndex_2 = -1;
    int currMatIndex_2 = 0;

    // 可视化进程
    QProcess *processVis_2;
    /************************************************************************/

    QString condaEnvName;
    QString pythonApiPath;


    // 缩放图像组件
    std::map<QGraphicsView*, ImageWidget*> all_Images;     // 防止内存泄露
    void recvShowPicSignal(QPixmap image, QGraphicsView* graphicsView);

};


#endif // MODELCAMPAGE_H
