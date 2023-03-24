#pragma once
#include <QMainWindow>
#include <QProcess>
// 数据记录类
#include "./lib/guiLogic/datasetInfo.h"
#include "./lib/guiLogic/modelInfo.h"
#include "./lib/guiLogic/projectsInfo.h"
// 主页面类
#include "./core/sensePage.h"
#include "./core/modelEvalPage.h"
#include "./core/modelTrainPage.h"
#include "./core/monitorPage.h"
#include "./uis/DialogNewProject.h"
#include "./core/modelVisPage.h"
#include "./core/modelCAMPage.h"

//#include "./lib/guiLogic/modelEval.h"
// 悬浮窗部件类
#include "./core/projectsWindow/projectDock.h"
#include "./lib/guiLogic/bashTerminal.h"
// 界面美化类
#include "./conf/QRibbon/QRibbon.h"


#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    static const std::string slash="\\";
#else
    static const std::string slash="/";
#endif

namespace Ui{
    class MainWindow; 
};

class MainWindow: public QMainWindow{
	Q_OBJECT

    public:
        MainWindow(QWidget *parent = Q_NULLPTR);
        ~MainWindow();

        BashTerminal *terminal; // 自定义终端
//        ModelEval *modeleval; // 模型评估页面控制类
    public slots:
        void switchPage();      // 页面切换
        void refreshPages();    // 页面刷新
        void fullScreen();      // 全屏
        void aboutApp();      // aboutApp
        void aboutQt();      // aboutQt
        void showManual();      //文档信息
        
    private:
        Ui::MainWindow *ui; 
        Ui::DialogNewProject *ui_newProject;
        ProjectDock *projectDock;
//        DatasetDock *datasetDock;
//        ModelDock *modelDock;

        SenseSetPage *senseSetPage;
        // ModelChoicePage *modelChoicePage;
        ModelEvalPage *modelEvalPage;
        ModelTrainPage *modelTrainPage;
        MonitorPage *monitorPage;
        
        ModelVisPage *modelVisPage;
        ModelCAMPage *modelCAMPage;

        ProjectsInfo *globalProjectInfo;
        DatasetInfo *globalDatasetInfo;
        ModelInfo *globalModelInfo;

};
