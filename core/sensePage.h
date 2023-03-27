#ifndef SENSEPAGE_H
#define SENSEPAGE_H

#include <vector>
#include <map>
#include <QObject>
#include <QDir>
#include <QButtonGroup>
#include <QGraphicsView>
#include <QGraphicsScene>
#include "ui_MainWindow.h"
#include "./lib/guiLogic/bashTerminal.h"
#include "./lib/guiLogic/datasetInfo.h"
#include "./lib/guiLogic/projectsInfo.h"
#include "./core/projectsWindow/chart.h"
#include "./lib/guiLogic/customWidget/imagewidget.h"
#include "./lib/guiLogic/tools/searchFolder.h"

class SenseSetPage:public QObject{
    Q_OBJECT
public:
    SenseSetPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo, ProjectsInfo *globalProjectInfo);
    ~SenseSetPage();

    std::map<std::string, QLabel*> attriLabelGroup;
    std::vector<QLabel*> imgGroup;
    std::vector<QLabel*> imgInfoGroup;
    std::vector<QLabel*> chartGroup;
    std::vector<QLabel*> chartInfoGroup;
    void refreshGlobalInfo();
    // 显示图片的最大数据索引

    QButtonGroup *BtnGroup_typeChoice = new QButtonGroup;



public slots:
    void confirmDataset(bool notDialog);
//    void saveDatasetAttri();

    void updateAttriLabel();
    void drawClassImage();
    void nextBatchChart();
    void saveDatasetNote();
    void saveModelNote();


private:
    Ui_MainWindow *ui;
    BashTerminal *terminal;

    DatasetInfo *datasetInfo;
    ProjectsInfo *projectsInfo;

    // 不同平台下文件夹搜索工具
    SearchFolder *dirTools = new SearchFolder();

    void minMatNum(int &minNum);
    void recvShowPicSignal(QPixmap image, QGraphicsView *graphicsView);
    std::map<QGraphicsView*, ImageWidget*> all_Images;     // 防止内存泄露
};
#endif // SENSEPAGE_H
