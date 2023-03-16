#ifndef PROJECTSDOCKH_H
#define PROJECTSDOCKH_H


#include <QObject>
#include <QMenu>
#include <QStandardItemModel>
#include <QFileSystemModel>
#include "ui_MainWindow.h"
#include "./lib/guiLogic/bashTerminal.h"
#include "./lib/guiLogic/projectsInfo.h"
#include "core/projectsWindow/chart.h"

#include "./lib/guiLogic/tools/searchFolder.h"
#include <mat.h>

class ProjectDock:public QObject{
    Q_OBJECT
public:
    ProjectDock(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, ProjectsInfo *globalProjectInfo);
    ~ProjectDock();

    std::map<std::string, QLabel*> attriLabelGroup;
    std::map<std::string, QTreeView*> projectTreeViewGroup;
    std::vector<QLabel*> chartGroup;
    std::vector<QLabel*> chartInfoGroup;

    void reloadTreeView();

    std::string leftSelType;
    std::string leftSelName;
    std::string rightSelType;
    std::string rightSelName;
// public slots:
//     void importDataset(std::string type);
//     void deleteDataset();

signals:
    void projectChanged();

private slots:
    // void handleTreeViewClick(const QModelIndex &index, const QPoint &pos, int mouseButton);
    void treeItemClicked(const QModelIndex &index);
    void onRequestMenu(const QPoint &pos);
    void onAction_DeleteProject();
    void onAction_ShotProject();
    void onAction_AddProject();
    void onAction_Expand();
    void onAction_Collapse();
//     void onActionExtractFea();

private:
    Ui_MainWindow *ui;
    BashTerminal *terminal;
    QModelIndex rightMsIndex;
    QModelIndex leftMsIndex;
    ProjectsInfo *projectsInfo;
    QString selectedMatFilePath = ""; 
    void addFilesToItem(QStandardItem *parentItem, const QString &path);
    QModelIndex findModelIndexByName(QStandardItem *item, const QString &name);
    std::string getPathByItemClicked(); 
    void drawExample();

    //保存上一个活动工程在treemodel的index和工程路径，方便作判断
    QModelIndex lastProjectIndex = QModelIndex();
    QString lastProjectPath = "";
    QString lastProjectDataType = "";
    // 不同平台下文件夹搜索工具
    SearchFolder *dirTools = new SearchFolder();
};

#endif // DATASETDOCK_H
