#ifndef PROJECTSDOCKH_H
#define PROJECTSDOCKH_H


#include <QObject>
#include <QMenu>
#include <QStandardItemModel>
#include <QFileSystemModel>
#include "ui_MainWindow.h"
#include "./uis/DialogNewProject.h"
#include "ui_DialogNewProject.h"
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
    QString workDir;
    void reloadTreeView();

    std::string leftSelType;
    std::string leftSelName;
    std::string rightSelType;
    std::string rightSelName;

    QString rightSelPath;
    // 新建工程文件路径名称
    QString projectNaming;
    // 新建的工程文件夹导入的四个数据集路径
    QMap<QString,QString> projectInputPath;
    // 新建工程文件夹的路径
    QString newProjectPath;



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
    void onAction_NewProject();
    void onAction_Expand();
    void onAction_Collapse();
    void onAction_Delete();


//     void onActionExtractFea();

private:
    Ui_MainWindow *ui;
    Ui_DialogNewProject *ui_newProject;
    DialogNewProject *newProject;
    BashTerminal *terminal;
    QModelIndex rightMsIndex;
    QModelIndex leftMsIndex;
    ProjectsInfo *projectsInfo;
    QString selectedMatFilePath = ""; 
    void addFilesToItem(QStandardItem *parentItem, const QString &path);
    QModelIndex findModelIndexByName(QStandardItem *item, const QString &name);
    std::string getPathByItemClicked(); 
    std::string getPathByRightClicked();
    void drawExample();

    //保存上一个活动工程在treemodel的index和工程路径，方便作判断
    QModelIndex lastProjectIndex = QModelIndex();
    QString lastProjectPath = "";
    QString lastProjectDataType = "";
    // 不同平台下文件夹搜索工具
    SearchFolder *dirTools = new SearchFolder();

    bool copyDir(const QString& srcPath, const QString& dstPath);
    QString makeNewProject(QString name, QMap<QString, QString> path);
    void makeProjectDock(QString projectName,QString projectPath);
    bool deleteDir(const QString& path);
    bool confirmProjectType(QString projectName);

};

#endif // DATASETDOCK_H
