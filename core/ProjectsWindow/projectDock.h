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
#include <io.h>
#include <string>
#include <QDomDocument>
#include <QDomElement>
#include <QDomNode>
#include <QTextStream>
#include <QXmlStreamReader>
#include <QXmlStreamWriter>

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
    void makeProjectDock(QString projectName,QString projectPath);
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
    // 工程文件夹预览
    std::map<std::string, QLabel*> projectPreview;

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
    void onAction_modifyProject();
    void onAction_NewProject();
    void onAction_Expand();
    void onAction_Collapse();
    void onAction_Delete();
    void onAction_openInWindows();


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
    bool copyFile(const QString &srcFilePath, const QString &tgtFilePath);
    // bool copyDir(const QString &srcDirPath, const QString &tgtDirPath);
    void copyDir(QString src, QString dst);
    QString makeNewProject(QString name, QMap<QString, QString> path);
    void ProjectDockMessage(QString projectName,QString projectPath);
    void updateDatasetInfo(QString projectName,QString projectPath);
    bool deleteDir(const QString &strPath);
    bool removeDir(const QString &dirPath);
    bool confirmProjectType(QString projectPath);
    void renameFiles(const QString& path, const QString& oldName, const QString& newName);
    void updateXmlFile(QString filePath, QString oldName, QString newName);
};

#endif // DATASETDOCK_H
