#ifndef DIALOGNEWPROJECT_H
#define DIALOGNEWPROJECT_H
#include <QDialog>
#include <QFileDialog>
#include <QDir>
#include <QMessageBox>

#include <opencv2/opencv.hpp>
#include "./lib/guiLogic/bashTerminal.h"
#include "./lib/guiLogic/tools/searchFolder.h"
#include "./lib/guiLogic/projectsInfo.h"
#include "./lib/guiLogic/customWidget/imagewidget.h"

namespace Ui {
class DialogNewProject;
}

class DialogNewProject : public QDialog
{
    Q_OBJECT

public:
    explicit DialogNewProject(QString *projectName,QMap<QString,QString> *projectPath,bool newflag,QWidget *parent = nullptr);
    ~DialogNewProject();

public slots:
    void on_pushButton_trainPath_clicked();
    void on_pushButton_valPath_clicked();
    void on_pushButton_testPath_clicked();
    void on_pushButton_unknown_clicked();

private:
    Ui::DialogNewProject *ui;
    QString *projectName;
    bool newflag;
    QMap<QString,QString> *projectPath;
    // 重写accept函数
    void accept();
    QString oldName;
    SearchFolder *dirTools = new SearchFolder();
    bool checkClass(QMap<QString,QString> *projectPath);
};

#endif // DIALOGNEWPROJECT_H
