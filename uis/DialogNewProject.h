#ifndef DIALOGNEWPROJECT_H
#define DIALOGNEWPROJECT_H

#include <QDialog>
#include <QFileDialog>
#include <QDir>
#include <QMessageBox>
namespace Ui {
class DialogNewProject;
}

class DialogNewProject : public QDialog
{
    Q_OBJECT

public:
    explicit DialogNewProject(QString *projectName,QMap<QString,QString> *projectPath,QWidget *parent = nullptr);
    ~DialogNewProject();

public slots:
    void on_pushButton_trainPath_clicked();
    void on_pushButton_valPath_clicked();
    void on_pushButton_testPath_clicked();
    void on_pushButton_unknown_clicked();

private:
    Ui::DialogNewProject *ui;
    QString *projectName;
    QMap<QString,QString> *projectPath;
    // 重写accept函数
    void accept();
};

#endif // DIALOGNEWPROJECT_H
